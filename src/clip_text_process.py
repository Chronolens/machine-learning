import os
import asyncio
import logging
import nats
import torch
import open_clip
import json
from dotenv import load_dotenv
import psycopg
from scipy.spatial.distance import cosine
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("face_process_service")

class EnvVars:
    def __init__(self):
        self.nats_endpoint = os.getenv("NATS_ENDPOINT")
        self.object_storage_endpoint = os.getenv("OBJECT_STORAGE_ENDPOINT")
        self.object_storage_bucket = os.getenv("OBJECT_STORAGE_BUCKET")
        self.object_storage_region = os.getenv("OBJECT_STORAGE_REGION")
        self.object_storage_access_key = os.getenv("OBJECT_STORAGE_ACCESS_KEY")
        self.object_storage_secret_key = os.getenv("OBJECT_STORAGE_SECRET_KEY")

        self.database_host = os.getenv("DATABASE_HOST")
        self.database_user = os.getenv("DATABASE_USERNAME")
        self.database_password = os.getenv("DATABASE_PASSWORD")
        self.database_port = os.getenv("DATABASE_PORT")
        self.database_name = os.getenv("DATABASE_NAME")


class ClipTextProcessor:
    def __init__(self, envs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "ViT-B-16"
        self.pretrained_dataset = "datacomp_l_s1b_b8k"

        self.model, _, _ = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained_dataset)
        self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(self.model_name)


    def generate_text_embedding(self, text):
        with torch.no_grad():
            text_tokenized = self.tokenizer([text]).to(self.device)
            text_features = self.model.encode_text(text_tokenized)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().flatten()


    def fetch_media_embeddings(self, db_conn):
        query = "SELECT id, clip_embeddings FROM media WHERE clip_embeddings IS NOT NULL"

        try:
            with db_conn.cursor() as cursor:
                cursor.execute(query)
                results = cursor.fetchall()
                media_data = [(row[0], np.array(json.loads(row[1]))) for row in results]
            logger.info("Fetched media embeddings from the database.")
            return media_data
        except Exception as e:
            logger.error(f"Error fetching media embeddings: {e}")
            return []


    def compare_and_save_matches(self, text_embedding, media_data, output_file="matches.txt"):
        try:
            matching_ids = [
                media_id for media_id, media_embedding in media_data
                if 1 - cosine(text_embedding, media_embedding) > 0.3
            ]

            with open(output_file, "w") as f:
                for media_id in matching_ids:
                    f.write(f"{media_id}\n")
            logger.info(f"Saved {len(matching_ids)} matching media_ids to {output_file}")
        except Exception as e:
            logger.error(f"Error comparing embeddings or saving matches: {e}")



async def message_handler(msg, clip_text_processor, db_conn):
    subject = msg.subject
    data = msg.data.decode()
    logger.info(f"Received a message on '{subject}': {data}")

    try:
        payload = json.loads(data)
        text = payload.get("text")
        if not text:
            raise ValueError("Message must contain 'text'")

        text_embedding = clip_text_processor.generate_text_embedding(text)

        media_data = clip_text_processor.fetch_media_embeddings(db_conn)
        
        clip_text_processor.compare_and_save_matches(text_embedding, media_data)

    except Exception as e:
        logger.error(f"Error handling message: {e}")

    await msg.ack()


async def main():
    load_dotenv()
    envs = EnvVars()

    db_conn = connect_to_database(envs)
    clip_text_processor = ClipTextProcessor(envs)

    try:
        nc = await nats.connect(envs.nats_endpoint)
        js = nc.jetstream()

        try:
            await js.stream_info("chronolens")
        except Exception:
            await js.add_stream(name="chronolens", subjects=["clip_process"], retention="workqueue")

        async def callback(msg):
            await message_handler(msg, clip_text_processor, db_conn)

        await js.subscribe("clip_process", cb=callback)
        while True:
            await asyncio.sleep(5)
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        db_conn.close()


def connect_to_database(envs: EnvVars):
    url = f"postgresql://{envs.database_user}:{envs.database_password}@" \
          f"{envs.database_host}:{envs.database_port}/{envs.database_name}"
    try:
        conn = psycopg.connect(url)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
        logger.info("Connected to the database and registered vector types.")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to the database: {e}")
        raise


if __name__ == '__main__':
    asyncio.run(main())
