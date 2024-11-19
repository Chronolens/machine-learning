import os
import asyncio
import logging
import psycopg
import nats
import torch
import open_clip
import json
import numpy as np
from dotenv import load_dotenv
from psycopg import sql
from scipy.spatial.distance import cosine


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("clip_text_processor_service")

MATCH_OUTPUT_FOLDER = "./matches"
os.makedirs(MATCH_OUTPUT_FOLDER, exist_ok=True)

class EnvVars:
    def __init__(self):
        self.nats_endpoint = os.getenv("NATS_ENDPOINT")
        self.database_host = os.getenv("DATABASE_HOST")
        self.database_user = os.getenv("DATABASE_USERNAME")
        self.database_password = os.getenv("DATABASE_PASSWORD")
        self.database_port = os.getenv("DATABASE_PORT")
        self.database_name = os.getenv("DATABASE_NAME")


class ClipTextProcessor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "ViT-B-16"
        self.pretrained_dataset = "datacomp_l_s1b_b8k"
        self.model, _, _ = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained_dataset)
        self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(self.model_name)

    def generate_text_embedding(self, text):
        """Generate a text embedding using the CLIP model."""
        with torch.no_grad():
            text_tokenized = self.tokenizer([text]).to(self.device)
            text_features = self.model.encode_text(text_tokenized)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().flatten()

    def fetch_media_embeddings(self, db_conn):
        """Retrieve media embeddings from the database."""
        query = "SELECT id, clip_embeddings FROM media WHERE clip_embeddings IS NOT NULL"
        try:
            with db_conn.cursor() as cursor:
                cursor.execute(query)
                results = cursor.fetchall()
                media_data = [(row[0], np.array(json.loads(row[1]))) for row in results]
                logger.info(f"Fetched {len(media_data)} media embeddings from the database.")
                return media_data
        except Exception as e:
            logger.error(f"Error fetching media embeddings: {e}")
            return []

    def compare_and_save_matches_to_file(self, text, text_embedding, media_data):
        """Compare text embedding to media embeddings and save matches to a file."""
        try:
            matching_ids = [
                media_id for media_id, media_embedding in media_data
                if 1 - cosine(text_embedding, media_embedding) > 0.3
            ]
            if not matching_ids:
                logger.info("No matching media found.")
                return

            
            match_file = os.path.join(MATCH_OUTPUT_FOLDER, f"matches_{text[:20].replace(' ', '_')}.json")
            match_data = {
                "text": text,
                "matching_media_ids": matching_ids
            }
            with open(match_file, "w") as f:
                json.dump(match_data, f, indent=4)
            logger.info(f"Saved {len(matching_ids)} matches to {match_file}.")
        except Exception as e:
            logger.error(f"Error comparing embeddings or saving matches to file: {e}")



def connect_to_database(envs: EnvVars):
    """Establish connection to the database."""
    url = f"postgresql://{envs.database_user}:{envs.database_password}@" \
          f"{envs.database_host}:{envs.database_port}/{envs.database_name}"
    try:
        conn = psycopg.connect(url)
        logger.info("Connected to the database successfully.")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to the database: {e}")
        raise


async def message_handler(msg, clip_text_processor, db_conn):
    """Handle incoming NATS messages."""
    logging.info(f"Received message: {msg.data.decode()}")
    try:
        
        text = msg.data.decode().strip()  

        if not text:
            raise ValueError("Received empty text message.")

        
        text_embedding = clip_text_processor.generate_text_embedding(text)
        media_data = clip_text_processor.fetch_media_embeddings(db_conn)
        clip_text_processor.compare_and_save_matches_to_file(text, text_embedding, media_data)
    except Exception as e:
        logging.error(f"Error processing message: {e}")
    finally:
        await msg.ack()


async def main():
    """Main function to set up services and event loop."""
    load_dotenv()
    envs = EnvVars()
    db_conn = connect_to_database(envs)
    clip_text_processor = ClipTextProcessor()

    try:
        nc = await nats.connect(envs.nats_endpoint)
        js = nc.jetstream()

        try:
            stream_info = await js.stream_info("chronolens")
            logging.info(f"Stream info: {stream_info}")
        except Exception as e:
            logging.error("Stream 'chronolens' not found or misconfigured. Attempting to create it.")
            await js.add_stream(name="chronolens", subjects=["clip_process"], retention="workqueue")

        
        sub = await js.subscribe("clip_process", cb=lambda msg: asyncio.create_task(message_handler(msg, clip_text_processor, db_conn)))
        logging.info("Subscribed to 'clip_process' stream.")

        
        while True:
            await asyncio.sleep(5)

    except Exception as e:
        logging.error(f"Error in main: {e}")
    finally:
        db_conn.close()



if __name__ == '__main__':
    asyncio.run(main())
