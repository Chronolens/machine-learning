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
import boto3
from botocore.exceptions import NoCredentialsError


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
    def __init__(self, envs: EnvVars):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "ViT-B-16"
        self.pretrained_dataset = "datacomp_l_s1b_b8k"
        self.model, _, _ = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained_dataset)
        self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        
        
        self.envs = envs

        
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.envs.object_storage_endpoint,
            aws_access_key_id=self.envs.object_storage_access_key,
            aws_secret_access_key=self.envs.object_storage_secret_key,
            region_name=self.envs.object_storage_region
        )


    def generate_text_embedding(self, text):
        with torch.no_grad():
            text_tokenized = self.tokenizer([text]).to(self.device)
            text_features = self.model.encode_text(text_tokenized)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().flatten()


    def generate_presigned_url(self, preview_id: str):
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.envs.object_storage_bucket, 'Key': preview_id},
                ExpiresIn=86400  
            )
            return url
        except NoCredentialsError:
            logger.error("No credentials found for generating presigned URL.")
            return None
        except Exception as e:
            logger.error(f"Error generating presigned URL: {e}")
            return None


    def fetch_media_embeddings(self, db_conn, user_id):
        query = sql.SQL("""
            SELECT id, preview_id, clip_embeddings 
            FROM media 
            WHERE clip_embeddings IS NOT NULL AND user_id = %s
            ORDER BY created_at DESC
        """)

        try:
            with db_conn.cursor() as cursor:
                cursor.execute(query, (user_id,))
                results = cursor.fetchall()

                media_data = [(row[0], row[1], np.array(json.loads(row[2]))) for row in results]
                logger.info(f"Fetched {len(media_data)} media embeddings for user {user_id}.")
                return media_data
        except Exception as e:
            logger.error(f"Error fetching media embeddings for user {user_id}: {e}")
            return []


    def get_matching_media(self, text_embedding, media_data, page, pagesize):
        try:
            matching_data = []

            for media_id, preview_id, media_embedding in media_data:
                similarity = 1 - cosine(text_embedding, media_embedding)

                if similarity > 0.3:
                    presigned_url = self.generate_presigned_url(preview_id)
                    if presigned_url:
                        matching_data.append({
                            "media_id": media_id,
                            "preview_url": presigned_url
                        })

            offset = (page - 1) * pagesize
            paged_data = matching_data[offset:offset + pagesize]

            if not paged_data:
                logging.info(f"No matching media found for page {page}.")

            return paged_data
        except Exception as e:
            logging.error(f"Error comparing embeddings: {e}")
            return []



async def message_handler(msg, clip_text_processor, db_conn):
    logging.info(f"Received message: {msg.data.decode()}")
    try:
        message = json.loads(msg.data.decode())
        user_id = message.get("user_id")
        query = message.get("query")
        page = message.get("page", 1)
        pagesize = message.get("page_size", 10)

        if not query:
            raise ValueError("Received empty text message.")

        text_embedding = clip_text_processor.generate_text_embedding(query)

        media_data = clip_text_processor.fetch_media_embeddings(db_conn, user_id)

        matching_media = clip_text_processor.get_matching_media(text_embedding, media_data, page, pagesize)

        response_payload = json.dumps(matching_media)

        logging.info(f"Responding with payload: {response_payload}")

        await msg.respond(response_payload.encode('utf-8'))
        logging.info(f"Responded with {len(matching_media)} matches.")
    except Exception as e:
        logging.error(f"Error processing message: {e}")
        error_response = {"error": str(e)}
        await msg.respond(json.dumps(error_response).encode('utf-8'))
    finally:
        await msg.ack()





def connect_to_database(envs: EnvVars):
    url = f"postgresql://{envs.database_user}:{envs.database_password}@" \
          f"{envs.database_host}:{envs.database_port}/{envs.database_name}"
    try:
        conn = psycopg.connect(url)
        logger.info("Connected to the database successfully.")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to the database: {e}")
        raise



async def main():
    load_dotenv()
    envs = EnvVars()
    db_conn = connect_to_database(envs)
    clip_text_processor = ClipTextProcessor(envs)

    try:
        nc = await nats.connect(envs.nats_endpoint)
        js = nc.jetstream()

        try:
            stream_info = await js.stream_info("machine-learning")
            logging.info(f"Stream info: {stream_info}")
        except Exception as e:
            logging.error("Stream 'machine-learning' not found or misconfigured. Attempting to create it.")
            await js.add_stream(name="machine-learning", subjects=["image-process","clip-process"], retention="workqueue")

        
        sub = await js.subscribe("clip-process", cb=lambda msg: asyncio.create_task(message_handler(msg, clip_text_processor, db_conn)))
        logging.info("Subscribed to 'clip-process' stream.")

        
        while True:
            await asyncio.sleep(5)

    except Exception as e:
        logging.error(f"Error in main: {e}")
    finally:
        db_conn.close()



if __name__ == '__main__':
    asyncio.run(main())
