import os
import asyncio
import logging
import boto3
import psycopg
import nats

from psycopg import sql
from dotenv import load_dotenv
from nats.aio.msg import Msg
from botocore.exceptions import ClientError
from face_recognition import FaceRecognition
from pgvecto_rs.psycopg import register_vector

logger = logging.getLogger("face_process_service")
logger.setLevel(logging.INFO)

BATCH_SIZE = 1
DOWNLOAD_IMAGES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_downloaded_images')
os.makedirs(DOWNLOAD_IMAGES_PATH, exist_ok=True)

class EnvVars:
    def __init__(self):
        # Retrieve environment variables
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

class ImageProcessor:
    def __init__(self, envs: EnvVars):
        self.envs = envs
        self.face_recognition = FaceRecognition()
        self.connection = self.connect_to_database()
        os.makedirs(DOWNLOAD_IMAGES_PATH, exist_ok=True)

    def connect_to_database(self):
        
        url = f"postgresql://{self.envs.database_user}:{self.envs.database_password}@" \
              f"{self.envs.database_host}:{self.envs.database_port}/{self.envs.database_name}"
        try:
            conn = psycopg.connect(url)
            with conn.cursor() as cur:
                cur.execute('CREATE EXTENSION IF NOT EXISTS vectors')
            register_vector(conn)
            logger.info("Connected to the database successfully and registered vector types.")
            return conn
        except Exception as e:
            logger.error(f"Error connecting to the database: {e}")
            raise

    async def handle_request(self, message: Msg, bucket):
        uuid = message.data.decode()
        face_data = []

        image_path = await self.fetch_image_from_s3(uuid, bucket)
        if image_path:
            face_data.extend(self.face_recognition.extract_face_embeddings(image_path))
        
        self.insert_face_data_to_db(face_data)
        os.remove(image_path)

    def insert_face_data_to_db(self, face_data):
        with self.connection.cursor() as cursor:
            for file_path, embedding, bb_coordinates, _ in face_data:
                embedding_str = f"[{', '.join(map(str, embedding))}]"

                bounding_box_str = f"[{', '.join(map(str, bb_coordinates))}]"

                media_id = os.path.basename(file_path).split('.')[0]

                insert_query = """
                INSERT INTO media_face (media_id, embedding, face_bounding_box)
                VALUES (%s, %s::vector, %s::vector)
                RETURNING id;
                """
                params = (media_id, embedding_str, bounding_box_str)

                try:
                    cursor.execute(sql.SQL(insert_query), params)
                    inserted_id = cursor.fetchone()[0]
                    logger.info(f"Inserted face data for media_id {media_id} with id {inserted_id}.")
                except Exception as e:
                    self.connection.rollback()
                    logger.error(f"Error inserting face data for media_id {media_id}: {e}")
                    continue
                self.connection.commit()

    async def fetch_image_from_s3(self, uuid, bucket):
        try:
            s3_object = bucket.Object(uuid)
            content_type = s3_object.content_type
            extension = self.get_extension_from_content_type(content_type)
            local_image_path = os.path.join(DOWNLOAD_IMAGES_PATH, f"{uuid}{extension}")
            bucket.download_file(uuid, local_image_path)
            logger.info(f"Image {uuid} downloaded to {local_image_path}")
            return local_image_path
        except ClientError as e:
            logger.error(f"Error fetching image {uuid} from S3: {e}")
            return None

    def get_extension_from_content_type(self, content_type):
        extensions = {
            'image/jpeg': '.jpg',
            'image/png': '.png'
        }
        return extensions.get(content_type, '.jpg')


async def setup_bucket(envs: EnvVars):
    try:
        s3 = boto3.resource(
            's3',
            region_name=envs.object_storage_region,
            endpoint_url=envs.object_storage_endpoint,
            aws_access_key_id=envs.object_storage_access_key,
            aws_secret_access_key=envs.object_storage_secret_key
        )
        bucket = s3.Bucket(envs.object_storage_bucket)
        if not bucket.creation_date:
            s3.create_bucket(Bucket=envs.object_storage_bucket)
            logger.info(f"Bucket {envs.object_storage_bucket} created successfully.")
        return bucket
    except ClientError as e:
        logger.error(f"Error setting up S3 bucket: {e}")
        raise


async def message_handler(msg: Msg, image_processor: ImageProcessor, bucket):

    logging.info(f"Received message: {msg.data.decode()}")
    await image_processor.handle_request(msg, bucket)
    await msg.ack()

async def main():
    load_dotenv()
    envs = EnvVars()
    bucket = await setup_bucket(envs)
    logging.info("Bucket setup complete.")

    image_processor = ImageProcessor(envs)

    try:
        nc = await nats.connect(envs.nats_endpoint)
        js = nc.jetstream()

        try:
            stream_info = await js.stream_info("chronolens")
            logging.info(f"Stream info: {stream_info}")
        except Exception as e:
            logging.error("Stream 'chronolens' not found or misconfigured. Attempting to create it.")
            await js.add_stream(name="chronolens", subjects=["machine-learning"], retention="workqueue")

        sub = await js.subscribe("machine-learning", cb=lambda msg: asyncio.create_task(message_handler(msg, image_processor, bucket)))
        logging.info("Subscribed to 'machine-learning'")

        while True:
            await asyncio.sleep(5)

    except Exception as conn_error:
        logging.error(f"Failed to connect to NATS: {conn_error}")



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())


# TODO: Insert bounding box as well as coordinates in the database