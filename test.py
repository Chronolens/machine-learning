import os
import asyncio
import logging
import boto3
import psycopg
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from face_recognition import FaceRecognition 
from nats.aio.msg import Msg
import nats
from psycopg import sql
from pgvecto_rs.psycopg import register_vector

BATCH_SIZE = 5
DOWNLOAD_IMAGES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_downloaded_images')
os.makedirs(DOWNLOAD_IMAGES_PATH, exist_ok=True)

class EnvVars:
    def __init__(self):
        self.nats_endpoint = "10.0.0.50:4222"
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
            logging.info("Connected to the database successfully and registered vector types.")
            return conn
        except Exception as e:
            logging.error(f"Error connecting to the database: {e}")
            raise

    async def handle_request(self, messages: list[Msg], bucket):
        logging.info(f"Processing batch of {len(messages)} messages")
        image_paths = []

        for msg in messages:
            try:
                uuid = msg.data.decode()
                logging.info(f"Processing image with UUID: {uuid}")
                image_path = await self.fetch_image_from_s3(uuid, bucket)
                if image_path:
                    image_paths.append(image_path)
                await msg.ack()
            except Exception as e:
                logging.error(f"Error processing message {msg.data.decode()}: {e}")
                continue

        if image_paths:
            face_data = self.face_recognition.process_images_in_directory(DOWNLOAD_IMAGES_PATH)
            logging.info(f"Processed face data for {len(image_paths)} images.")
            self.insert_face_data_to_db(face_data)
            # self.cleanup_temp_images(image_paths)
        else:
            logging.warning("No images found to process.")

    def insert_face_data_to_db(self, face_data):
        with self.connection.cursor() as cursor:
            for file_path, embedding, coordinates, face_id in face_data:

                embedding_str = f"[{', '.join(map(str, embedding))}]"
                coordinates_str = f"[{', '.join(map(str, coordinates))}]"
                media_id = os.path.basename(file_path).split('.')[0]

                insert_query = """
                INSERT INTO face_data (media_id, embedding, coordinates)
                VALUES (%s, %s::vector, %s::vector)
                RETURNING id;
                """
                params = (media_id, embedding_str, coordinates_str)

                try:
                    cursor.execute(sql.SQL(insert_query), params)
                    inserted_id = cursor.fetchone()[0]
                    logging.info(f"Inserted face data for media_id {media_id} with id {inserted_id}.")
                except Exception as e:
                    self.connection.rollback()
                    logging.error(f"Error inserting face data for media_id {media_id}: {e}")
                    continue
                self.connection.commit()

    async def fetch_image_from_s3(self, uuid, bucket):
        try:
            s3_object = bucket.Object(uuid)
            content_type = s3_object.content_type
            extension = self.get_extension_from_content_type(content_type)
            local_image_path = os.path.join(DOWNLOAD_IMAGES_PATH, f"{uuid}{extension}")
            bucket.download_file(uuid, local_image_path)
            logging.info(f"Image {uuid} downloaded to {local_image_path}")
            return local_image_path
        except ClientError as e:
            logging.error(f"Error fetching image {uuid} from S3: {e}")
            return None

    def get_extension_from_content_type(self, content_type):
        extensions = {
            'image/jpeg': '.jpg',
            'image/png': '.png'
        }
        return extensions.get(content_type, '.jpg')

    def cleanup_temp_images(self, image_paths):
        for image_path in image_paths:
            try:
                os.remove(image_path)
                logging.info(f"Deleted temporary image: {image_path}")
            except OSError as e:
                logging.error(f"Error deleting temporary image {image_path}: {e}")

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
            logging.info(f"Bucket {envs.object_storage_bucket} created successfully.")
        return bucket
    except ClientError as e:
        logging.error(f"Error setting up S3 bucket: {e}")
        raise

async def main():
    load_dotenv()
    envs = EnvVars()
    image_processor = ImageProcessor(envs)
    bucket = await setup_bucket(envs)
    logging.info("Bucket setup complete.")

    nc = await nats.connect(envs.nats_endpoint)
    js = nc.jetstream()
    await js.add_stream(name="chronolens", subjects=["machine-learning"])
    sub = await js.subscribe("machine-learning")

    message_batch = []

    async def process_messages():
        while True:
            msg = await sub.next_msg(timeout=50)
            message_batch.append(msg)
            logging.info(f"Received message: {msg.data.decode()}")

            if len(message_batch) == BATCH_SIZE:
                await image_processor.handle_request(message_batch, bucket)
                message_batch.clear()

    await process_messages()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
