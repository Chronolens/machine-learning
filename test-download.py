import os
import logging
import boto3
import psycopg
from psycopg import sql
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from face_recognition import FaceRecognition
from pgvecto_rs.psycopg import register_vector

DOWNLOAD_IMAGES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_downloaded_images')
os.makedirs(DOWNLOAD_IMAGES_PATH, exist_ok=True)


class EnvVars:
    def __init__(self):
        self.database_host = os.getenv("DATABASE_HOST")
        self.database_user = os.getenv("DATABASE_USERNAME")
        self.database_password = os.getenv("DATABASE_PASSWORD")
        self.database_port = os.getenv("DATABASE_PORT")
        self.database_name = os.getenv("DATABASE_NAME")
        self.object_storage_endpoint = os.getenv("OBJECT_STORAGE_ENDPOINT")
        self.object_storage_bucket = os.getenv("OBJECT_STORAGE_BUCKET")
        self.object_storage_region = os.getenv("OBJECT_STORAGE_REGION")
        self.object_storage_access_key = os.getenv("OBJECT_STORAGE_ACCESS_KEY")
        self.object_storage_secret_key = os.getenv("OBJECT_STORAGE_SECRET_KEY")


class ImageDownloaderAndProcessor:
    def __init__(self):
        self.envs = EnvVars()
        self.face_recognition = FaceRecognition()
        os.makedirs(DOWNLOAD_IMAGES_PATH, exist_ok=True)
        self.connection = self.connect_to_database()

    def connect_to_database(self):
        url = f"postgresql://{self.envs.database_user}:{self.envs.database_password}@{self.envs.database_host}:{self.envs.database_port}/{self.envs.database_name}"
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

    def fetch_images_from_s3(self, bucket):
        logging.info(f"Fetching images from bucket: {bucket.name}")
        try:
            image_paths = []
            for obj in bucket.objects.filter(Prefix=""):
                uuid = obj.key
                logging.info(f"Found image with UUID: {uuid}")
                image_path = self.download_image_from_s3(uuid, bucket)
                if image_path:
                    image_paths.append(image_path)
            return image_paths
        except ClientError as e:
            logging.error(f"Error fetching images from S3: {e}")
            return []

    def download_image_from_s3(self, uuid, bucket):
        try:
            clean_uuid = uuid.replace("/", "_")
            s3_object = bucket.Object(uuid)
            content_type = s3_object.content_type
            extension = self.get_extension_from_content_type(content_type)

            local_image_path = os.path.join(DOWNLOAD_IMAGES_PATH, f"{clean_uuid}{extension}")
            os.makedirs(os.path.dirname(local_image_path), exist_ok=True)

            bucket.download_file(uuid, local_image_path)
            return local_image_path
        except ClientError as e:
            logging.error(f"Error downloading image {uuid} from S3: {e}")
            return None

    def get_extension_from_content_type(self, content_type):
        extensions = {
            'image/jpeg': '.jpg',
            'image/png': '.png',
            'image/heif': '.heif',
            'image/heic': '.heic',
            'image/bmp': '.bmp',
            'image/tiff': '.tiff',
            'image/gif': '.gif',
            'image/webp': '.webp'
        }
        return extensions.get(content_type, '.jpg')

    def process_downloaded_images(self, image_paths):
        face_data = []
        for image_path in image_paths:
            embeddings = self.face_recognition.extract_face_embeddings(image_path)
            if embeddings:
                face_data.extend(embeddings)
        if face_data:
            logging.info(f"Processed face data for {len(image_paths)} images.")
            self.insert_face_data_to_db(face_data)
        else:
            logging.warning("No faces found in downloaded images.")


    def insert_face_data_to_db(self, face_data):
        with self.connection.cursor() as cursor:
            for file_path, embedding, coordinates, face_id in face_data:
                # Convert embeddings and coordinates to PostgreSQL vector-compatible strings
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
                    # Execute the insert statement
                    cursor.execute(sql.SQL(insert_query), params)
                    inserted_id = cursor.fetchone()[0]
                    logging.info(f"Inserted face data for media_id {media_id} with id {inserted_id}.")
                except Exception as e:
                    self.connection.rollback()  # Roll back if error occurs in transaction
                    logging.error(f"Error inserting face data for media_id {media_id}: {e}")
                    continue  # Skip to the next entry if insertion fails

                self.connection.commit()

    def cleanup_temp_images(self, image_paths):
        for image_path in image_paths:
            try:
                os.remove(image_path)
                logging.info(f"Deleted temporary image: {image_path}")
            except OSError as e:
                logging.error(f"Error deleting temporary image {image_path}: {e}")


def setup_bucket(envs: EnvVars):
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
            raise ValueError(f"Bucket {envs.object_storage_bucket} does not exist.")
        return bucket
    except ClientError as e:
        logging.error(f"Error setting up S3 bucket: {e}")
        raise


def main():
    logging.basicConfig(level=logging.INFO)
    load_dotenv()
    envs = EnvVars()
    bucket = setup_bucket(envs)
    image_downloader_processor = ImageDownloaderAndProcessor()
    image_paths = image_downloader_processor.fetch_images_from_s3(bucket)
    image_downloader_processor.process_downloaded_images(image_paths)
    # image_downloader_processor.cleanup_temp_images(image_paths)


if __name__ == '__main__':
    main()
