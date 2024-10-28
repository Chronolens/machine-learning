import os
import asyncio
import logging

from face_recognition import FaceRecognition 
from nats.aio.msg import Msg
import nats

from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError

BATCH_SIZE = 1

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

class ImageProcessor:
    def __init__(self):
        self.face_recognition = FaceRecognition() 
        os.makedirs(DOWNLOAD_IMAGES_PATH, exist_ok=True)

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
            
            for file_path, face_idx, embedding, (center_x, center_y) in face_data:
                file_name = os.path.basename(file_path)  
                embedding_shape = embedding.shape
                logging.info(f"File: {file_name}, Face Index: {face_idx}, Embedding Shape: {embedding_shape}, Coordinates: ({center_x}, {center_y})")
            
            # testing only
            csv_file_path = os.path.join(DOWNLOAD_IMAGES_PATH, 'face_data.csv')
            self.face_recognition.save_to_csv(face_data, csv_file=csv_file_path)
            logging.info(f"Face data saved to {csv_file_path}")
            
            # self.cleanup_temp_images(image_paths)
        else:
            logging.warning("No images found to process.")

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
        if content_type == 'image/jpeg':
            return '.jpg'
        elif content_type == 'image/png':
            return '.png'
        else:
            logging.warning(f"Unknown content type: {content_type}. Defaulting to .jpg")
            return '.jpg'

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
        print(f"Checking if bucket {envs.object_storage_bucket} exists...")
        if not bucket.creation_date:
            s3.create_bucket(Bucket=envs.object_storage_bucket)
            print(f"Bucket {envs.object_storage_bucket} created successfully.")
        return bucket
    except ClientError as e:
        logging.error(f"Error setting up S3 bucket: {e}")
        raise

async def main():

    load_dotenv()

    envs = EnvVars()

    image_processor = ImageProcessor()

    bucket = await setup_bucket(envs)

    print("Bucket setup complete.")

    nc = await nats.connect(envs.nats_endpoint)
    js = nc.jetstream()

    await js.add_stream(name="chronolens", subjects=["machine-learning"])

    print(f"Stream 'chronolens' with subject 'machine-learning' created.")

    sub = await js.subscribe("machine-learning")

    message_batch = []

    async def process_messages():
        while True:
            msg = await sub.next_msg(timeout=50)
            message_batch.append(msg)
            print(f"Received message: {msg.data.decode()}")

            if len(message_batch) == BATCH_SIZE:
                await image_processor.handle_request(message_batch, bucket)
                message_batch.clear()
            msg.ack()

    await process_messages()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())



#TODO: FIX ACK NOT REMOVING MESSAGES FROM QUEUE 
#TODO: STORE EMBEDDINGS IN DATABASE
#TODO: FIX NEXT MSG TIMEOUT DISCONNECT ISSUE
#TODO: GIVEN AN IMAGE ID, COMPARE IT WITH ALL THE IMAGES WITH EMDEDINGS IN THE DATABASE