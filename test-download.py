import os
import logging
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from face_recognition import FaceRecognition  # Importing the class for face recognition

# Load environment variables from .env file
load_dotenv()

# Define global variables
DOWNLOAD_IMAGES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_downloaded_images')

# Create the directory to download images
os.makedirs(DOWNLOAD_IMAGES_PATH, exist_ok=True)

class EnvVars:
    def __init__(self):
        self.object_storage_endpoint = os.getenv("OBJECT_STORAGE_ENDPOINT")
        self.object_storage_bucket = os.getenv("OBJECT_STORAGE_BUCKET")
        self.object_storage_region = os.getenv("OBJECT_STORAGE_REGION")
        self.object_storage_access_key = os.getenv("OBJECT_STORAGE_ACCESS_KEY")
        self.object_storage_secret_key = os.getenv("OBJECT_STORAGE_SECRET_KEY")


class ImageDownloaderAndProcessor:
    def __init__(self):
        self.face_recognition = FaceRecognition()  # Assuming FaceRecognition has methods to process images
        os.makedirs(DOWNLOAD_IMAGES_PATH, exist_ok=True)

    def fetch_images_from_s3(self, bucket):
        logging.info(f"Fetching images from bucket: {bucket.name}")
        try:
            image_paths = []
            for obj in bucket.objects.all():
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
            # Clean the UUID to remove any unwanted characters
            clean_uuid = uuid.replace("/", "_")  # Replace any forward slashes with underscores
            s3_object = bucket.Object(uuid)
            content_type = s3_object.content_type
            extension = self.get_extension_from_content_type(content_type)
            
            # Create a valid file path and ensure directories exist
            local_image_path = os.path.join(DOWNLOAD_IMAGES_PATH, f"{clean_uuid}{extension}")
            os.makedirs(os.path.dirname(local_image_path), exist_ok=True)  # Ensure directory exists
            
            # Download the file
            bucket.download_file(uuid, local_image_path)
            logging.info(f"Image {uuid} downloaded to {local_image_path}")
            return local_image_path
        except ClientError as e:
            logging.error(f"Error downloading image {uuid} from S3: {e}")
            return None

    def get_extension_from_content_type(self, content_type):
        if content_type == 'image/jpeg':
            return '.jpg'
        elif content_type == 'image/png':
            return '.png'
        else:
            logging.warning(f"Unknown content type: {content_type}. Defaulting to .jpg")
            return '.jpg'


    def process_downloaded_images(self, image_paths):
        if image_paths:
            face_data = self.face_recognition.process_images_in_directory(DOWNLOAD_IMAGES_PATH)
            logging.info(f"Processed face data for {len(image_paths)} images.")
        else:
            logging.warning("No images found to process.")

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
        print(f"Checking if bucket {envs.object_storage_bucket} exists...")
        if not bucket.creation_date:
            raise ValueError(f"Bucket {envs.object_storage_bucket} does not exist.")
        return bucket
    except ClientError as e:
        logging.error(f"Error setting up S3 bucket: {e}")
        raise


def main():
    logging.basicConfig(level=logging.INFO)

    # Load environment variables
    envs = EnvVars()

    # Set up the S3 bucket
    bucket = setup_bucket(envs)

    # Initialize the image downloader and processor
    image_downloader_processor = ImageDownloaderAndProcessor()

    # Fetch and download all images from the bucket
    image_paths = image_downloader_processor.fetch_images_from_s3(bucket)

    # Process the downloaded images
    image_downloader_processor.process_downloaded_images(image_paths)



if __name__ == '__main__':
    main()
