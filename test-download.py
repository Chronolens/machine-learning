import os
import logging
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from face_recognition import FaceRecognition 


DOWNLOAD_IMAGES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_downloaded_images')

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
        self.face_recognition = FaceRecognition()  
        os.makedirs(DOWNLOAD_IMAGES_PATH, exist_ok=True)

    def fetch_images_from_s3(self, bucket):
        logging.info(f"Fetching images from bucket: {bucket.name}")
        try:
            image_paths = []
            for obj in bucket.objects.filter(Prefix=""): #trying to filter only the full images
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
            
            # Download the file
            bucket.download_file(uuid, local_image_path)
            return local_image_path
        except ClientError as e:
            logging.error(f"Error downloading image {uuid} from S3: {e}")
            return None

    def get_extension_from_content_type(self, content_type):

        if content_type == 'image/jpeg':
            return '.jpg'
        elif content_type == 'image/png':
            return '.png'
        elif content_type in ['image/heif', 'image/heic']:
            return '.heif'
        elif content_type == 'image/bmp':
            return '.bmp'
        elif content_type == 'image/tiff':
            return '.tiff'
        elif content_type == 'image/gif':
            return '.gif'
        elif content_type == 'image/webp':
            return '.webp'
        

        elif content_type == 'video/mp4':
            return '.mp4'
        elif content_type == 'video/x-msvideo':
            return '.avi'
        elif content_type == 'video/x-matroska': 
            return '.mkv'
        elif content_type == 'video/webm':
            return '.webm'
        elif content_type == 'video/quicktime':
            return '.mov'
        

        else:
            logging.warning(f"Unknown content type: {content_type}. Defaulting to .jpg")
            return '.jpg'


    def process_downloaded_images(self, image_paths):
        face_data = []

        for image_path in image_paths:
            embeddings = self.face_recognition.extract_face_embeddings(image_path)
            if embeddings:
                face_data.extend(embeddings)

        if face_data:
            logging.info(f"Processed face data for {len(image_paths)} images.")
            self.face_recognition.save_to_csv(face_data) 
            self.face_recognition.compare_all_faces(face_data) # Hardcoded comparing between vectors for now, TODO: Implement clustering
        else:
            logging.warning("No faces found in downloaded images.")

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

    load_dotenv()
    envs = EnvVars()
    bucket = setup_bucket(envs)

    image_downloader_processor = ImageDownloaderAndProcessor()

    image_paths = image_downloader_processor.fetch_images_from_s3(bucket)

    image_downloader_processor.process_downloaded_images(image_paths)

    # Cleanup the images
    # image_downloader_processor.cleanup_temp_images(image_paths)


if __name__ == '__main__':
    main()
