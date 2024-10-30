import os
import csv
from local_testing.face_recognition_toCSV import FaceRecognition
import time
import logging

photo_path = "C:\\Users\\Despacito4\\Desktop\\test_images\\large_dataset"


def main():
    
    logging.basicConfig(level=logging.INFO)

    start_time = time.time()

    model = FaceRecognition()
    model.process_images_in_directory(photo_path, "large_dataset_face_data.csv")

    end_time = time.time()
    logging.info(f"Time taken: {end_time - start_time}")

if __name__ == "__main__":
    main()