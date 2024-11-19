import cv2
import os
import logging
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class FaceRecognition:
    def __init__(self, model_name="buffalo_l", ctx_id=0, det_size=(640, 640), providers=['CPUExecutionProvider']):
        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def extract_face_embeddings(self, file_path):
        img = cv2.imread(file_path)
        if img is None:
            logging.error(f"Error reading - {os.path.basename(file_path)}")
            return []
        
        faces = self.app.get(img) 
        
        if len(faces) == 0:
            logging.info(f"No faces found in - {os.path.basename(file_path)}")
            return []
        
        embeddings_data = []
        
        for face_id, face in enumerate(faces):
            x1, y1, x2, y2 = face.bbox.astype(int)
            
            # Coordinate system is left to right positive x and top to bottom positive y
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            logging.info(f"Bounding Box for Face {face_id + 1} in {os.path.basename(file_path)}: "
                         f"({x1}, {y1}) to ({x2}, {y2})")

            coordinates = [center_x, center_y]
            embeddings_data.append((file_path, face.normed_embedding, [x1, y1, x2, y2], face_id))

        return embeddings_data

    def compare_faces(self, embedding1, embedding2, threshold=0.5):
        similarity = cosine(embedding1, embedding2)
        return similarity < threshold

    def process_images_in_directory(self, image_dir):
        face_data = []

        for filename in os.listdir(image_dir):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                file_path = os.path.join(image_dir, filename)
                face_embeddings = self.extract_face_embeddings(file_path)
                face_data.extend(face_embeddings)
        
        return face_data

    def compare_all_faces(self, face_data):
        for i, (file1, face_idx1, embedding1, _) in enumerate(face_data):
            for j, (file2, face_idx2, embedding2, _) in enumerate(face_data):
                if i != j:
                    if self.compare_faces(embedding1, embedding2):
                        logging.info(f"Face {face_idx1 + 1} in {os.path.basename(file1)} matches "
                                     f"Face {face_idx2 + 1} in {os.path.basename(file2)}")
