import cv2
import os
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine

class FaceRecognition:
    def __init__(self, model_name="buffalo_l", ctx_id=0, det_size=(640, 640), providers=['CPUExecutionProvider']):
        """
        Initializes the FaceRecognition class with the FaceAnalysis model.
        
        :param model_name: Name of the model to use (default: 'buffalo_l').
        :param ctx_id: Context ID (0 for CPU, -1 for GPU if available).
        :param det_size: Detection size for the face analysis model (default: (640, 640)).
        :param providers: Providers to use for running the model (default: ['CPUExecutionProvider']).
        """
        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def extract_face_embeddings(self, file_path):
        """
        Process an image and extract face embeddings.
        
        :param file_path: Path to the image file.
        :return: List of face embeddings and file path (if any faces detected).
        """
        img = cv2.imread(file_path)
        if img is None:
            print(f"Error reading - {file_path}")
            return []
        
        faces = self.app.get(img)
        
        if len(faces) == 0:
            print(f"No faces found in - {file_path}")
            return []
        
        embeddings = [(file_path, idx, face.normed_embedding) for idx, face in enumerate(faces)]
        return embeddings

    def compare_faces(self, embedding1, embedding2, threshold=0.5):
        """
        Compare two face embeddings and determine if they are similar based on a cosine similarity threshold.
        
        :param embedding1: First face embedding.
        :param embedding2: Second face embedding.
        :param threshold: Similarity threshold (default is 0.5).
        :return: True if the faces are similar, False otherwise.
        """
        similarity = cosine(embedding1, embedding2)
        return similarity < threshold

    def process_images_in_directory(self, image_dir):
        """
        Process all images in a given directory and extract face embeddings.
        
        :param image_dir: Path to the directory containing image files.
        :return: List of face data (filename, face index, embedding).
        """
        face_data = []

        for filename in os.listdir(image_dir):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                file_path = os.path.join(image_dir, filename)
                face_embeddings = self.extract_face_embeddings(file_path)
                face_data.extend(face_embeddings)
        
        return face_data

    def compare_all_faces(self, face_data):
        """
        Compare all detected faces based on their embeddings and print the matching results.
        
        :param face_data: List of tuples containing (filename, face index, embedding).
        """
        for i, (file1, face_idx1, embedding1) in enumerate(face_data):
            for j, (file2, face_idx2, embedding2) in enumerate(face_data):
                if i != j:
                    if self.compare_faces(embedding1, embedding2):
                        print(f"Face {face_idx1 + 1} in {file1} matches Face {face_idx2 + 1} in {file2}")
