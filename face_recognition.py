import cv2
import os
import csv
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine

class FaceRecognition:
    def __init__(self, model_name="buffalo_l", ctx_id=0, det_size=(640, 640), providers=['CPUExecutionProvider']):
        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def extract_face_embeddings(self, file_path):
        img = cv2.imread(file_path)
        if img is None:
            print(f"Error reading - {os.path.basename(file_path)}")
            return []
        
        original_height, original_width = img.shape[:2] 
        faces = self.app.get(img) 
        
        if len(faces) == 0:
            print(f"No faces found in - {os.path.basename(file_path)}")
            return []
        
        embeddings_data = []
        
        for idx, face in enumerate(faces):

            x1, y1, x2, y2 = face.bbox.astype(int)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Testing to see if coordinates are correct
            # cv2.drawMarker(img, (center_x, center_y), color=(0, 0, 255), 
            #             markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)


            print(f"Bounding Box for Face {idx + 1} in {os.path.basename(file_path)}: "
                f"({x1}, {y1}) to ({x2}, {y2})")
            print(f"Calculated Center: ({center_x}, {center_y})")

            # Store the embeddings and coordinates
            embeddings_data.append((file_path, idx, face.normed_embedding, (center_x, center_y)))


        # testing only
        # marked_image_path = f"marked_{os.path.basename(file_path)}"
        # cv2.imwrite(marked_image_path, img)
        # print(f"Marked image saved as: {marked_image_path}")
    

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

    # Temporarily storing all face data locally just for testing the clustering
    def save_to_csv(self, face_data, csv_file='face_data.csv'):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(["media_Id", "embedding", "coordinates"]) 
            
            for file_path, face_idx, embedding, (center_x, center_y) in face_data:
                embedding_list = embedding.tolist()
                coordinates = f"({center_x}, {center_y})"
                writer.writerow([os.path.basename(file_path), embedding_list, coordinates])


        print(f"Data saved to {csv_file}")


    def compare_all_faces(self, face_data):
        for i, (file1, face_idx1, embedding1, _) in enumerate(face_data):
            for j, (file2, face_idx2, embedding2, _) in enumerate(face_data):
                if i != j:
                    if self.compare_faces(embedding1, embedding2):
                        print(f"Face {face_idx1 + 1} in {os.path.basename(file1)} matches "
                              f"Face {face_idx2 + 1} in {os.path.basename(file2)}")
