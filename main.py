import cv2
import os
import time

from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine



IMAGE_DIR = "C:\\Users\\Despacito4\\Desktop\\test_images"
output_folder = "identified_faces"
os.makedirs(output_folder, exist_ok=True)

face_data = []  # To store the embeddings - tuples (filename, face_index, embedding)


# TODO: Using the packaged model we can only use CPU, if we are going to use GPU we need to implement the model usage from scratch
def initialize_face_analysis():
    app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 for CPU
    return app


def process_image(file_path, app):

    img = cv2.imread(file_path)
    if img is None:
        print(f"Error reading - {file_path}")
        return
    
    faces = app.get(img)
    
    if len(faces) == 0:
        print(f"No faces found in - {file_path}")
        return
    
    for idx, face in enumerate(faces):
        bbox = face.bbox.astype(int).flatten()
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        embedding = face.normed_embedding
        
        face_data.append((file_path, idx, embedding))
        
        print(f"Face {idx + 1} in {file_path}: {embedding[:3]}")
        
        cv2.putText(img, f"Face {idx + 1}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    output_path = os.path.join(output_folder, os.path.basename(file_path))
    cv2.imwrite(output_path, img)
    print(f"Success - saved result to {output_path}")



def compare_faces(embedding1, embedding2, threshold=0.5):
    # Compare cosine similarity between two embeddings, closer to 0 is more similar
    similarity = cosine(embedding1, embedding2)
    return similarity < threshold


start_time = time.time()

app = initialize_face_analysis()

for filename in os.listdir(IMAGE_DIR):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        file_path = os.path.join(IMAGE_DIR, filename)
        process_image(file_path, app)


print("\n--- Face Recognition Results ---\n")
for i, (file1, face_idx1, embedding1) in enumerate(face_data):
    for j, (file2, face_idx2, embedding2) in enumerate(face_data):
        if i != j:
            if compare_faces(embedding1, embedding2):
                print(f"Face {face_idx1 + 1} in {file1} matches Face {face_idx2 + 1} in {file2}")



print(f"\n--- Total time taken: {time.time() - start_time:.2f} seconds ---")