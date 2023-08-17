import os
import time
import pickle
import numpy as np
from deepface import DeepFace
from deepface.commons import distance as dst

def verify_faces(img1_emb, img2_emb, model_name="ArcFace",  distance_metric="cosine"):
    distances = []
    img1_representation = img1_emb[0]['embedding']
    img2_representation = img2_emb
    distance = dst.findCosineDistance(img1_representation, img2_representation)
    distances.append(distance)
    threshold = dst.findThreshold(model_name, distance_metric)
    best_distance = min(distances)
    return {
        "verified": best_distance <= threshold,
        "distance": best_distance,
        "threshold": threshold,
        "model": model_name,
    }

def main():
    embedding_path = 'embeddings/embeddings3000.pkl'
    with open(embedding_path, 'rb') as pickle_file:
        face_embeddings = pickle.load(pickle_file)

    name_path = 'embeddings/link_list3000.pkl'
    with open(name_path, 'rb') as pickle_file:
        face_links = pickle.load(pickle_file)

    print("Lists loaded from pickle files.")

    start_time = time.time()
    test_image_path = 'Test_faces\sam.PNG'
    test_image_emb = DeepFace.represent(test_image_path, model_name="ArcFace")
    embeddings_array = np.array(face_embeddings)
    cosine_similarities = np.dot(test_image_emb[0]['embedding'], embeddings_array.T)
    top_indices = np.argsort(cosine_similarities)[-5:][::-1]
    print("Results:")
    for idx in top_indices:
        similarity_score = cosine_similarities[idx]
        if similarity_score > 7:
            obj = verify_faces(test_image_emb, embeddings_array[idx], model_name='ArcFace')
            if obj['verified'] and obj['distance'] < 0.6:
                print("\nMatched with:")
                print(f"Matching Person: {os.path.basename(face_links[idx])}")
                print(f"Distance: {obj['distance']}")
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTime: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
