import matplotlib.pyplot as plt
from deepface import DeepFace
import os
import pickle

folder_path = "Humans"
file_names = os.listdir(folder_path)
file_links = [os.path.join(folder_path, file_name) for file_name in file_names]
file_links_list = [ file_link  for file_link in file_links]
print("Total Image Present  :  "+ str(len(file_links_list )))

embeddings = []
counter = 0
for img_path in file_links_list:
    img_embedding=DeepFace.represent(img_path,enforce_detection=False, model_name='ArcFace')
    embeddings.append(img_embedding[0]['embedding'])
    counter += 1
    print(counter)
print("Embedding generation completed.")

embeddings_file_path = 'embeddings.pkl'
# Dump the embeddings list into the file using pickle
with open(embeddings_file_path, 'wb') as file:
    pickle.dump(embeddings, file)
print(f"Embeddings list saved to '{embeddings_file_path}'")