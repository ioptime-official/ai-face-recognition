from utils.image_processing import face_det
from utils.create_embeddings import get_embedding
from utils.database_connect import  input_data, check_user, initialize_db
import matplotlib.pyplot as plt
import json
from PIL import Image

def register(name, image_file):  
            conn=initialize_db()
            if image_file:
                check=check_user(name, conn)
                if check==1:
                    text_content="Use Different name, this name is already present in the database"
                else:
                    try:
                        face=face_det(image_file, 'opencv')
                        image = Image.fromarray(face)
                        try:
                            folder_path = 'user_faces/'
                            file_path = folder_path + name
                            image.save(file_path)
                            print(f"Image saved at: {file_path}")
                        except Exception as e:
                            print(e)

                    except:
                        face=face_det(image_file, 'mtcnn')
                        try:
                            plt.imshow(face)
                            plt.axis('off')  # Turn off axis numbers and ticks
                            plt.savefig(f'user_faces/{name}.png', bbox_inches='tight', pad_inches=0)
                            plt.close()
                            print("Image saved ")
                        except Exception as e:
                            print(e)
                    embedding= get_embedding(face)
                    embedding_list  = [embedding.tolist() for embedding in embedding]
                    embedding = json.dumps(embedding_list)
                    
                    input_data( name,embedding, conn)
                    text_content='Image processed and embeddings stored successfully.'
                    return text_content