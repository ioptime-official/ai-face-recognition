import mysql.connector
import numpy as np
import json
import os

def find_person2(input_embedding, threshold=0.40):
    input_embedding = np.array(input_embedding) / np.linalg.norm(input_embedding)
    conn = mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="freebies#123",
        database="testing",
    )
    try:
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM newuser")
        rows = cursor.fetchall()
        if not rows:
            st = "Database is empty"
            return st
        else:
            names = [row[0] for row in rows]
            embedding_list = [json.loads(row[1]) for row in rows]
            embedding_list = [np.array(embedding) / np.linalg.norm(embedding) for embedding in embedding_list]
            cosine_similarities = np.dot(np.array(input_embedding), np.array(embedding_list).T)
            if len([num for num in cosine_similarities if num >= threshold])==0:
                            st = "No match found."
                            return st
            else:
                top_indices = np.argsort(cosine_similarities)[-5:][::-1]
                for idx in top_indices:
                    similarity_score = cosine_similarities[idx]
                    if similarity_score > threshold:
                        #  st=  f"Best match: {os.path.basename(names[idx])}, Similarity: {similarity_score}"
                         st = os.path.basename(names[idx])
                         return st
    except Exception as e:
        return f"An error occurred: {e}"
    finally:
        conn.close()
