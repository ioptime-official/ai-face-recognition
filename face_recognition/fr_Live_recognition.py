# from utils.image_processing import image_preprocessing, image_processing_2
from utils.create_embeddings import get_embedding
from deepface import DeepFace
import cv2
from face_recognition.fr_Image_recognition import  find_person2
import numpy as np
from utils.functions import alignment_procedure_upd

def liveface_opencv(frame):
    face_objs = DeepFace.extract_faces(
                                img_path=frame,
                                detector_backend='opencv',
                                enforce_detection=True)
    for detected_face in face_objs:
        facial_area = detected_face["facial_area"]
        x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
        #face_extracted = image_preprocessing(detected_face['face'])
        face_extracted=detected_face['face']
        try:
            embedding = get_embedding(face_extracted / 255)
        except Exception as e:
            print(e)
            continue  # Skip this face if embedding extraction fails

        st = find_person2(embedding)
        st = str(st)
        text_size = cv2.getTextSize(st, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y - 10

        cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), (255, 0, 0), -1)
        cv2.putText(frame, str(st), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return frame

def liveface_yunet(image, face_detector):
            channels = 1 if len(image.shape) == 2 else image.shape[2]
            if channels == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if channels == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            height, width, _ = image.shape
            face_detector.setInputSize((width, height))
        # try:
            _, faces = face_detector.detect(image)
            faces = faces if faces is not None else []
            for face in faces:
                # print(f'total {face}')
                box = list(map(int, face[:4]))
                x, y, w, h = box[0], box[1], box[2], box[3]
                roi= image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
                color = (0, 0, 255)
                thickness = 2
                landmarks = list(map(int, face[4:len(face)-1]))  
                landmarks = np.array_split(landmarks, len(landmarks) / 2)
                image_1= alignment_procedure_upd(roi, landmarks[0], landmarks[1])
                #face_extracted = image_processing_2(image_1)
                try:
                    embedding = get_embedding(image_1 / 255)
                except Exception as e:
                    continue
                st = find_person2(embedding)
                st = str(st)
                text_size = cv2.getTextSize(st, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = x + (w - text_size[0]) // 2
                text_y = y - 10
                cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)
                cv2.rectangle(image, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), (0, 0, 255), -1)
                cv2.putText(image,str(st), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return image
        # except:
        #     return image
