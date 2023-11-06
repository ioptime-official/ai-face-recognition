import cv2
# from deepface import DeepFace
import numpy as np
# from utils.image_processing import image_preprocessing
from utils.create_embeddings import get_embedding
# from utils.image_processing import image_processing_2
from face_recognition.fr_Image_recognition import  find_person2
import os
from utils.functions import alignment_procedure_upd


def face_recognition_video(input_video_path, output_video_path, weights):
    capture = cv2.VideoCapture(input_video_path)
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 10, (frame_width, frame_height))
    
    if not capture.isOpened():
        exit()
    # weights = os.path.join(directory, "models/face_detection_yunet_2023mar.onnx")
    face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))
    try:

        while True:
                result, image = capture.read()
                if result is False:
                    cv2.waitKey(0)
                    break
                channels = 1 if len(image.shape) == 2 else image.shape[2]
                if channels == 1:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                if channels == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                height, width, _ = image.shape
                face_detector.setInputSize((width, height))
                _, faces = face_detector.detect(image)
                faces = faces if faces is not None else []
                for face in faces:
                    box = list(map(int, face[:4]))
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    roi= image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
                    color = (0, 0, 255)
                    thickness = 2
                    landmarks = list(map(int, face[4:len(face)-1]))  
                    landmarks = np.array_split(landmarks, len(landmarks) / 2)
                    image_1= alignment_procedure_upd(roi, landmarks[0], landmarks[1])
                    # face_extracted = image_processing_2(image_1)
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
                if isinstance(image, np.ndarray):
                    out.write(image)
                else:
                    print("Invalid frame format. Skipping...")

    except Exception as e:
        print(f"Error processing video: {e}")
        return None

    finally:
        # Release the VideoWriter object
        out.release()

    return output_video_path

