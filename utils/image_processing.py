import cv2
from PIL import Image
import numpy as np
from deepface.DeepFace import extract_faces
# import mediapipe as mp


# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

def face_det(image_file, detect):
    image = Image.open(image_file)
    image = image.convert('RGB')
    image_array = np.array(image)
    bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    face = extract_faces(bgr_image, detector_backend=detect)[0]['face']
    return face

# def image_preprocessing(face):   
#     roi_1=np.array(face)
    
#     face_image1_uint8 = (roi_1 * 255).astype(np.uint8) 
#     image_1=cv2.resize(face_image1_uint8, (roi_1.shape[0],roi_1.shape[1]))
#     with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
#         results = face_mesh.process(image_1)
#         if results.multi_face_landmarks:
#             face_landmarks = results.multi_face_landmarks[0]
#             landmarks_np = np.zeros((468, 2), dtype=np.int32)
#             for i, landmark in enumerate(face_landmarks.landmark):
#                 landmarks_np[i] = (int(landmark.x * image_1.shape[1]), int(landmark.y * image_1.shape[0]))
#             mask = np.zeros((image_1.shape[0], image_1.shape[1]), dtype=np.uint8)
#             hull = cv2.convexHull(landmarks_np)
#             cv2.fillConvexPoly(mask, hull, 255)
#             face_extracted = cv2.bitwise_and(image_1, image_1, mask=mask)
#             return face_extracted

# def image_processing_2(image):
#                 bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#                 roi_1=np.array(bgr_image)
#                 image1=cv2.resize(roi_1, (roi_1.shape[0],roi_1.shape[1]))
#                 with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
#                     results = face_mesh.process(image1)
#                     if results.multi_face_landmarks:
#                         face_landmarks = results.multi_face_landmarks[0]
#                         landmarks_np = np.zeros((468, 2), dtype=np.int32)
#                         for i, landmark in enumerate(face_landmarks.landmark):
#                             landmarks_np[i] = (int(landmark.x * image1.shape[1]), int(landmark.y * image1.shape[0]))
#                         mask = np.zeros((image1.shape[0], image1.shape[1]), dtype=np.uint8)
#                         hull = cv2.convexHull(landmarks_np)
#                         cv2.fillConvexPoly(mask, hull, 255)
#                         face_extracted = cv2.bitwise_and(image1, image1, mask=mask)
#                         return face_extracted   


   
    