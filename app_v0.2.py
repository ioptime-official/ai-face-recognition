import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from deepface import DeepFace
from deepface.commons import functions,distance as dst
import cv2
import mediapipe as mp
gamma = 1.5
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

st.title('Face Verification v0.2')
uploaded_images_1 = st.file_uploader("Upload Image 1", type=["jpg", "png", "jpeg"])
if uploaded_images_1: 
    image1 = Image.open(uploaded_images_1)
    image1 =image1.convert('RGB')
    image1_array = np.array(image1)
    st.image(image1, caption='Uploaded Image 1', use_column_width=True)

    try: 
        bgr_image1 = cv2.cvtColor(image1_array, cv2.COLOR_RGB2BGR)
        face_1=DeepFace.extract_faces(bgr_image1 , detector_backend='mtcnn')
        roi_1 = face_1[0]['face']
        #roi_2  = cv2.cvtColor(roi_1 , cv2.COLOR_BGR2RGB)
        face_image1_uint8 = (roi_1 * 255).astype(np.uint8) 
        image_1=cv2.resize(face_image1_uint8, (roi_1.shape[0],roi_1.shape[1]))
        with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
            results = face_mesh.process(image_1)
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks_np = np.zeros((468, 2), dtype=np.int32)
                for i, landmark in enumerate(face_landmarks.landmark):
                    landmarks_np[i] = (int(landmark.x * image_1.shape[1]), int(landmark.y * image_1.shape[0]))
                mask = np.zeros((image_1.shape[0], image_1.shape[1]), dtype=np.uint8)
                hull = cv2.convexHull(landmarks_np)
                cv2.fillConvexPoly(mask, hull, 255)
                face_extracted = cv2.bitwise_and(image_1, image_1, mask=mask)
                gamma_corrected_channels = [np.power(face_extracted[:, :, c] / 255.0, gamma) * 255.0 for c in range(3)]
                face_extracted= np.stack(gamma_corrected_channels, axis=-1).astype(np.uint8)
                # yuv_image = cv2.cvtColor(face_extracted, cv2.COLOR_BGR2YUV)
                # yuv_image[:,:,0] = cv2.equalizeHist(yuv_image[:,:,0])
                # face_extracted = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

                st.image(face_extracted, caption='Uploaded Image 1', use_column_width=True)



    except:
        st.warning("Face is not detected in the uploaded image. Kindly use different image")


uploaded_images_2 = st.file_uploader("Upload Image 2", type=["jpg", "png", "jpeg"])
if uploaded_images_2:
    image2 = Image.open(uploaded_images_2)
    image2 =image2.convert('RGB')
    image2_array = np.array(image2)
    st.image(image2, caption='Uploaded Image 2', use_column_width=True)
    try: 

        bgr_image2 = cv2.cvtColor(image2_array, cv2.COLOR_RGB2BGR)
        face_2=DeepFace.extract_faces(bgr_image2, detector_backend='mtcnn')
        roi_2 = face_2[0]['face']
        #roi_2  = cv2.cvtColor(roi_2 , cv2.COLOR_BGR2RGB)
        face_image2_uint8 = (roi_2 * 255).astype(np.uint8) 
        image_2=cv2.resize(face_image2_uint8, (roi_2.shape[0],roi_2.shape[1]))
        with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
            results = face_mesh.process(image_2)
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks_np = np.zeros((468, 2), dtype=np.int32)
                for i, landmark in enumerate(face_landmarks.landmark):
                    landmarks_np[i] = (int(landmark.x * image_2.shape[1]), int(landmark.y * image_2.shape[0]))
                mask = np.zeros((image_2.shape[0], image_2.shape[1]), dtype=np.uint8)
                hull = cv2.convexHull(landmarks_np)
                cv2.fillConvexPoly(mask, hull, 255)
                face_extracted_2 = cv2.bitwise_and(image_2, image_2, mask=mask)
                gamma_corrected_channels2 = [np.power(face_extracted_2[:, :, c] / 255.0, gamma) * 255.0 for c in range(3)]
                face_extracted_2= np.stack(gamma_corrected_channels2, axis=-1).astype(np.uint8)
                # yuv_image2 = cv2.cvtColor(face_extracted_2, cv2.COLOR_BGR2YUV)
                # yuv_image2[:,:,0] = cv2.equalizeHist(yuv_image2[:,:,0])
                # face_extracted_2 = cv2.cvtColor(yuv_image2, cv2.COLOR_YUV2BGR)

                
                st.image(face_extracted_2 , caption='Uploaded Image 1', use_column_width=True)

    except:
        st.warning("Face is not detected in the uploaded image. Kindly use different image")

# Load your TFLite model for face embeddings
tflite_model_path = 'arcface_model_quant.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def get_embedding(image):
    input_shape = input_details[0]['shape']
    input_image = tf.image.resize(image, (input_shape[1], input_shape[2]))
    input_image = np.expand_dims(input_image, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    output_tensor = interpreter.get_tensor(output_details[0]['index'])
    # normalized_embedding = output_tensor[0] / np.sqrt(np.sum(output_tensor[0]**2))
    
    return output_tensor[0] 

def get_embedding(image):
    input_shape = input_details[0]['shape']
    input_image = tf.image.resize(image, (input_shape[1], input_shape[2]))
    input_image = np.expand_dims(input_image, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    output_tensor = interpreter.get_tensor(output_details[0]['index'])
    return output_tensor[0]

if st.button("Verify"):
    if 'image1' in locals() and 'image2' in locals():
        # obj=DeepFace.verify(face_extracted, face_extracted_2, model_name="ArcFace", detector_backend='mtcnn')
        # st.write((1-obj['distance'])*100)
        # if ((1-obj['distance'])*100) >= 40:
        #     st.write('Face Matched')
        # else:
        #     st.write("Face is not Matched")

        embedding_first = get_embedding(face_extracted/255)
        embedding_second = get_embedding(face_extracted_2/255)
       # st.image(face_extracted , caption='Uploaded Image 1', use_column_width=True)
        #st.image(face_extracted_2 , caption='Uploaded Image 1', use_column_width=True)
        cosine_similarity = dst.findCosineDistance(embedding_first, embedding_second)
        st.write(f"Cosine Similarity Score: {(1-cosine_similarity)*100:.2f}")
    else:
        st.warning("Please upload both images for verification.")
