# Face Recognition Project

Welcome to our Face Recognition project! This project utilizes Flask API to create a user-friendly application for face recognition. Below, you'll find detailed information on how to use the application, the technologies used, and instructions for setting up the project locally.

## Features

- **User Registration:**
  - Users can register using an image.
  - Users can register by capturing their face using the device camera.
  
- **Face Recognition Options:**
  - **Image Recognition:**
    - Users can input an image, and the application will identify and match the person in the database.
    - For accurate face detection, we have used mtcnn model
    - For face recognition, we have used Arcface quantized model with float 16 quantization.
  - **Real-time Face Recognition:**
    - Real-time face detection is performed using the Yunet model as it provide better FPS.
    - Face recognition is done using the Arcface quantized model with float 16 quantization.
  - **Video Face Recognition:**
    - Users can upload a video, and the application will perform face recognition on the video.
    - Output video will be saved in folder output_video

## Technologies Used

- **Backend:**
  - Flask (Python)
  - MySQL Database

- **Frontend:**
  - HTML
  - CSS
  
- **Face Detection and Recognition:**
  - Yunet Model (For real-time face detection)
  - MTCNN (for face detection in an image)
  - Arcface Quantized Model (For face recognition with float 16 quantization)


## How to Use

1. **User Registration:**
   - Click on the Manager User button persent on the home page.
   - Click on the Register User button
   - Choose to register using an image or by capturing your face with the camera.

2. **Face Recognition:**
   To open Face Recogntion  page, click on "Face Recogntion" button present on the Home Page.
   - **Image Recognition:**
     - Click on the "Image " button.
     - Upload an image.
     - The application will display the matched person from the database.
   - **Real-time Face Recognition:**
     - Click on the "Real-time " button.
     - The application will open the device camera.
     - Faces will be detected in real-time, and the recognized person will be displayed.
   - **Video Face Recognition:**
     - Click on the "Upload Video" button.
     - Upload a video file.
     - The application will perform face recognition on the video and save the     output video  in folder output_video  

## How to Set Up Locally

1. **Clone the Repository:**
   ```
   git clone https://github.com/ioptime-official/ai-face-recognition
   ```

2. **Install Dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Database Setup:**
   - Set up a MySQL database.
   - Update the database configuration in the Flask app (`app.py`).
  
4. **Run the Application:**
   ```
   python app.py
   ```

5. **Access the Application:**
   - Open your web browser and go to `http://localhost:5000`.
  
## Important Notes

- Ensure that you have a compatible device with a camera for real-time face recognition.
- We are using cosine similarity to find the match
- Proper lighting conditions are essential for accurate face detection and recognition.
- Make sure to set up the MySQL database and update the database configuration in the Flask app before running the application.
