##################################################################################################################

from utils.image_processing import face_det
from utils.create_embeddings import get_embedding
from flask import Flask, request, jsonify, render_template, Response, redirect, send_from_directory, url_for
from utils.database_connect import  find_delete, update_username, get_users_from_database, initialize_db
import os
import cv2
from utils.register import register
from face_recognition.fr_Image_recognition import  find_person2
from face_recognition.fr_Recorded_video import face_recognition_video
from face_recognition.fr_Live_recognition import liveface_yunet
import time
import configparser

###################################################################################################################

config = configparser.ConfigParser()
config.read('config\config.ini')

###################################################################################################################

ADMIN_USERNAME = config['Admin']['admin_login']
ADMIN_PASSWORD = config['Admin']['admin_password']


#####################################

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'user_faces'
@app.route("/")
def home():
    return render_template('admin_login.html')
@app.route('/user_faces/<filename>')
def user_faces(filename):
    return send_from_directory(os.path.join(app.root_path, 'user_faces'), filename)

### Foor final capture
@app.route('/secdcam')
def cam():
    return render_template('camera2.html')

@app.route("/fr_menu")
def fr_menu():
    return render_template('FR_menu.html')

@app.route("/fr_image")
def fr_image():
    return render_template('FR_image.html')

@app.route("/open_camera")
def open_cam():
    return render_template('FR_image_camera.html')

@app.route("/register_menu")
def register_menu():
    return render_template('register_menu.html')

########################################################################################################################3

@app.route("/fr_open_camera")
def index():
    return render_template('index.html')

def capture_by_frames(): 
    global camera
    camera = cv2.VideoCapture(0)
    directory = os.path.dirname(__file__)
    weights = os.path.join(directory, "models/face_detection_yunet_2023mar.onnx")
    face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))
    while True:
        try:
            success, frame = camera.read() 
            if success:
                image = frame  # read the camera frame
                try:
                    # frame = liveface(frame)
                    frame = liveface_yunet(image, face_detector)
                except Exception as e:
                    print(f"Error in liveface function: {e}")
                    # Handle the error as needed
                    frame = image  # Use the original frame if an error occurs

                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error in capture_by_frames: {e}")
            # Handle the error as needed

@app.route('/video_capture')
def video_capture():
    return Response(capture_by_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop',methods=['POST', 'GET'])
def stop():
    if request.method == 'POST':

        # global camera
        if camera.isOpened():
            camera.release()
            return redirect(url_for('fr_menu'))

##############################################################################################################

@app.route("/fr_record")
def fr_video():
    return render_template('recorded_video.html')

##############################################################################################################

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return render_template('warning.html', error_message='No file part')
    try:
        video_file = request.files['video']
        if video_file.filename == '':
            return render_template('warning.html', error_message='No selected file')
        video_path = os.path.join('uploads', video_file.filename)
        output_path = os.path.join('output_video', video_file.filename)
        video_file.save(video_path)
        directory = os.path.dirname(__file__)
        weights = os.path.join(directory, "models/face_detection_yunet_2023mar.onnx")
        face_recognition_video(video_path, output_path, weights)
        return render_template('success.html')
    except Exception as e:
        print(f'Error in upload_video: {e}')
        # Handle the error as needed
        return render_template('warning.html', error_message='An error occurred during video processing')


################################################################################################################

@app.route('/admin_login', methods=['POST', 'GET'])
def admin():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            return redirect(url_for('admin_panel'))
        else:
            error_message = 'Invalid username or password'
            return render_template('admin_login.html', error_message=error_message)
    return render_template('admin_login.html', error_message=None)

################################################################################################################

@app.route('/face_identification', methods=['POST','GET'])
def face_identification():
    if request.method == 'POST':
        try:
            image_file = request.files['image']
            if image_file:
                    start_time = time.time()
                    try:
                        face=face_det(image_file, 'opencv')
                    except:
                        face=face_det(image_file, 'mtcnn')
                    embedding= get_embedding(face)
                    embedding_list = embedding
                    st = find_person2(embedding_list)
                    end_time = time.time()
                    total_time = end_time - start_time
                    return jsonify( {'result': st, 
                                        "time": total_time}  )

        except Exception as e:
            return jsonify({'result': f"An error occurred: {str(e)}", 'error': type(e).__name__})
    return render_template("face_identification_page.html")

##############################################################################################################

@app.route('/admin_panel')
def admin_panel():
    return render_template("admin_panel.html")

@app.route('/edituser')
def edit_panel():
    return render_template("edituser_panel.html")

##############################################################################################################

@app.route('/register_user', methods=['POST', 'GET'])
def register_user():
    text_content=" "
    if request.method == 'POST':
            name=request.form['name']
            image_file = request.files['image']
            text_content= register(name, image_file)
            return jsonify({'message': text_content })
    return render_template('register_user.html')

##############################################################################################################

@app.route('/edit_user', methods=['PUT'])
def update_user():
        original_username = request.args.get('name')
        new_username= request.args.get('new_name')
        conn=initialize_db()
        directory = "user_faces"
        try:
            current_name = f"{original_username}.png"
            new_name = f"{new_username}.png"
            current_path = os.path.join(directory, current_name)
            new_path = os.path.join(directory, new_name)
            os.rename(current_path, new_path)\
            
            msg=update_username(original_username, new_username, conn)
            if msg==0:
                return render_template('warning.html', message='User Not Found')
            else:
                return render_template('success.html', message='Name Updated Successfuly')
        except Exception as e:       
             return str(e), 500 
           
###########################################################################
@app.route('/list_users', methods=['GET'])
def get_users():
    conn=initialize_db()
    users = get_users_from_database(conn)
    if isinstance(users, list):
        return jsonify({'users': users})
    else:
        return jsonify({'error': users})

###########################################################################
@app.route('/delete_user2', methods=['DELETE'])
def delete_user():
    try:
        name = request.args.get('name')
        conn=initialize_db()
        con, _ = find_delete(name,conn)
        directory = "user_faces"
        del_name = f"{name}.png"
        current_path = os.path.join(directory, del_name)
        os.remove(current_path)
        return render_template('success.html', message=con)
    except Exception as e:
        print(f'Error in delete_user: {e}')
        # Handle the error as needed
        return str(e), 500
###########################################################################
       
@app.route('/list_edit')
def show_edit():
    return render_template('adminedit3.html')

if __name__ == '__main__':
    app.run(debug=True)