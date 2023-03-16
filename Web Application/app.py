from flask import Flask, render_template, Response, request
from keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

app = Flask(__name__, template_folder='templates', static_folder='static')

face_classifier = cv2.CascadeClassifier(r'C:/Users/Snigdho/Pictures/Emotion detection/haarcascade_frontalface_default.xml')
classifier =load_model(r'C:/Users/Snigdho/Pictures/Emotion detection/Emotion_Detection.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def gen_frames():  # Generate camera frames
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()

        if not success:
            break

        labels = []
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(224,224),interpolation=cv2.INTER_AREA)
            roi = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
            roi = roi.astype('float')/255.0
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
