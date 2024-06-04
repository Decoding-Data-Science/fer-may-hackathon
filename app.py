import streamlit as st
import cv2
import tempfile
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# This can be changed between cnnModel.h5, emotion_recognition_model.h5, and mobilenet.h5
model_path = 'mobilenet.h5'
emotion_model = load_model(model_path)

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def main():
    # Title of the app
    st.title("Video Input App with Face Detection")

    with st.expander("Demo Video"):
        st.video('assets/demo.webm')

    # File uploader for video input
    video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

    if video_file is not None:
        # Create a temporary file to save the uploaded video
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        # Play the video and perform face detection
        st.write("Processing video for face detection...")
        process_video(tfile.name)

        # Clean up: remove the temporary file
        tfile.close()
        os.unlink(tfile.name)

def process_video(video_path):
    # Load OpenCV's pre-trained Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    video_capture = cv2.VideoCapture(video_path)
    
    stframe = st.empty()

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Draw rectangles around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extract the face ROI
            face = gray_frame[y:y+h, x:x+w]
            # Resize the face to 48x48 pixels
            face_resized = cv2.resize(face, (224, 224))
            # Normalize the pixel values
            face_normalized = face_resized / 255.0
            # Expand dimensions to match model input shape
            face_rgb = np.stack((face_normalized,) * 3, axis=-1)
            face_input = np.expand_dims(face_rgb, axis=0)
            
            # Predict the emotion
            emotion_prediction = emotion_model.predict(face_input)
            emotion_label = emotion_labels[np.argmax(emotion_prediction)]

            # Draw a rectangle around the face and put the emotion label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        
        # Convert the frame back to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame
        stframe.image(rgb_frame, channels="RGB")

    video_capture.release()

if __name__ == "__main__":
    main()
