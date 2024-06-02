

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tempfile

# load trained model
model = load_model('facial_emotion_recognition_model.h5')

# emotions
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

st.title('Facial Emotion Recognition')
# vid file
uploaded_file = st.file_uploader("upload a video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # create temp file for storing and processing video
    tFile = tempfile.NamedTemporaryFile(delete=False)
    tFile.write(uploaded_file.read())

    capture = cv2.VideoCapture(tFile.name)

    stframe = st.empty()
    while capture.isOpened():
        _, frame = capture.read()
        if not _:
            break
        # convert frames to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect face in frame using haarcascase classifier
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray_frame, 1.1, 4)

        for (x, y, w, h) in faces:
            # extract region of interest and resize for model
            roi = gray_frame[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype('float32') / 255
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)

            # predict emotion for roi
            predictions = model.predict(roi)
            max_index = int(np.argmax(predictions))
            emotion = emotions[max_index]

            # label frame with preficted emotion and detection box
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # display in streamlit
        stframe.image(frame, channels="BGR")

    capture.release()

