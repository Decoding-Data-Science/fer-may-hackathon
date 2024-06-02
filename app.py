from moviepy.editor import ImageSequenceClip
from collections import Counter
import tempfile
import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
import time
import os
import base64

# Load the pre-trained model once


@st.cache_resource
def load_emotion_model():
    return load_model('best_model.h5')


# Load the model
model = load_emotion_model()

# Load the face detection classifier
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define the labels for facial expressions
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear',
               3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Function to perform facial expression recognition on an image


def process_image(image):
    # Resize image to reduce computation time
    resized_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.1, 5)

    for x, y, w, h in faces:
        sub_face_img = gray[y:y + h, x:x + w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]

        cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(resized_image, (x, y - 20),
                      (x + w, y), (50, 50, 255), -1)
        cv2.putText(resized_image, labels_dict[label], (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return resized_image


def process_video(video_file):
    cap = cv2.VideoCapture(video_file)
    processed_frames = []

    frame_window = st.empty()
    processing_placeholder = st.empty()

    uploaded_fps = cap.get(cv2.CAP_PROP_FPS)
    if uploaded_fps == 0:
        st.error("Failed to retrieve valid frame rate for the video.")
        return

    emotions_list = []  # List to store emotions of all frames
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % 13 == 0:  # Skip 5 frames
            processed_frame = process_image(frame)
            processed_frame_rgb = cv2.cvtColor(
                processed_frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            processed_frames.append(processed_frame_rgb)

            # Display the processed frame
            frame_window.image(processed_frame_rgb,
                               channels="RGB", use_column_width=True)

            # Display processing text and GIF
            with open("load.svg", "rb") as f:
                html_content = f.read()
                processing_placeholder.html(
                    f'Processing... <img src="data:image/svg+xml;base64,{base64.b64encode(html_content).decode()}" width="50"/>')

            # Get emotion of the frame and add it to the list
            emotion = get_dominant_emotion(processed_frame)
            emotions_list.append(emotion)

        frame_number += 1  # Increment by 1 to process each frame, not skipping them

    cap.release()

    # Create video clip
    clip = ImageSequenceClip(processed_frames, fps=4)

    # Write the processed video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        output_video_path = temp_file.name
        clip.write_videofile(output_video_path, codec='libx264')

    # Hide the video being processed
    frame_window.empty()
    processing_placeholder.empty()

    # Display the processed video
    st.video(output_video_path)

    # Find the most dominant emotion
    dominant_emotion = Counter(emotions_list).most_common(1)[0][0]

    # Display the frame with the most dominant emotion
    most_dominant_frame = get_frame_with_emotion(
        processed_frames, dominant_emotion)
    st.image(most_dominant_frame,
             caption=f"Most dominant emotion: {labels_dict[dominant_emotion]}", use_column_width=True)

    # Remove the temporary file
    os.remove(output_video_path)


def get_dominant_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.1, 5)

    emotions = []
    for x, y, w, h in faces:
        sub_face_img = gray[y:y + h, x:x + w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        emotions.append(label)

    if emotions:
        return max(set(emotions), key=emotions.count)
    else:
        return None


def get_frame_with_emotion(frames, emotion):
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.1, 5)

        for x, y, w, h in faces:
            sub_face_img = gray[y:y + h, x:x + w]
            resized = cv2.resize(sub_face_img, (48, 48))
            normalize = resized / 255.0
            reshaped = np.reshape(normalize, (1, 48, 48, 1))
            result = model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]

            if label == emotion:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y - 20),
                              (x + w, y), (50, 50, 255), -1)
                cv2.putText(frame, labels_dict[label], (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                return frame

    return None

# Function to start the webcam and perform facial expression recognition


def webcam_stream():
    video = cv2.VideoCapture(0)
    frame_window = st.image([])

    while st.session_state['run']:
        ret, frame = video.read()
        if not ret:
            st.write("Failed to grab frame")
            break

        frame = process_image(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb)

        time.sleep(0.03)

    video.release()
    frame_window.image([])


# Streamlit UI
st.title("Facial Expression Recognition")

# Upload a file
uploaded_file = st.file_uploader("Please upload a video", type=["mp4", "avi"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    if uploaded_file.type.startswith('image'):
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, channels="BGR")
        processed_image = process_image(image)
        st.image(processed_image, channels="BGR")
    else:
        video_path = f"temp_video.{uploaded_file.type.split('/')[-1]}"
        with open(video_path, 'wb') as out_file:
            out_file.write(file_bytes)
        process_video(video_path)

if 'run' not in st.session_state:
    st.session_state['run'] = False

st.title("Real Time Facial Expression Recognition")

# Display the start and stop buttons side by side
col1, col2 = st.columns(2)
with col1:
    if st.button("Start Webcam"):
        st.session_state['run'] = True
        st.experimental_rerun()

with col2:
    if st.button("Stop Webcam"):
        st.session_state['run'] = False
        st.experimental_rerun()

if st.session_state['run']:
    st.write("")
    webcam_stream()
else:
    st.write("")
