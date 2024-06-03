from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import av
import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import tempfile


try:
    classifier = load_model('model.h5')
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
except Exception as e:
    st.write(f"Error loading model or cascade classifier: {e}")

emotion_counts = {label: 0 for label in emotion_labels}


class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            emotion_counts[label] += 1
            cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        return img


def process_video(uploaded_file):
    image_list = []
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_capture = cv2.VideoCapture(tmp_file.name)
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                emotion_counts[label] += 1
                cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_list.append(frame_rgb)
        video_capture.release()
    
    st.write('Detected Emotion: ', max(emotion_counts, key=emotion_counts.get))
    st.write('Confidence Level (%): ', round((emotion_counts[max(emotion_counts, key=emotion_counts.get)] / len(image_list) * 100), 2))
    st.write('------------------------')

    st.write('Emotion Detected Per Frame:')
    for label, count in emotion_counts.items():
        st.write(f'{label}: {count}')

    image_index = st.slider("Please use the slider to drag across different frames", 0, len(image_list) - 1, 0)
    st.image(image_list[image_index], use_column_width=True)


def main():
    st.title("Emotion Detection in Uploaded Videos")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "flv"])
    if uploaded_file is not None:
        process_video(uploaded_file)


if __name__ == "__main__":
    main()
