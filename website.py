# To run this website, download the modules from requirements file. Then run this command on your terminal:
# streamlit run website.py We currently are processing only one face per frame to reduce computational 
# overhead. The latency might go down when streaming frames directly to the streamlit app.



import streamlit as st
import cv2 as cv
import tempfile
import numpy as np
import streamlit as st
import tensorflow as tf

# Load the emotion recognition model
interpreter = tf.lite.Interpreter('emotion_quantized.tflite')
interpreter.allocate_tensors()

def get_input_details(interpreter):
    input_details = interpreter.get_input_details()
    for detail in input_details:
        if detail['name'] == 'input':
            return detail['index']
    raise ValueError("Input tensor not found.")

def get_output_details(interpreter):
    output_details = interpreter.get_output_details()
    return output_details[0]['index']


st.title("Team Cipher Pol Facial Emotion Detector")
f = st.file_uploader("Upload file")





def preprocess(input_image, target_size=(48, 48)):
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img_gray = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        # change the following line to detect all the faces
        x, y, w, h = faces[0]
        face_img = img_gray[y:y+h, x:x+w]
        cv.rectangle(input_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        resized_face = cv.resize(face_img, target_size)
        normalized_face = resized_face / 255.0
        normalized_face = np.expand_dims(normalized_face, axis=0)
        return normalized_face
    return None

stframe = st.empty()

def predict_emotion_from_video_file(model, video_file_path):
    cap = cv.VideoCapture(video_file_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        preprocessed_frame = preprocess(frame, target_size=(48, 48))
        if preprocessed_frame is not None:
            input_details = interpreter.get_input_details()
            input_index = input_details[0]['index']
            preprocessed_frame_float32 = preprocessed_frame.astype(np.float32).reshape(1, 48, 48, 1)
            interpreter.set_tensor(input_index, preprocessed_frame_float32)  # Use the converted array
            interpreter.invoke()

            output_index = get_output_details(interpreter)
            predictions = interpreter.get_tensor(output_index)
            top_classes_indices = np.argsort(predictions)[0, -2:][::-1]
            top_classes_labels = ['neutral', 'happy', 'angry', 'surprise', 'sad']
            top1_class_index = top_classes_indices[0]
            top1_class_label = top_classes_labels[top1_class_index]
            top1_class_percentage = predictions[0, top1_class_index] * 100
            top2_class_index = top_classes_indices[1]
            top2_class_label = top_classes_labels[top2_class_index]
            top2_class_percentage = predictions[0, top2_class_index] * 100
            cv.putText(frame, f"{top1_class_label}: {top1_class_percentage:.2f}% | {top2_class_label}: {top2_class_percentage:.2f}%", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv.imshow('Real-Time Emotion Detection', frame)
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            stframe.image(frame_rgb)
              

if f is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(f.read())
    predict_emotion_from_video_file(interpreter, tfile.name)