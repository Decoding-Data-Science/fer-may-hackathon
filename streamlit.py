import cv2
import numpy as np
from keras.models import model_from_json, load_model
import streamlit as st
import tempfile
import tensorflow as tf

# Ensure TensorFlow uses the GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

st.title('Facial Emotion Detection - Vedanth Aggarwal ')

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
#emotion_model = load_model('Emotion_detection_with_CNN-main/emotion.h5')
# load json and create model
json_file = open('model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("model/emotion2.h5")
print("Loaded model from disk")

# start the webcam feed
#cap = cv2.VideoCapture(0)

# pass here your video path
# you may download one from here : https://www.pexels.com/video/three-girls-laughing-5273028/
#cap = cv2.VideoCapture(0)#"Emotion_detection_with_CNN-main/emotion_video.mp4")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
previous = [''] * 10
if uploaded_file is not None:
    # Save the uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        st.write("Error: Could not open video.")
    else:
        # Process the video
        stframe = st.empty()
        count = 0
        while cap.isOpened():
    # Find haar cascade to draw bounding box around face
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1280, 720))
            if not ret:
                break
            face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect faces available on camera
            num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
            if count%10 == 0:
                #previous = [''] * len(num_faces)
            # take each face available on the camera and Preprocess it
                for i,(x, y, w, h) in enumerate(num_faces):
                    
                        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
                        
                        roi_gray_frame = gray_frame[y:y + h, x:x + w]

                        # Convert ROI to RGB
                        #roi_rgb_frame = cv2.cvtColor(roi_gray_frame, cv2.COLOR_GRAY2RGB)

                        # Resize the ROI to (48, 48)
                        #resized_img = cv2.resize(roi_gray_frame, (48, 48))

                        # Normalize the image
                        #normalized_img = resized_img / 255.0

                        # Expand dimensions to match the input shape of the model
                        #cropped_img = np.expand_dims(normalized_img, axis=0)

                        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)),axis=-1),axis=0)
                        #np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

                        # predict the emotions
                        
                        emotion_prediction = emotion_model.predict(cropped_img)
                        maxindex = int(np.argmax(emotion_prediction))
                        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        previous[i] = emotion_dict[maxindex]
            else:
                 for i,(x, y, w, h) in enumerate(num_faces):
                    
                        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
                        
                        #roi_gray_frame = gray_frame[y:y + h, x:x + w]

                        # Convert ROI to RGB
                        #roi_rgb_frame = cv2.cvtColor(roi_gray_frame, cv2.COLOR_GRAY2RGB)

                        # Resize the ROI to (48, 48)
                        #resized_img = cv2.resize(roi_rgb_frame, (48, 48))

                        # Normalize the image
                        #normalized_img = resized_img / 255.0

                        # Expand dimensions to match the input shape of the model
                        #cropped_img = np.expand_dims(normalized_img, axis=0)

                        #cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)),axis=0),axis=0)
                        #np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

                        # predict the emotions
                        
                        #emotion_prediction = emotion_model.predict(cropped_img)
                        #maxindex = int(np.argmax(emotion_prediction))
                        #print(previous[i])
                        cv2.putText(frame, previous[i], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        #previous[i] = emotion_dict[maxindex]
                 

            stframe.image(frame, channels="BGR")
            count+=1

        cap.release()
        cv2.destroyAllWindows()
