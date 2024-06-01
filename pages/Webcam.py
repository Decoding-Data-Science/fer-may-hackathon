import cv2
import streamlit as st

st.set_page_config(
    page_title="FaceBolt | DDS FER Challenge", page_icon=":tada:")

st.title("FaceBolt | DDS FER Challenge")

run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

while run:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
else:
    st.write('Stopped')


with st.sidebar:
    st.subheader("Upload a video")
