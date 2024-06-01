import streamlit as st
import tempfile
import cv2

st.set_page_config(
    page_title="FaceBolt | FER DDS Hackathon", 
    initial_sidebar_state="expanded",
    page_icon=":tada:", 
    )

st.title("FaceBolt | FER DDS Hackathon")

with st.sidebar:
    video = st.file_uploader("Upload a video", type=["mp4", "mov", "mkv"])

temp = tempfile.NamedTemporaryFile(delete=False)

if video:
    st.video(video)
    temp.write(video.read())
    vid = cv2.VideoCapture(temp.name)
    
