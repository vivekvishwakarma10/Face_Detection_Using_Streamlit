import streamlit as st
import streamlit.components.v1 as components
import cv2
import logging as log
import datetime as dt
from time import sleep


cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

# Streamlit code

# Title
st.title("Face Detection Through Opencv Using Streamlit")
st.header("Streamlit")

# Render the h1 block, contained in a frame of size 700x100.
components.html("<html><body><h3>Streamlit is an open-source Python library that makes it easy to "
                "create and share beautiful, custom web apps for machine learning and data science. "
                "In just a few minutes you can build and deploy powerful data apps.</h3></body></html>"
                , width=700, height=100)


# Building a sidebar
st.sidebar.subheader("Details of the person")
t1 = st.sidebar.text_input("Name of the Person 1")
s1 = st.sidebar.slider("Age of the person 1")

st.sidebar.markdown("---")


st.write("Name: ",t1)
st.write("Age: ", s1) # taking data from the sidebar




# Selection box
# first argument takes the titleof the selectionbox second argument takes options
How_is_streamlit = st.selectbox("likings: ",['Very much', 'So so', 'Boring', 'Useless'])
st.write("Your review is: ", How_is_streamlit)

st.markdown(f'<hr style="height:2px;border:none;color:#333;background-color:#333;" />', unsafe_allow_html=True)
# Face Detection works

st.header("Opencv - Detection of Faces")
st.subheader("Understanding the methods we have used - what is Haar Cascades? ")

st.write("Object Detection using Haar feature-based cascade classifiers is an effective object detection "
         "method proposed by Paul Viola and Michael Jones in their paper, 'Rapid Object Detection using a "
         "Boosted Cascade of Simple Features' in 2001. It is a machine learning based approach where a "
         "cascade function is trained from a lot of positive and negative images. It is then used to "
         "detect objects in other images.")

# Accessign haar files: https://github.com/opencv/opencv/tree/master/data/haarcascades
components.html('<html><body>Note: The haar cascade files can be downloaded from the '
            '<a href = "https://github.com/opencv/opencv/tree/master/data/haarcascades">'
            'OpenCV Github repository</a></body></html>')



if st.button("Detect your face By clicking here"):
    while True:
        if not video_capture.isOpened():
            print('Unable to load camera.')
            sleep(5)
            pass

        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if anterior != len(faces):
            anterior = len(faces)
            log.info("faces: " + str(len(faces)) + " at " + str(dt.datetime.now()))

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Display the resulting frame
        cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()