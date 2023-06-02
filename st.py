import cv2
import streamlit as st
from keras.models import load_model
import numpy as np
# Load the trained drowsiness detector
detector = cv2.CascadeClassifier("C:/Users/HP/Desktop/projects/driver drowsiness detection/Drowsiness detection/haar cascade files/haarcascade_frontalface_alt.xml")
model = load_model("C:/Users/HP/Desktop/projects/driver drowsiness detection/Drowsiness detection/models/cnnCat2.h5")
eye_cascade = cv2.CascadeClassifier('C:/Users/HP/Desktop/projects/driver drowsiness detection/Drowsiness detection/haar cascade files/haarcascade_righteye_2splits.xml')
# Start the video stream
cap = cv2.VideoCapture(0)

# Define the alert sound
alert_sound = cv2.imread("path/to/alert_sound.jpg")

# Define a function to check for drowsiness
def detect_drowsiness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            eye_roi = roi_color[ey: ey + eh, ex: ex + ew]
            eye = cv2.resize(eye_roi, (224, 224))
            eye = cv2.cvtColor(eye, cv2.COLOR_BGR2RGB)
            eye = np.expand_dims(eye, axis=0)
            eye = eye/255.0
            preds = model.predict(eye)
            if preds < 0.5:
                cv2.putText(frame, "DROWSY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                cv2.imshow("Alert", alert_sound)
                cv2.waitKey(2000)

    return frame

# Define the Streamlit web app
st.title("Driver Drowsiness Detection")
st.write("This app uses OpenCV and Streamlit to detect drowsiness in drivers in real-time.")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = detect_drowsiness(frame)
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
