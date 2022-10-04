#Importing Library
import cv2
#Defining the dataset of the frontal faces
face_data = cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml')
#capturing the video and setting parameter to 0 for recording from webcam/default cam
webcam = cv2.VideoCapture(0)

#Creating an infinite loop for proccessing for until manual brak command
while True:
    #giving library orders to obtain our video and scan it
    boolean_return, frame_return = webcam.read()
    #converting frame value returns to the grayscale to be able for detecting pixels by their contrast easily
    gray_scaled_frames = cv2.cvtColor(frame_return, cv2.COLOR_BGR2GRAY)
    #detecting scale by scale the frame
    face_coordinates = face_data.detectMultiScale(frame_return)

    for(x, y, w, h) in face_coordinates:
        cv2.rectangle(frame_return, (x,y), (x+h, y+w), (0, 0, 255), 2)

    cv2.imshow("aa", frame_return)
    cv2.waitKey(66)