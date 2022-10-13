#Defining the libraries
import time
import cv2
import mediapipe as mp #for better detection and frame rate and accuracy

pTime = 0
#capturing the video and setting parameter to 0 for recording from webcam/default cam
cap = cv2.VideoCapture(0)
#face detection module
mpFaceDetection = mp.solutions.mediapipe.python.solutions.face_detection
#drawing rectangle module
mpDraw = mp.solutions.mediapipe.python.solutions.drawing_utils
#initializator for our module
faceDetection = mpFaceDetection.FaceDetection()

#infinite loop for detecting through all time
while True:
    #2 different returns that we get from .read funct so 2 different variables we gotta assign
    success, img =cap.read()
    #to send media pipe BGR frames as RGB frames
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #Processes an RGB image and returns a list of the detected face location data
    results = faceDetection.process(imgRGB)
    #FPS info shown
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    if results.detections:
        #enumerating the data (x,y,height,width) for sending the rectangle drawing module properly as needed
        for id,detection in enumerate(results.detections):
           #drawing rectangles and points in multiple faces
            mpDraw.draw_detection(img, detection)
 
    

    #voila
    cv2.imshow("detection", img)
    cv2.waitKey(5)

