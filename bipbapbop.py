
import cv2


trained_Face_data = cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml')
trained_Face2_data = cv2.CascadeClassifier('haarcascade_frontalcatface_extended.xml')
trained_Face3_data = cv2.CascadeClassifier('haarcascade_profileface.xml')



webcam = cv2.VideoCapture(0)
start_frame_number = 5
# webcam.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
webcam.set(cv2.CAP_PROP_FRAME_COUNT,start_frame_number)
length_of_frame = int(webcam.set(cv2.CAP_PROP_FRAME_COUNT,5))


while True: 
    successful_frame_read, frame = webcam.read()
    
    
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_Face_data.detectMultiScale(frame)
    face_coordinates = trained_Face2_data.detectMultiScale(frame)
    face_coordinates = trained_Face3_data.detectMultiScale(frame)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 8)


    print(length_of_frame)

    cv2.imshow("aa",frame)

    cv2.waitKey(1)



