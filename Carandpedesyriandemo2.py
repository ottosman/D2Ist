import cv2

cap = cv2.VideoCapture('Cars_traffic.mp4')
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')
pedestrian_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml')

while True:
     
    read_successful, frames = cap.read()
    
    if read_successful:
        grayscaled_image = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    else:
        break
    
    cars = car_cascade.detectMultiScale(grayscaled_image)
    # pedestrians = pedestrian_cascade.detectMultiScale(frames)
    # faces = face_cascade.detectMultiScale(frames)

    for (x,y,w,h) in cars:
        cv2.rectangle(frames, (x,y), (x+w,y+h), (0, 0, 255), 3)

    # for (x, y, w, h) in pedestrians:
    #     cv2.rectangle(frames, (x,y), (x+w, y+h), (0, 255, 0), 2)
        
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frames, (x,y), (x+w, y+h), (0, 255, 255), 2)

    if cv2.waitKey(33) == 13:
        break
    
    cv2.imshow('aaa',frames)


cv2.destroyAllWindows()

