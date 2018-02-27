import cv2
#WOW!
#imports model - TODO Use different model for better face detection
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    #Loop through the detected faces
    for (x,y,w,h) in faces:
        #Draws rectangle on the face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),4)
        
        #Extracted Faces
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

    #Show image
    cv2.imshow("Video",frame)

    #Wait for escape key to exit the loop
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

#Close all objects
cap.release()
cv2.destroyAllWindows()