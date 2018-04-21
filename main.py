import numpy as np
import time
import cv2
import utils
from align_faces import FaceAligner
from predict import FacePredict

net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "resnet10.caffemodel")
print("Model Loaded!")

cap = cv2.VideoCapture(0)
time.sleep(2.0)
print("Starting...")

faceAligner = FaceAligner("dlibModel.dat", 384)


# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
 
    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
 
    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()
    faces = []
    
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        if confidence < 11.2/100:
            continue

        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        (startX, startY, endX, endY) = utils.makeRectToSqr(startX, startY, endX, endY)
        try:
            #Face Alignment
            #faces.append(faceAligner.alignFace((startX, startY, endX, endY), frame))
            #Without Alignment
            faces.append(cv2.resize(frame[startY:endY, startX:endX], (384, 384)))
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 1)
        except:
            pass

    facePredict = FacePredict()
    facePredict.faceCompare(faces)
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()