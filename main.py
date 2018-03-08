import numpy as np
import time
import cv2
import utils
from align_faces import FaceAligner
import pygame

net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "resnet10SSD.caffemodel")
print("Model Loaded!")

cap = cv2.VideoCapture(0)
time.sleep(2.0)
print("Starting...")

faceAligner = FaceAligner("dlibModel.dat", 384)
pygame.init()
pygame.display.set_caption("OpenCV camera stream on Pygame")
screen = pygame.display.set_mode([1920,1080], pygame.FULLSCREEN)

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

    screen.fill([0,0,0])

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
        except:
            pass

    # show the output frame

    for i in range(10):
        try:
            wow = faces[i]
            wow = np.rot90(wow)
            wow = cv2.cvtColor(wow, cv2.COLOR_BGR2RGB)
            wow = pygame.surfarray.make_surface(wow)
            if(i < 5):
                screen.blit(wow, (384*i, 0))
            else:
                screen.blit(wow, (384*(i-5), 384))
                print("WOW")
            #cv2.imshow("face" + str(i), faces[i])
        except:
            pass
            #cv2.destroyWindow("face" + str(i))

    cv2.imshow("Frame", frame)
    pygame.display.update()
    key = cv2.waitKey(1) & 0xFF
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()