import cv2
import os

os.chdir("/")

#TODO: Fix Error
#imports model - TODO Use different model for better face detection
net = cv2.dnn.readNetFromCaffe("deploy.prototext.txt", "model")
#net = cv2.dnn.readNetFromCaffe("deploy.prototext.txt", "resnet10.caffemodel")

cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()

    #Max with is 400 px
    frame = cv2.resize(frame, (711, 400))

    #Convert image to blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
	# predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
        confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
        if confidence < 0.1:
            continue

		# compute the (x, y)-coordinates of the bounding box for the
		# object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
 
		# draw the bounding box of the face along with the associated
		# probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    #Show image
    cv2.imshow("Video",frame)

    #Wait for escape key to exit the loop
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

#Close all objects
cap.release()
cv2.destroyAllWindows()