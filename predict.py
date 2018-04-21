import tensorflow as tf
import cv2

class FacePredict(object):

    def faceCompare(self, faces): 
        #print("Begin Compare.")
        gray = cv2.cvtColor(faces[0], cv2.COLOR_BGR2GRAY)

        ###for idx, i in enumerate(faces):
        ### windowName = "Face - " + str(idx)
        ### cv2.imshow(windowName, i)
        
        #TODO Use this photo and predicts

        