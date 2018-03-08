import dlib
import cv2
from imutils.face_utils import FaceAligner as otherFaceAligner
from utils import shape_to_np

class FaceAligner:
    def __init__(self, modelPath, size):
        self.predictor = dlib.shape_predictor(modelPath)
        self.fa = otherFaceAligner(self.predictor, desiredFaceWidth=size)
    def alignFace(self, bounding_box, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rect = dlib.rectangle(bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3])
        return self.fa.align(img, gray, rect)
    def getLandmark(self, bounding_box, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rect = dlib.rectangle(bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3])
        shape = self.predictor(gray, rect)
        shape = shape_to_np(shape)
        return shape