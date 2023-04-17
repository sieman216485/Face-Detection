import os
import cv2 as cv

_SCALE_FACTOR = 1.3
_MIN_NEIGHBORS = 4
_MIN_WIDTH = 30
_MIN_HEIGHT = 30

class HaarcascadesFaceDetector:

    def __init__(self):
        self._face_cascade = cv.CascadeClassifier(os.path.join(os.path.dirname(__file__), "haarcascades/haarcascade_frontalface_default.xml"))
        self._eye_cascade = cv.CascadeClassifier(os.path.join(os.path.dirname(__file__), "haarcascades/haarcascade_eye.xml"))

    def _detect(self, cascade, image):
        rects = cascade.detectMultiScale(image, scaleFactor = _SCALE_FACTOR, minNeighbors = _MIN_NEIGHBORS, minSize = (_MIN_WIDTH, _MIN_HEIGHT), flags = cv.CASCADE_SCALE_IMAGE)

        if len(rects) == 0:
            return []

        rects[:, 2:] += rects[:, :2]
        return rects

    def detect(self, image, detect_eye = False):
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray_image = cv.equalizeHist(gray_image)

        face_rects = self._detect(self._face_cascade, image)

        if len(face_rects) == 0:
            return []

        result = []

        if detect_eye:
            for x1, y1, x2, y2 in face_rects:
                roi = gray_image[y1:y2, x1:x2]

                eye_rects = self._detect(self._eye_cascade, roi.copy())

                if len(eye_rects) > 0:
                    # eye_rects = [(x1 + eye_x1, y1 + eye_y1, x1 + eye_x2, y1 + eye_y2) for eye_x1, eye_y1, eye_x2, eye_y2 in eye_rects]
                    eye_rects[:, :] += (x1, y1, x1, y1)

                result.append(((x1, y1, x2, y2), eye_rects))

        return result
