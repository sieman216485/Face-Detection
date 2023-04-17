import sys
import numpy as np
import cv2 as cv

from haarcascades.haarcascades_face_detector import *

def draw_rects(image, rects, color):
    for x1, y1, x2, y2 in rects:
        cv.rectangle(image, (x1, y1), (x2, y2), color, 2)

# Main function
if __name__ == "__main__":

    haarcascades_face_detector = HaarcascadesFaceDetector()

    # Open RTSP URL or test video file
    capture = cv.VideoCapture("demo.mp4")

    if not capture.isOpened():
        print("Cannot open video file")
        sys.exit()

    ok, frame = capture.read()
    if not ok:
        print("Cannot read video file")
        sys.exit()

    tm = cv.TickMeter()

    while True:
        # Read a new frame
        ok, frame = capture.read()
        if not ok:
            break

        if frame is None:
            break

        tm.start()
        result = haarcascades_face_detector.detect(frame, True)
        tm.stop()

        for face_rect, eye_rects in result:
            draw_rects(frame, [face_rect], (0, 255, 0))
            draw_rects(frame, eye_rects, (0, 0, 255))

        cv.putText(frame, "FPS: {:.2f}".format(tm.getFPS()), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv.imshow("Face Detection", frame)

        key = cv.waitKey(30)
        if key == 27:
            break

    capture.release()
    cv.destroyAllWindows()
