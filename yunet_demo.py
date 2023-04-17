import sys
import numpy as np
import cv2 as cv

from yunet.yunet_face_detector import *

def draw_face_rects(image, face_rects, color):
    for (x1, y1, x2, y2, _) in face_rects:
        cv.rectangle(image, (x1, y1), (x2, y2), color, 2)

# Main function
if __name__ == "__main__":

    yunet_face_detector = YuNetFaceDetector()

    # Open RTSP URL or test video file
    capture = cv.VideoCapture("demo.mp4")

    if not capture.isOpened():
        print("Cannot open video file")
        sys.exit()

    ok, frame = capture.read()
    if not ok:
        print("Cannot read video file")
        sys.exit()


    frame_width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))

    yunet_face_detector.set_input_image_size(frame_width, frame_height)

    tm = cv.TickMeter()

    while True:
        # Read a new frame
        ok, frame = capture.read()
        if not ok:
            break

        if frame is None:
            break

        tm.start()
        faces = yunet_face_detector.detect(frame)
        tm.stop()

        draw_face_rects(frame, faces, (0, 255, 0))

        cv.putText(frame, "FPS: {:.2f}".format(tm.getFPS()), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv.imshow("Face Detection", frame)

        key = cv.waitKey(30)
        if key == 27:
            break

    capture.release()
    cv.destroyAllWindows()
