import os
import numpy as np
import cv2 as cv

_CONFIDENCE_THRESHOLD = 0.9 # SCORE_THRESHOLD
_NMS_THRESHOLD = 0.3
_TOP_K = 5000

class YuNetFaceDetector:

    def __init__(self, scale = 1.0):
        self._scale = scale

        # Valid combinations of backends and targets
        backend_target_pairs = [
            [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],         # 0: OpenCV + CPU
            [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],        # 1: CUDA + GPU (CUDA)
            [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],   # 2: CUDA + GPU (CUDA FP16)
            [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],         # 3: TIM-VX + NPU
            [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]          # 4: CANN + NPU
        ]

        backend_target = 0

        backend_id = backend_target_pairs[backend_target][0]
        target_id = backend_target_pairs[backend_target][1]

        self._detector = cv.FaceDetectorYN.create(
            os.path.join(os.path.dirname(__file__), "models/face_detection_yunet_2022mar.onnx"),
            "",
            (320, 320),
            _CONFIDENCE_THRESHOLD,
            _NMS_THRESHOLD,
            _TOP_K,
            backend_id,
            target_id
        )

    def set_input_image_size(self, width, height):
        self._input_image_width = int(width * self._scale)
        self._input_image_height = int(height * self._scale)

        self._detector.setInputSize([self._input_image_width, self._input_image_height])

    def detect(self, image):
        image = cv.resize(image, (self._input_image_width, self._input_image_height))

        faces = self._detector.detect(image)

        if faces[1] is None:
            return []

        face_rects = []

        for face in faces[1]:
            confidence = face[-1]
            coords = (face[:-1] / self._scale).astype(np.int32)

            face_rect = (coords[0], coords[1], (coords[0] + coords[2]), (coords[1] + coords[3]), confidence)

            face_rects.append(face_rect)

        return face_rects
