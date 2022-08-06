import cv2
import numpy as np


class YOLO:
    def __init__(self, weights_path: str, cfg_path: str,
    nms_thr: float=0.4, conf_thr: float=0.5,
    img_size: float=416, enable_cuda: float=True) -> None:

        self.nmsThreshold = nms_thr
        self.confThreshold = conf_thr
        self.image_size = img_size

        net = cv2.dnn.readNet(weights_path, cfg_path)

        if enable_cuda:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.model = cv2.dnn_DetectionModel(net)

        self.model.setInputParams(size=(self.image_size, self.image_size),
        scale=1/255)

    def detect(self, frame: np.ndarray):
        return self.model.detect(frame, nmsThreshold=self.nmsThreshold,
        confThreshold=self.confThreshold)

