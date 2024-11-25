import os
import numpy as np

import cv2

class YuNet:
    def __init__(self, modelPath: str, 
                 inputSize = [320, 320],
                 confThreshold = 0.6,
                 nmsThreshold=0.3, 
    ):
        self.modelPath = modelPath
        self.inputSize = inputSize,
        self.confThreshold = confThreshold,
        self.nmsThreshold = nmsThreshold,

        self.model = cv2.FaceDetectorYN.create(
            modelPath=self.modelPath,
            inputSize=self.inputSize,
            score_threshold=self.confThreshold,
            nms_threshold = self.nmsThreshold
        )
    
    def run(self, image):
        faces, _ = self.model.detect(image)
        