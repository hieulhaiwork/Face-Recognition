import os
import numpy as np
import cv2
from typing import Optional, Dict

from .helper import download_weights, configs

yunet_configs = configs.get("model", {}).get("yunet", {})
assert yunet_configs != {}

BASE_MODEL_URL = yunet_configs.get("BASE_MODEL_URL", "")
assert BASE_MODEL_URL != ""

# Checking version of opencv
opencv_python_version = lambda str_version: tuple(map(int, str_version.split(".")))
assert opencv_python_version(cv2.__version__) >= opencv_python_version("4.10.0"), "Please install latest opencv version >= 4.10.0"

class YuNet:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = self._build_model()
        
      
    def _build_model(self):
        """
        Build YuNet detector model.
        """
        if not self.model_path:
            self.model_path = download_weights(
                filename="yunet.onnx",
                source_url=BASE_MODEL_URL
            )

        try:
            face_detector = cv2.FaceDetectorYN.create(
                                                model= self.model_path,
                                                config="",
                                                input_size=(0,0),
                                                score_threshold=yunet_configs.get("conf_threshold", 0.8),
                                                nms_threshold=yunet_configs.get("conf_threshold", 0.3),
            )
        except Exception as e:
            print("There is an error occured: ", e)

        return face_detector
    
    def detect(self, image: np.ndarray):
        height, width = image.shape[0], image.shape[1]
        model_input_size = yunet_configs.get("input_size", 640)
        isresized = False

        faces_dict = {}

        if height > model_input_size or width > model_input_size:
            r = model_input_size / max(height, width)
            image = cv2.resize(image, (int(width*r), int(height*r)))
            height, width = image.shape[0], image.shape[1]
            isresized = True

        self.model.setInputSize((width, height))

        _, faces = self.model.detect(image)

        if faces is None:
            return faces_dict
        
        for idx, face in enumerate(faces):
            """
            The format of each face is [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
            where:
                - x, y, w, h: define the bounding box of the face
                - x_re, y_re: locate the right eye
                - x_le, y_le: locate the left eye
                - x_nt, y_nt: locate the nose tip
                - x_rcm, y_rcm, x_lcm, y_lcm: locate the right corner and left corner of mouth
                - score: represents confident score

            """
            (x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm) = list(map(int, face[:14]))

            # YuNet returns negative coordinates if it thinks part of the detected face is outside the frame.
            x = max(x, 0)
            y = max(y, 0)
            if isresized:
                x, y, w, h = int(x / r), int(y / r), int(w / r), int(h / r)
                x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm = (
                    int(x_re / r),
                    int(y_re / r),
                    int(x_le / r),
                    int(y_le / r),
                    int(x_nt / r),
                    int(y_nt / r),
                    int(x_rcm / r),
                    int(y_rcm / r),
                    int(x_lcm / r),
                    int(y_lcm / r),
                )
    
            faces_dict[idx] = {
                'bounding_box': (x, y, x+w, y+h),
                'features': [(x_re, y_re), (x_le, y_le), (x_rcm, y_rcm), (x_lcm, y_lcm)],
                'score': face[-1]
            }

        return faces_dict
        