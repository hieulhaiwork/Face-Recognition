import os
from typing import Dict, Any, List

import numpy as np
import cv2

from .vis_base import VisBase

class VisPredict(VisBase):
    def __init__(self):
        super().__init__()

    def __call__(self, image, faces_dict, names_list, distance_list):
        for key, face in faces_dict.items():
            bounding_box = face['bounding_box']
            cv2.rectangle(image, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (0,255,0), 1)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, f"{names_list[key]}, {distance_list[key]}", (bounding_box[0], bounding_box[1] - 10), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imshow("Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return True