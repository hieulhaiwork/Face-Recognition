import os
from typing import Union, Tuple, Dict, Any, List

import cv2
import numpy as np

class VisBase:
    def __init__(self):
        self.color = [
            (0, 0, 255),        # red
            (255, 0, 0),        # blue
            (0, 165, 255),      # orange
            (0, 255, 255),      # yellow
            (128, 0, 128)       # purple
        ]

    def _show(self, image: np.ndarray, title: str, waitkey: int = 0):
        """
        Show image.
        """
        cv2.imshow(f"{title}", image)
        cv2.waitKey(waitkey)
    
    def _close(self):
        cv2.destroyAllWindows()

    # Optional for visualizing faces detection process
    def coord2img(self, image: np.ndarray, faces_dict: Dict[str, Union[Tuple[int, int], List[Tuple[int, int]]]]):
        for key, face in faces_dict.items():
            bounding_box = face['bounding_box']
            cv2.rectangle(image, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (0,255,0), 1)

            for color, feature in enumerate(face['features']):
                cv2.circle(image, feature, radius=5, color=self.color[color])
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, f"Score: {face['score']:.2f}", (bounding_box[0], bounding_box[1] - 10), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Detector", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return True

