import os
import numpy as np
import cv2
from typing import Dict, Tuple, List

class Visualize:
    def __init__(self):
        self.color = [(0, 0, 255), (255, 0, 0), (0, 165, 255), (0, 255, 255), (128, 0, 128)] # red, blue, orange, yellow, purple

    def detection_vis(self, image: np.ndarray, faces_dict: Dict[str, Tuple[int, int]]):
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
    
    def align_vis(self, image_dict: Dict[str, np.ndarray]):
        for key, image in image_dict.items():
            cv2.imshow(f"Key {key}", image)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def __call__(self, image: np.ndarray, faces_dict: Dict[str, Tuple[int, int]], name_list: List[str], distance_list: List[float]):
        for key, face in faces_dict.items():
            # if distance_list[key] > 100:
            #     break
            # else:
                bounding_box = face['bounding_box']
                cv2.rectangle(image, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (0,255,0), 1)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, f"Name: {name_list[key]}, Distance: {distance_list[key]:.2f}", (bounding_box[0], bounding_box[1] - 10), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Detector", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return True
        

