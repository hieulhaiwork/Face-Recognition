import os
from typing import Dict

import cv2
import numpy as np

class OpencvAlign:
    """
    Class that uses alignment methods from opencv library.
    """
    def __init__(self, base_size: int = 128):
        self.base_size = base_size

    def __call__(self, faces_dict, image: np.ndarray):
        """
        Args:
        - faces_dict (Dict): got from detector, includes "bounding_box", "features", "score"
        - image (np.ndarray): full image, not resized
        """
        # Expected output
        aligned_img = {}

        for key, face_dict in faces_dict.items():
            image_clone = image.copy()
            cropped_face, new_face_dict = self._crop(face_dict, image_clone)
            aligned_face = self._align(new_face_dict, cropped_face)
            aligned_img[key] = aligned_face
        return aligned_img

    def _crop(self, face_dict, image):
        """
        Crop faces by bounding box coords and update features' coords to match with new cropped faces.
        """
        # Crop face
        (xmin, ymin, xmax, ymax) = face_dict['bounding_box']
        cropped_face = image[ymin:ymax, xmin:xmax]

        # Update coords
        features = face_dict['features']
        new_features = []
        for feature in features:
            new_features.append(tuple(a - b for a, b in zip(feature, (xmin,ymin))))
        face_dict['features'] = new_features

        return cropped_face, face_dict


    def _align(self, face_dict: Dict, image: np.ndarray):
        """
        Rotate and resize face to standard configs. 
        """
        h, w = image.shape[:2]
        
        eyes = face_dict['features'][:2]
        right_eye = eyes[0]
        left_eye = eyes[1]

        # Calculate the angle between 2 eyes
        delta_x = right_eye[0] - left_eye[0]
        delta_y = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(delta_y, delta_x)) - 180 

        # Calculate the center between 2 eyes
        center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        
        # Rotate
        M = cv2.getRotationMatrix2D(center, angle, 1)
        aligned_face = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        # Resize
        aligned_face = cv2.resize(aligned_face, (self.base_size, self.base_size))

        return aligned_face