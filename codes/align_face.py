import os
import numpy as np

import cv2

class AlignFace:
    def __init__(self):
        self.faces = {}

    def __call__(self, faces_dict, image):
        image_dict = {}
        for key, face_dict in faces_dict.items():
            image_clone = image.copy()
            cropped_face, new_face_dict = self._crop(face_dict, image_clone)
            aligned_face = self._align(new_face_dict, cropped_face)
            image_dict[key] = aligned_face
        return image_dict

    def _crop(self, face_dict, image):
        """
        """
        (xmin, ymin, xmax, ymax) = face_dict['bounding_box']
        cropped_face = image[ymin:ymax, xmin:xmax]

        features = face_dict['features']
        new_features = []
        for t in features:
            new_features.append(tuple(a - b for a, b in zip(t, (xmin,ymin))))
        face_dict['features'] = new_features

        return cropped_face, face_dict


    def _align(self, face_dict, image):
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
        
        M = cv2.getRotationMatrix2D(center, angle, 1)

        aligned_face = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        return aligned_face