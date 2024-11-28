import os
from typing import Dict, Any, List

import numpy as np
import cv2

from .vis_base import VisBase

class VisAlign(VisBase):
    def __init__(self):
        super().__init__()

    def add_name(self, img_dict: Dict[Any, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Get names of all people in the images.
        Args:
        - img_dict (dict): Got after alignment process.
        """
        assert type(img_dict) != "dict", f"Invalid type of input, expected dict but got {type(img_dict)}"
        output_dict = {}
        for key, image in img_dict.items():
            assert type(image) != "np.ndarray", f"Invalid type of image, expected numpy.ndarray but got {type(image)}"
            # Show face and get name from keyboard
            self._show(image=image, title=f"Person {key}")
            name = input("Please input name of this person: ")

            # Add to new dict
            output_dict[name] = image
        self._close()
        return output_dict
        
    def show_all(self, img_dict: Dict[str, np.ndarray]):
        """
        Optional visualize all faces with their names.
        Args:
        - img_dict: Output of function self.add_name
        """
        assert type(img_dict) != "dict", f"Invalid type of input, expected dict but got {type(img_dict)}"
        for key, image in img_dict.items():
            assert type(image) != "np.ndarray", f"Invalid type of image, expected numpy.ndarray but got {type(image)}"
            self._show(image, title=f"Name: {key}")
        self._close()




