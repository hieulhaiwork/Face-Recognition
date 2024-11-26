import os
import numpy as np
import cv2
from typing import Optional

import torch
import torchvision.transforms as transforms

from utils import download_weights, configs
from model import MobileFacenet

mobilefacenet_configs = configs.get("model", {}).get("mobilefacenet", {})
assert mobilefacenet_configs != {}

BASE_MODEL_URL = mobilefacenet_configs.get("BASE_MODEL_URL", "")
assert BASE_MODEL_URL != ""

class MobileFaceNet_em:
    def __init__(self, checkpoint_path: Optional[str] = None):
        self.checkpoint_path = checkpoint_path
        self.model = self._load_model()
        
    def _load_model(self):
        """
        Build MobileFaceNet embedding model.
        """
        if self.checkpoint_path is None:
            self.checkpoint_path = download_weights(
                filename="mobilefacenet.ckpt",
                source_url=BASE_MODEL_URL
            )

        model = MobileFacenet()

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=torch.device('cpu'), weights_only=True)
            model.load_state_dict(checkpoint['net_state_dict'])
            model.eval()
        except Exception as e:
            print("There is an error occured: ", e)

        return model
    
    def embedding(self, image: np.ndarray):

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)

        with torch.no_grad():
            embedding = self.model(image_tensor)
            return embedding


