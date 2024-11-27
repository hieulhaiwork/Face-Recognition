import os
import numpy as np
import cv2
from typing import Optional, Dict

import torch
import torchvision.transforms as transforms

from .helper import download_weights, configs
from .model import MobileFacenet

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
    
    def embedding(self, image_dict: Dict[int, np.ndarray]):
        """
        Embedding images got from Dictionary.
        Args:
            - image_dict (Dict[int, np.ndarray]): Aligned image dictionary
        """
        embeddings_list = list(image_dict.values())
        image_tensor = torch.tensor(np.stack(embeddings_list, axis=0)).permute(0, 3, 1, 2).float()
        # Chuẩn hóa

        with torch.no_grad():
            embedding = self.model(image_tensor)
        
        numpy_array = embedding.cpu().numpy()
        embedding_list = [numpy_array[i] for i in range(numpy_array.shape[0])]
        return embedding_list


