import os
from typing import Optional, Any
import json

import gdown

def load_configs():
    with open('codes/utils/configs.json', 'r') as f:
        configs = json.load(f)
    return configs

def download_weights(filename: str, source_url: str) -> str:
    """
    Download pre-trained model weights if necessary.
    Args:
        - filename (str): Name of weights-saved file, containing file extension.
        - source_url (url): source url of the pre-trained model to download.
    Return:
        weights_file (str): path of weights-saved file.
    """

    assert len(os.path.splitext(filename)) > 1
    
    home = "models"
    saved_path = home + '/' + filename

    if not os.path.exists(home):
        os.makedirs(home)
    
    if os.path.isfile(saved_path):
        print("The weights of this kind of model have been already downloaded.")
        return saved_path
    
    # Download
    try:
            gdown.download(source_url, saved_path, quiet=False)
    except Exception as e:
        raise ValueError(
            f"There is an error when downloading {filename} from {source_url}. " 
            f"Consider download it manually from {source_url} to {saved_path}"
            ) from e
    
    return saved_path
    
    
def load_weight():
    pass

if __name__ == "__main__":
    configs = load_configs()