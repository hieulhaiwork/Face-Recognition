"""
Created by Lê Hiền Hiếu
Github: https://github.com/hieulhaiwork
Mail: hieulh.work@gmail.com
Country: Viet Nam 
"""
import os
import numpy as np
import cv2
import time

from yunet import YuNet
from visualization import Visualize
from align_face import AlignFace
from mobilefacenet import MobileFaceNet_em
from vector_db import VectorDatabase

vis = Visualize()
detector = YuNet(model_path=None)
aligner = AlignFace()
embedder = MobileFaceNet_em(checkpoint_path=None)
database = VectorDatabase(quantized=False)

def main():
    image = cv2.imread("images/Zhang_Ziyi_0001.jpg")
    faces_dict = detector.detect(image)
    # vis.detection_vis(image, faces_dict)
    image_dict = aligner(faces_dict, image)
    # vis.align_vis(image_dict)
    embeddings_list = embedder.embedding(image_dict)
    database.new(embeddings_list)

if __name__ == '__main__':
    main()