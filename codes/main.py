"""
Created by Lê Hiền Hiếu
Github: https://github.com/hieulhaiwork
Mail: hieulh.work@gmail.com
Country: Viet Nam 
"""
import os
import numpy as np
import cv2

from yunet import YuNet
from visualization import Visualize
from align_face import AlignFace
from mobilefacenet import MobileFaceNet_em

vis = Visualize()
detector = YuNet()
aligner = AlignFace()
embedder = MobileFaceNet_em()

def main():
    image = cv2.imread("images/Untitled.png")
    faces_dict = detector.detect(image)
    # vis.detection_vis(image, faces_dict)
    image_dict = aligner(faces_dict, image)
    # vis.align_vis(image_dict)
    for key, image in image_dict.items():
        embedding = embedder.embedding(image)
    print(embedding.numpy().shape)


if __name__ == '__main__':
    main()