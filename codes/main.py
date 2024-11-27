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

t0 = time.time()
vis = Visualize()
detector = YuNet(model_path=None)
aligner = AlignFace()
embedder = MobileFaceNet_em(checkpoint_path=None)
database = VectorDatabase(quantized=False)
t1 = time.time()

def main():
    t2 = time.time()
    image = cv2.imread("images/Untitled.png")
    faces_dict = detector.detect(image)
    t3 = time.time()
    # vis.detection_vis(image, faces_dict)
    image_dict = aligner(faces_dict, image)
    t4 = time.time()
    # vis.align_vis(image_dict)
    embeddings_list = embedder.embedding(image_dict)
    t5 = time.time()
    database.new(embeddings_list)
    t6 = time.time()
    print("Thời gian load các hàm: ", t1 - t0)
    print("Thời gian detect: ", t3 - t2)
    print("Thời gian align: ", t4 - t3)
    print("Thời gian embedding: ", t5 - t4)
    print("Tổng time: ", t6 - t0)

if __name__ == '__main__':
    main()