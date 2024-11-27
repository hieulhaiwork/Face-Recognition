import os
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

def inference():
    image_test = cv2.imread("images/Untitled.png")
    faces_dict = detector.detect(image_test)
    image_dict = aligner(faces_dict, image_test)
    embeddings_list = embedder.embedding(image_dict)
    name_list = []
    distance_list = []
    for embedding in embeddings_list:
        distance, index, name = database.get_name(embedding)
        name_list.append(name)
        distance_list.append(distance)
    # vis(image_test, faces_dict, name_list, distance_list)

if __name__ == "__main__":
    inference()