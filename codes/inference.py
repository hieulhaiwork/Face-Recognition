import os
import cv2
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from utils.yunet import YuNet
from utils.visualization import Visualize
from utils.align_face import AlignFace
from utils.mobilefacenet import MobileFaceNet_em
from utils.vector_db import VectorDatabase

t0 = time.time()
vis = Visualize()
detector = YuNet(model_path=None)
aligner = AlignFace()
embedder = MobileFaceNet_em(checkpoint_path=None)
database = VectorDatabase(quantized=False)

def inference():
    image_test = cv2.imread("images/Zhang_Ziyi_0002.jpg")
    faces_dict = detector.detect(image_test)
    image_dict = aligner(faces_dict, image_test)
    embeddings_list = embedder.embedding(image_dict)
    name_list = []
    distance_list = []
    for embedding in embeddings_list:
        distance, index, name = database.get_name(embedding)
        name_list.append(name)
        distance_list.append(distance)
    distance_list = [int(x[0, 0]) for x in distance_list]
    print(distance_list)
    t1 = time.time()
    vis(image_test, faces_dict, name_list, distance_list)
    print("Inference time: ", t1 - t0)
if __name__ == "__main__":
    inference()