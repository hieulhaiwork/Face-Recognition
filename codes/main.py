import os
import numpy as np
import cv2
import time

from utils import YuNet, VisAlign, VisBase, OpencvAlign, MobileFaceNet_em, FaissDB

detector = YuNet(model_path=None)
vis_align = VisAlign()
vis_base = VisBase()
aligner = OpencvAlign(base_size=256)
embedder = MobileFaceNet_em()

def main():
    image = cv2.imread("images/demo/Zhang_Ziyi_0004.jpg")
    faces_dict = detector.detect(image)
    # vis_base.coord2img(image, faces_dict)
    image_dict = aligner(faces_dict, image)
    image_dict = vis_align.add_name(image_dict)
    # vis_align.show_all(image_dict)
    embeddings_list, dim = embedder.embedding(image_dict)
    database = FaissDB(embedding_dim=dim[0])
    database.new(embeddings_list)

if __name__ == '__main__':
    main()