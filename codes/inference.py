import os
import cv2

from utils import YuNet, OpencvAlign, MobileFaceNet_em, FaissDB, VisPredict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

detector = YuNet(model_path=None)
vis_predict = VisPredict()
aligner = OpencvAlign(base_size=256)
embedder = MobileFaceNet_em()

def inference():
    image = cv2.imread("images/demo/Zhang_Ziyi_0002.jpg")
    faces_dict = detector.detect(image)
    image_dict = aligner(faces_dict, image)
    embeddings_list, dim = embedder.embedding(image_dict)
    database = FaissDB(embedding_dim=dim[0])
    print(database._show_names())
    name_list = []
    distance_list = []
    for (i, embedding) in embeddings_list:
        distance, index, name = database.get_name(embedding)
        name_list.append(name)
        distance_list.append(distance)
    distance_list = [int(x[0, 0]) for x in distance_list]
    vis_predict(image, faces_dict, name_list, distance_list)
    
if __name__ == "__main__":
    inference()