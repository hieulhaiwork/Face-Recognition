import numpy as np
from sklearn.preprocessing import normalize

# Giả sử bạn có một mảng các vector embedding
embeddings = np.array([[25,4,23,5], [45,8,6,0.2]])

# Chuẩn hóa vector (đảm bảo độ dài mỗi vector = 1)
embeddings_n = normalize(embeddings, axis=1)

print(embeddings)
print("##############################")
print(embeddings_n)


