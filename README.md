# Face Recognition Project

In many situations, if we want to deploy a facial recognition device to protect private areas from strangers by security doors or something like that, the edge devices, limited in hardware, can make the facial recognition process slow and unreliable if using inappropriate models. 

This is my personal project where I will deploy a pre-trained `YuNet` model for face detection, `MobileFaceNet` for facial features embedding, then applying some techniques: `Product Quantization` and `Inverted File Inde` to enhace the performance of retrieving in the vector database (`FAISS`) to ensure accuracy and execution time. Also, set up a mechanism to add new faces without retraining the models. 

### YuNet

YuNet is a light-weight, fast and accurate face detection model, which achieves 0.834(easy), 0.824(medium), 0.708(hard) on the WIDER Face validation set.

### MobileFaceNet

### FAISS

Why FAISS? Simply, I am looking for a simple and quick database to deploy this project. Otherwise, FAISS supports many techniques for retrieving similar vectors in the large database such as Product Quantization and Inverted File Index. 

In reality, the face recognition system is usually accompanied by an admin system controlling all sub-systems with the same database. In this situation, we can use some databases which support well for distributed systems like Milvus, Pinecone,...

### Future Improvement

With Product Quantization and Inverted File Index approaches, it requires a sample of faces to train the FAISS database to get `codebook` which is similar to "rules" the database follows to store vector data. I have already trained to get pre-trained codebook from dataset to use in this project. This may be annoyed in some situations.

Moreover, I believe my codes is not the best solution when applying in edge devices because of my weakness in optimizing at the time I code this repo. I will improve it in the future.

### Run




