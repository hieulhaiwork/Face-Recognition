# Face Recognition Project

In many situations, if we want to deploy a facial recognition device to protect private areas from strangers by security doors or something like that, the edge devices, limited in hardware, can make the facial recognition process slow and unreliable if using inappropriate models. 

This is my personal project where I will deploy a pre-trained YuNet model for face detection, MobileFaceNet for face embedding and combined with an ANN method to retrieve faces in the vector database (FAISS) to ensure accuracy and execution time. Also, set up a mechanism to add new faces without retraining the models. 

### YuNet

YuNet is a light-weight, fast and accurate face detection model, which achieves 0.834(easy), 0.824(medium), 0.708(hard) on the WIDER Face validation set.

### ANN

