import onnxruntime as ort
import numpy as np
import cv2


sess = ort.InferenceSession("face_model.onnx")


img = cv2.imread("aligned.jpg")
img = cv2.resize(img, (112,112))
img = img.transpose(2,0,1)[None].astype(np.float32)


embedding = sess.run(None, {"input": img})[0]
print(embedding.shape)