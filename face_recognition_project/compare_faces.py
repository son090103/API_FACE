# import cv2
# import numpy as np
# from insightface.app import FaceAnalysis

# app = FaceAnalysis(name="buffalo_l")
# app.prepare(ctx_id=-1)  # CPU

# def get_embedding(img_path):
#     img = cv2.imread(img_path)
#     faces = app.get(img)
#     if len(faces) == 0:
#         return None
#     return faces[0].embedding

# def cosine_similarity(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# img1 = "datasets/aligned/linh/linh.png"
# img2 = "datasets/raw/linh/nam.png"

# emb1 = get_embedding(img1)
# emb2 = get_embedding(img2)

# if emb1 is None or emb2 is None:
#     print("❌ Không detect được mặt")
# else:
#     sim = cosine_similarity(emb1, emb2)
#     print("Similarity:", sim)

#     if sim > 0.5:
#         print("✅ Cùng người")
#     else:
#         print("❌ Khác người")
