import cv2
import numpy as np
from insightface.app import FaceAnalysis

# 1. Load model
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# 2. Đọc ảnh
img1 = cv2.imread("linh.png")
img2 = cv2.imread("nam.png")

# 3. Detect face
faces1 = app.get(img1)
faces2 = app.get(img2)

if len(faces1) == 0 or len(faces2) == 0:
    print("❌ Không phát hiện được khuôn mặt")
    exit()

# 4. Lấy embedding
emb1 = faces1[0].embedding
emb2 = faces2[0].embedding

# 5. Cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

score = cosine_similarity(emb1, emb2)

print("🔢 Similarity score:", score)

# 6. Đánh giá
if score > 0.6:
    print("✅ CÙNG NGƯỜI")
else:
    print("❌ KHÁC NGƯỜI")
