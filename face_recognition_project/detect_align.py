# #Phải detect đúng mặt

# #Phải align (mắt – mũi – miệng thẳng hàng)
# import cv2
# import os
# from insightface.app import FaceAnalysis

# # Khởi tạo InsightFace
# app = FaceAnalysis(
#     name="buffalo_l",
#     allowed_modules=["detection", "landmark"]
# )
# app.prepare(ctx_id=0)  # ctx_id=0: GPU | -1: CPU

# INPUT_DIR = "datasets/raw"
# OUTPUT_DIR = "datasets/aligned"

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# for person in os.listdir(INPUT_DIR):
#     in_dir = os.path.join(INPUT_DIR, person)
#     out_dir = os.path.join(OUTPUT_DIR, person)

#     if not os.path.isdir(in_dir):
#         continue

#     os.makedirs(out_dir, exist_ok=True)

#     for img_name in os.listdir(in_dir):
#         img_path = os.path.join(in_dir, img_name)
#         img = cv2.imread(img_path)

#         if img is None:
#             continue

#         faces = app.get(img)
#         if not faces:
#             continue

#         # Lấy khuôn mặt lớn nhất
#         face = max(
#             faces,
#             key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
#         )

#         # Crop bằng bbox (ổn định)
#         x1, y1, x2, y2 = face.bbox.astype(int)

#         h, w, _ = img.shape
#         x1 = max(0, x1)
#         y1 = max(0, y1)
#         x2 = min(w, x2)
#         y2 = min(h, y2)

#         aligned = img[y1:y2, x1:x2]

#         if aligned.size == 0:
#             continue

#         save_path = os.path.join(out_dir, img_name)
#         cv2.imwrite(save_path, aligned)

# print("✅ Detect & align completed")
