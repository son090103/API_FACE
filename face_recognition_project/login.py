# import cv2
# import numpy as np
# from insightface.app import FaceAnalysis
# from numpy.linalg import norm
# import onnxruntime as ort

# # ================= CHECK GPU =================
# print("ONNX Providers:", ort.get_available_providers())

# # ================= HÀM COSINE =================
# def cosine_sim(a, b):
#     return np.dot(a, b) / (norm(a) * norm(b))

# # ================= LOAD EMBEDDING =================
# known_emb = np.load("embeddings/son.npy")

# # ================= INIT FACE APP (GPU) =================
# app = FaceAnalysis(
#     name="buffalo_l",
#     providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
# )

# app.prepare(
#     ctx_id=0,          # 0 = GPU | -1 = CPU
#     det_size=(640, 640)
# )

# # ================= CAMERA =================
# cap = cv2.VideoCapture(0)

# frame_id = 0
# similarity_percent = 0.0

# # ================= LOOP =================
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         continue

#     frame_id += 1
#     frame = cv2.resize(frame, (640, 480))

#     faces = []
#     if frame_id % 5 == 0:   # giảm tải GPU
#         faces = app.get(frame)

#     if len(faces) > 0:
#         face = faces[0]

#         # VẼ KHUNG MẶT
#         x1, y1, x2, y2 = map(int, face.bbox)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#         emb = face.normed_embedding
#         sim = cosine_sim(emb, known_emb)
#         similarity_percent = sim * 100

#     # ================= HIỂN THỊ % GÓC TRÁI =================
#     cv2.putText(
#         frame,
#         f"Similarity: {similarity_percent:.1f}%",
#         (20, 40),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1,
#         (0, 255, 0),
#         2
#     )

#     cv2.imshow("Face Similarity (GPU)", frame)

#     if cv2.waitKey(1) == 27:  # ESC để thoát
#         break

# # ================= CLEAN =================
# cap.release()
# cv2.destroyAllWindows()
