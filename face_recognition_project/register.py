import cv2
import numpy as np
import os
import onnxruntime as ort
from insightface.app import FaceAnalysis
from pymongo import MongoClient
from datetime import datetime
client = MongoClient("mongodb://localhost:27017/")
db = client["face_auth"]
users_col = db["users"]
# ================= CHECK GPU =================
print("ONNX Providers:", ort.get_available_providers())

# ================= CONFIG =================
USER_NAME = "son"
SAVE_DIR = "embeddings"
os.makedirs(SAVE_DIR, exist_ok=True)

# ================= INIT MODEL (GPU) =================
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

print("📸 Yêu cầu quay mặt: TRÁI → GIỮA → PHẢI")

# ================= STATE =================
state = 0   # 0=trái | 1=giữa | 2=phải
embeddings = []

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.resize(frame, (640, 480))
    faces = app.get(frame)

    if len(faces) == 1:
        face = faces[0]
        yaw = face.pose[0]   # quay trái/phải
        emb = face.normed_embedding

        x1, y1, x2, y2 = map(int, face.bbox)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        # ========= LOGIC QUAY =========
        if state == 0:
            cv2.putText(frame, "Turn LEFT", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            if yaw < -15:
                embeddings.append(emb)
                state = 1

        elif state == 1:
            cv2.putText(frame, "LOOK STRAIGHT", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            if -10 < yaw < 10:
                embeddings.append(emb)
                state = 2

        elif state == 2:
            cv2.putText(frame, "Turn RIGHT", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            if yaw > 15:
                embeddings.append(emb)

                # ====== SAVE AVERAGE EMBEDDING ======
                #final_emb = np.mean(embeddings, axis=0)
                #save_path = os.path.join(SAVE_DIR, f"{USER_NAME}.npy")
                #np.save(save_path, final_emb)
                # ====== SAVE AVERAGE EMBEDDING ======
                final_emb = np.mean(embeddings, axis=0)

                user_doc = {
                    "username": USER_NAME,
                    "embedding": final_emb.tolist(),  # ⚠️ numpy → list
                    "created_at": datetime.utcnow()
                }

                # nếu user tồn tại thì update, chưa có thì insert
                users_col.update_one(
                    {"username": USER_NAME},
                    {"$set": user_doc},
                    upsert=True
                )

                print("✅ ĐÃ LƯU KHUÔN MẶT VÀO MONGODB")
                break


                print("✅ ĐĂNG KÝ THÀNH CÔNG (LEFT → CENTER → RIGHT)")
                break

    elif len(faces) > 1:
        cv2.putText(frame, "ONLY ONE FACE!",
                    (20,40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,0,255), 2)

    else:
        cv2.putText(frame, "NO FACE",
                    (20,40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,0,255), 2)

    cv2.imshow("Register Face (Turn Head)", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
