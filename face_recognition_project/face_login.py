import os
import base64
import cv2
import numpy as np
from numpy.linalg import norm
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime

from face_recognition_project.face_model import get_face_app

# ===================== MONGODB =====================
MONGO_URI = os.environ.get(
    "MONGO_URI",
    "mongodb+srv://daoson090103_db_user:Np2xYjUwyTQOwb1M@cluster0.5qswe4y.mongodb.net/DriverSystem"
)
DB_NAME = os.environ.get("DB_NAME", "DriverSystem")
USER_COLLECTION = "users"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
users_col = db[USER_COLLECTION]

# ===================== UTILS =====================
def decode_base64_image(base64_str: str):
    try:
        # 🔥 FIX: xử lý data:image/...;base64,...
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]

        img_bytes = base64.b64decode(base64_str)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        return img
    except Exception as e:
        print("❌ Decode error:", e)
        return None


def cosine_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


# ===================== LOGIN BY FACE =====================
def face_login(user_id: str, image_base64: str):
    if not user_id:
        return {"success": False, "message": "Missing user_id"}

    # 🔥 check ObjectId
    try:
        user_object_id = ObjectId(user_id)
    except Exception:
        return {"success": False, "message": "Invalid user_id format"}

    # 🔥 decode ảnh
    frame = decode_base64_image(image_base64)
    if frame is None:
        return {"success": False, "message": "Invalid image"}

    # 🔥 detect face
    face_app = get_face_app()
    faces = face_app.get(frame)

    print("🔍 Faces detected:", len(faces))

    if not faces:
        return {"success": False, "message": "No face detected"}

    # 🔥 FIX: lấy mặt rõ nhất (quan trọng)
    face = max(faces, key=lambda x: x.det_score)
    current_embedding = face.normed_embedding

    print("🧠 Current embedding (first 5):", current_embedding[:5])

    # 🔥 lấy user từ DB
    user = users_col.find_one({"_id": user_object_id})
    if not user:
        return {"success": False, "message": "User not found"}

    if "face_embedding" not in user:
        return {"success": False, "message": "User has no registered face"}

    stored_embedding = np.array(user["face_embedding"])

    print("💾 Stored embedding length:", len(stored_embedding))
    print("💾 Stored embedding (first 5):", stored_embedding[:5])

    # 🔥 tính similarity (KHÔNG nhân 100 ở đây)
    score = cosine_sim(current_embedding, stored_embedding)

    print("📊 Cosine score:", score)

    # 🔥 threshold chuẩn
    is_match = score >= 0.5   # bạn có thể chỉnh 0.4 - 0.6

    return {
        "success": bool(is_match),
        "similarity": float(round(score * 100, 2)),
        "user_id": user_id,
        "time": datetime.utcnow()
    }
