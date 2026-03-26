import os
import base64
import cv2
import numpy as np
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
        img_bytes = base64.b64decode(base64_str)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception:
        return None

# ===================== REGISTER / UPDATE FACE =====================
def face_register(user_id: str, image_base64: str):
    if not user_id:
        return {"success": False, "message": "Missing user_id"}

    try:
        user_object_id = ObjectId(user_id)
    except Exception:
        return {"success": False, "message": "Invalid user_id format"}

    frame = decode_base64_image(image_base64)
    if frame is None:
        return {"success": False, "message": "Invalid image"}

    face_app = get_face_app()
    faces = face_app.get(frame)
    if len(faces) == 0:
        return {"success": False, "message": "No face detected"}

    embedding = faces[0].normed_embedding.tolist()

    result = users_col.update_one(
        {"_id": user_object_id},
        {
            "$set": {
                "face_embedding": embedding,
                "face_updated_at": datetime.utcnow()
            }
        }
    )

    if result.matched_count == 0:
        return {"success": False, "message": "User not found"}

    return {
        "success": True,
        "message": "Face registered successfully",
        "user_id": user_id
    }