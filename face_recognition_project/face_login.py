import base64
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from numpy.linalg import norm
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime

# ===================== MONGODB =====================
# MONGO_URI = "mongodb://localhost:27017/"
# DB_NAME = "DriverSystem"
# USER_COLLECTION = "users"
MONGO_URI = "mongodb+srv://daoson090103_db_user:Np2xYjUwyTQOwb1M@cluster0.5qswe4y.mongodb.net/DriverSystem"
DB_NAME="DriverSystem"
USER_COLLECTION = "users"
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
users_col = db[USER_COLLECTION]

# ===================== LOAD MODEL =====================
face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
face_app.prepare(ctx_id=0, det_size=(640, 640))

# ===================== UTILS =====================
def decode_base64_image(base64_str: str):
    img_bytes = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def cosine_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# ===================== LOGIN BY FACE =====================
def face_login(user_id: str, image_base64: str):
    if not user_id:
        return {"success": False, "message": "Missing user_id"}

    # Convert string -> ObjectId
    try:
        user_object_id = ObjectId(user_id)
    except Exception:
        return {"success": False, "message": "Invalid user_id format"}

    frame = decode_base64_image(image_base64)
    if frame is None:
        return {"success": False, "message": "Invalid image"}

    faces = face_app.get(frame)
    if not faces:
        return {"success": False, "message": "No face detected"}

    current_embedding = faces[0].normed_embedding

    # 🔍 Find user
    user = users_col.find_one({"_id": user_object_id})

    if not user:
        return {"success": False, "message": "User not found"}

    if "face_embedding" not in user:
        return {"success": False, "message": "User has no registered face"}

    stored_embedding = np.array(user["face_embedding"])

    similarity = cosine_sim(current_embedding, stored_embedding) * 100

    return {
        "success": bool(similarity >= 80),
        "similarity": float(round(similarity, 2)),
        "user_id": user_id,
        "time": datetime.utcnow()
    }
