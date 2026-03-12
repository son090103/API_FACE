# Phát hiện & căn chỉnh khuôn mặt
import cv2
import os
from insightface.app import FaceAnalysis


app = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection", "landmark"]
)
app.prepare(ctx_id=0)


INPUT_DIR = "datasets/raw"
OUTPUT_DIR = "datasets/aligned"
os.makedirs(OUTPUT_DIR, exist_ok=True)


for person in os.listdir(INPUT_DIR):
    in_dir = os.path.join(INPUT_DIR, person)
    out_dir = os.path.join(OUTPUT_DIR, person)
    os.makedirs(out_dir, exist_ok=True)

    for img_name in os.listdir(in_dir):
        img_path = os.path.join(in_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        faces = app.get(img)
        if not faces:
            continue

        face = faces[0]  # lấy khuôn mặt lớn nhất/đầu tiên
        aligned = face.crop_img

        cv2.imwrite(os.path.join(out_dir, img_name), aligned)
