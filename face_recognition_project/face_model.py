from insightface.app import FaceAnalysis

# 🔥 load ngay khi import
face_app = FaceAnalysis(
    name="buffalo_s",
    providers=["CPUExecutionProvider"]
)

face_app.prepare(ctx_id=0, det_size=(640, 640))

print("✅ InsightFace model loaded (CPU)")


def get_face_app():
    return face_app
