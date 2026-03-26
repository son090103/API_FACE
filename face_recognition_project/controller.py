from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from face_login import face_login
from face_register import face_register 
# from face_recognition_project.face_login import face_login
# from face_recognition_project.face_register import face_register
# ================= APP =================
app = FastAPI()

# ================= ROUTER =================
router = APIRouter()


class FaceLoginRequest(BaseModel):
    user_id: str
    image: str

class FaceRegisterRequest(BaseModel):
    user_id: str
    image: str

@router.post("/face-login")
def face_login_controller(data: FaceLoginRequest):
    try:
        return face_login(user_id=data.user_id,image_base64=data.image)
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@router.post("/face-register")
def face_register_controller(data: FaceRegisterRequest):
    try:
        return face_register(
            user_id=data.user_id,
            image_base64=data.image
        )
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


app.include_router(router)