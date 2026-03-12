from fastapi import FastAPI
from face_recognition_project.controller import router

app = FastAPI()

app.include_router(router)
