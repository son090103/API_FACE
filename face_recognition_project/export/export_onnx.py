import torch
from models.backbone import FaceNet


model = FaceNet()
model.load_state_dict(torch.load("model.pth"))
model.eval()


dummy = torch.randn(1,3,112,112)
torch.onnx.export(model, dummy, "face_model.onnx",
input_names=["input"], output_names=["embedding"])