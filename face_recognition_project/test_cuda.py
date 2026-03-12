import torch
print("START")
print("CUDA available:", torch.cuda.is_available())

x = torch.rand(3000, 3000, device="cuda")
print("DONE", x.sum())
