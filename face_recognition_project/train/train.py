import os
import torch
from torch.utils.data import DataLoader
from models.backbone import FaceNet
from models.arcface_loss import ArcFaceLoss
from train.dataset import FaceDataset


DATASET = "datasets/aligned"
NUM_CLASSES = len([
    d for d in os.listdir(DATASET)
    if os.path.isdir(os.path.join(DATASET, d))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FaceNet().to(device)
criterion = ArcFaceLoss(NUM_CLASSES).to(device)

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(criterion.parameters()),
    lr=1e-3
)

loader = DataLoader(
    FaceDataset(DATASET),
    batch_size=32,
    shuffle=True,
    num_workers=4,
    drop_last=True
)

for epoch in range(10):
    model.train()
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        embeddings = model(imgs)
        loss = criterion(embeddings, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}: loss={loss.item():.4f}")
