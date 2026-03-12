import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class FaceDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        self.transform = T.Compose([
            T.Resize((112, 112)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        for label, person in enumerate(os.listdir(root)):
            person_dir = os.path.join(root, person)
            if not os.path.isdir(person_dir):
                continue
            for img in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img)
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label
