import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, embedding_size=512, s=30.0, m=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.ce = nn.CrossEntropyLoss()

    def forward(self, embeddings, labels):
        # 1️⃣ Normalize embeddings & weights
        embeddings = F.normalize(embeddings)
        weight = F.normalize(self.weight)

        # 2️⃣ Cosine similarity
        cosine = F.linear(embeddings, weight)  # = embeddings @ weight.T

        # 3️⃣ Add angular margin
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cosine)
        cosine_m = torch.cos(theta + self.m)

        # 4️⃣ One-hot labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # 5️⃣ Apply margin only to target class
        output = (one_hot * cosine_m) + ((1.0 - one_hot) * cosine)

        # 6️⃣ Scale
        output *= self.s

        # 7️⃣ Cross entropy
        loss = self.ce(output, labels)
        return loss
