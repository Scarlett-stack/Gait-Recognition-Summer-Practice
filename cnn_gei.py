import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

CASIA_dataset = None
from nm_loader import TRAIN_IDS
from Casia_B_NM_Train import Casia_B_NM_Train as CASIA_dataset

#configurari

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42 
DATA_ROOT = "/home/daria/Documents/PRACTICA-CLEMENTIN/COD/data/CASIA-B-GEI"
SAVE_PATH = "models/gei_cnn_nm.pth"
EPOCHS    = 20
BATCH     = 64
LR        = 1e-3
#nu mai stiu in ce tutorial citisem dar e important sa ai acelasi seed si sa fie setat 
def set_seed(seed=SEED):
    import random, numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(seed=SEED)

#creeam reteaua

class CNN_GEI(nn.Module):
    def __init__(self, num_classes):
        super().__init__() #clasa parinte e nn.Module
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),  # 128x88 -> 64x44
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # 64x44 -> 32x22
            nn.Conv2d(64,128,3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # -> [B,128,1,1]
        )
        self.head = nn.Linear(128, num_classes) #intrare vector[128] iesire vector[num_classes], y = W*x + bias

    def forward(self, x):
        z = self.features(x).flatten(start_dim=1) #dupa avg pool am [B, 128, 1, 1] , vreau [B, 128] pt linear
        return self.head(z) #aplic stratul linear => [B, num_classes]

#load CASIA B GEI
def loader():
    ds = CASIA_dataset(DATA_ROOT)
    dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    return dl

#antrenamentul
def train():
    os.makedirs("models", exist_ok=True)

    num_classes = len(TRAIN_IDS) #74
    model = CNN_GEI(num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr = LR)
    criterion = nn.CrossEntropyLoss() #pt clasificare multi clasa

    dl = loader()

    for epoch in range(1, EPOCHS+1):
        model.train()
        total = 0
        correct = 0
        running_loss = 0.0

        for x, y , _ in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x) #forward pass
            loss = criterion(logits, y) #calculez loss-ul

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()

        avg_loss = running_loss / total
        accuracy = correct / total
        print(f"Epoch {epoch}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    torch.save(model.state_dict(), SAVE_PATH)

if __name__ == "__main__":
    train()