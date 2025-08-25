
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.seq_loader import SequenceDataset as SeqDataset
from models.cnn_gru import CNN_GRU

# 3) iau IDs pt train
try:
    from nm_loader import TRAIN_IDS
except Exception:
    TRAIN_IDS = [f"{i:03d}" for i in range(1,75)]  # fallback 001..074

# ------------------ CONFIG ------------------
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SEED        = 42
ROOT_FRAMES = "datasets/output"     
SEQ_KINDS   = [f"nm-0{i}" for i in range(1,7)]  # nm-01..nm-06
L_FRAMES    = 20                    # nr. cadre / secventa
BATCH       = 8                     # samples/secventa per batch
EPOCHS      = 20
LR          = 1e-3
WD          = 1e-4                  # weight decay mic, ajuta la generalizare
CLIP_NORM   = 1.0                   # gradient clipping
SAVE_PATH   = "models/cnn_gru_nm.pth"
AMP         = True                  # mixed precision pe GPU (rapid & memorie)
NUM_WORKERS = 2                     # DataLoader

# --------------------------------------------

def set_seed(seed=SEED):
    import random, numpy as np
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_loader():
    ds = SeqDataset(
        root=ROOT_FRAMES,
        subject_ids=TRAIN_IDS,
        seq_ids=SEQ_KINDS,
        L=L_FRAMES
    )
    dl = DataLoader(
        ds, batch_size=BATCH, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False
    )
    return dl

def train():
    set_seed(SEED)
    os.makedirs("models", exist_ok=True)

    # 1) Date + model
    dl = make_loader()
    num_classes = len(TRAIN_IDS)  # 74
    model = CNN_GRU(
        num_classes=num_classes,
        feat_dim=128, gru_hidden=128,
        num_layers=1, bidirectional=False,
        proj_to_feat=True, dropout=0.1
    ).to(DEVICE)

    # 2) Optimizare + loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.CrossEntropyLoss()

    # 3) Mixed precision 
    scaler = torch.cuda.amp.GradScaler(enabled=AMP and DEVICE=="cuda")

    best_acc = 0.0
    t0 = time.time()

    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for xb, yb, _ in dl:
            # xb: [B, L, 1, 128, 88] ; yb: [B]
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # forward cu AMP
            with torch.cuda.amp.autocast(enabled=AMP and DEVICE=="cuda"):
                logits = model(xb)               # [B, num_classes]
                loss   = criterion(logits, yb)   # CE loss

            # backward + update
            scaler.scale(loss).backward()
            # gradient clipping pentru stabilitate
            if CLIP_NORM is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            # metrici simple pe train
            running_loss += loss.item() * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total   += yb.numel()

        avg_loss = running_loss / total
        acc      = correct / total
        print(f"Epoch {epoch:02d}/{EPOCHS}  |  Loss: {avg_loss:.4f}  Acc: {acc:.4f}")

        # 4) salvez ce e mai bun
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ↳ Saved best to {SAVE_PATH} (acc={best_acc:.4f})")

    dt = time.time() - t0
    print(f"[DONE] Train finished in {dt/60:.1f} min. Best train acc={best_acc:.4f}")

if __name__ == "__main__":
    train()
