from torch.utils.data import DataLoader
import torch, os
from datasets.seq_loader import SequenceDataset as SeqDataset
try:
    from nm_loader import TRAIN_IDS
except Exception:
    TRAIN_IDS = [f"{i:03d}" for i in range(1,75)]

ROOT = "datasets/output"
SEQ_KINDS = [f"nm-0{i}" for i in range(1,7)]
L = 20

def main():
    ds = SeqDataset(ROOT, TRAIN_IDS, SEQ_KINDS, L=L)
    print(f"[INFO] Nr secvente gasite: {len(ds)}")

    x, y, debug_paths = ds[0]
    print(f"[INFO] Un sample: x.shape={tuple(x.shape)} (astept [L,1,128,88]), y={y}")
    print("[INFO] Primele 5 cadre alese (trebuie sa fie in ordine crescatoare):")
    for p in debug_paths:
        print("   ", os.path.basename(p))

    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)
    xb, yb, dbg = next(iter(dl))

    # ——— FIX: dacă pentru vreun motiv yb e listă, îl convertim la tensor
    if isinstance(yb, list):
        yb = torch.tensor(yb)

    print(f"[INFO] Batch: xb.shape={tuple(xb.shape)} (astept [B,L,1,128,88]), yb.shape={tuple(yb.shape)})")
    # arată și 2 nume de cadre din primul sample din batch, ca să vezi ordinea
    print("[INFO] Debug batch[0], primele 2 cadre:", [os.path.basename(p) for p in dbg[0][:2]])

if __name__ == "__main__":
    main()
