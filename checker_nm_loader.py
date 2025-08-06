import os
from torch.utils.data import DataLoader
import torch

# IMPORTĂ din loader-ul tău
from nm_loader import build_nm_protocol_files, TRAIN_IDS, TEST_IDS, NM_GALLERY, NM_PROBE
from Casia_B_NM_Train import Casia_B_NM_Train

DATA_ROOT = "data/CASIA-B-GEI"

def main():
    # ---------- 1) Verificăm TRAIN ----------
    try:
        ds = Casia_B_NM_Train(DATA_ROOT)
    except Exception as e:
        print("[EROARE] Nu am putut construi Casia_B_NM_Train:", e)
        print("Verifică dacă există PNG-uri în data/CASIA-B-GEI/<001..074>/nm-0*/<view>.png")
        return

    print(f"[TRAIN] număr mostre: {len(ds)} (subiecți {TRAIN_IDS[0]}..{TRAIN_IDS[-1]}, nm-01..nm-06, toate view-urile)")

    # un DataLoader mic doar ca test
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    x, y, p = next(iter(dl))  # un batch
    print(f"[TRAIN] batch shapes: x={tuple(x.shape)}, y={tuple(y.shape)}  (ar trebui x=[B,1,128,88])")
    print(f"[TRAIN] y min={int(y.min())}, y max={int(y.max())} (ar trebui în 0..73)")
    print(f"[TRAIN] exemplu path: {p[0]}")

    # ---------- 2) Verificăm TEST: gallery/probe (NM-only) ----------
    gallery, probe = build_nm_protocol_files(DATA_ROOT)
    print(f"[TEST] gallery files: {len(gallery)}  (nm-01..nm-04 pentru {TEST_IDS[0]}..{TEST_IDS[-1]})")
    print(f"[TEST] probe   files: {len(probe)}    (nm-05..nm-06 pentru {TEST_IDS[0]}..{TEST_IDS[-1]})")
    if gallery:
        print(f"[TEST] exemplu gallery: {gallery[0]}")
    if probe:
        print(f"[TEST] exemplu probe:   {probe[0]}")

    # ---------- 3) Sanity pe un eșantion din gallery ----------
    from PIL import Image
    import torchvision.transforms as T
    tfm = T.Compose([T.Resize((128,88)), T.ToTensor(), T.Normalize([0.5],[0.5])])

    if gallery:
        im = Image.open(gallery[0]).convert("L")
        tens = tfm(im)
        print(f"[TEST] tensor exemplu din gallery: shape={tuple(tens.shape)} (aștept [1,128,88])")

    print("\n[OK] Loader-ul pare în regulă. Poți trece la training.")

if __name__ == "__main__":
    main()
