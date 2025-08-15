# eval_rank1_nm.py — Rank-1 pe CASIA-B (NM-only): gallery nm-01..04, probe nm-05..06, subiecti 075–124

import os, glob, sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random

DATA_ROOT = "/home/daria/Documents/PRACTICA-CLEMENTIN/COD/data/CASIA-B-GEI"
MODEL_PATH = "/home/daria/Documents/PRACTICA-CLEMENTIN/COD/models/gei_cnn_nm.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_IDS = [f"{i:03d}" for i in range(75, 125)]
NM_GALLERY = [f"nm-{i:02d}" for i in range(1, 5)]
NM_PROBE = [f"nm-{i:02d}" for i in range(5, 7)]

def list_nm_pngs(root, subject_ids, nm_list):
    files = []
    for sid in subject_ids:
        for nm in nm_list:
            seq_dir = os.path.join(root, sid, nm)
            if not os.path.isdir(seq_dir):
                continue
            files.extend(sorted(glob.glob(os.path.join(seq_dir, "*.png"))))
    return files

def subject_from_path(path):
    # ../<root>/<SID>/<nm-xx>/<view>.png 
    parts = os.path.normpath(path).split(os.sep)
    return parts[-3] if len(parts) >= 4 else None

def angle_from_path(path):
    try:
        return int(Path(path).stem)   # '090' -> 90
    except:
        return -1


transform = T.Compose([
    T.Resize((128, 88)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])

class GEIPNGs(Dataset):
    def __init__(self, files):
        self.files = files
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]
        x = transform(Image.open(path).convert("L"))
        y = int(subject_from_path(path)) - 1 #labels 0..49

        return x, y, path


def load_module():
    try:
        from models.cnn_gei import CNN_GEI
        num_classes = 74
        model = CNN_GEI(num_classes).to(DEVICE)
    except Exeption as e:
        print(f"[EROARE] Nu am putut importa GEICNN din models/cnn_gei.py: {e}")
        sys.exit(1)
    
    sd = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model

#in aceasta functie nu e nevoie sa retinem gradientii, rualm doar forward
@torch.no_grad()
def embed_files(model, files, batch=128):
    dl = DataLoader(GEIPNGs(files), batch_size = batch, shuffle=False, num_workers=2, pin_memory=True)
    Z, Y, P = [], [], []

    def extractor(x):
        if hasattr(model, "features"):
            z = model.features(x).flatten(start_dim=1)
            return z
        return model.backbone(x).flatten(start_dim=1)
    for x, y, p in dl:
        x = x.to(DEVICE)
        
        if hasattr(model, "features"):
            z = model.features(x).flatten(start_dim=1)
        else:
            z = model(x)
            raise RuntimeError("Modelul nu are atributul 'features' sau 'backbone' pentru extragerea embedding-urilor.")
        z = F.normalize(z, dim=1) #normalizare 
        Z.append(z.cpu()); Y.extend(y.tolist()); P.extend(p)
        #pune batch-ul de embeddinguri pe CPU si il adauga la lista
        #baga toate etichetele din batch in lista Y
        #baga toate batchurile in lista P
    return torch.cat(Z,0), torch.tensor(Y), P #concatenam embeddingurile intr-o singura matrice, transform lista de etichete intr-un tensor, si returnez caile la png-uri\

#rank -1 testing

def rank1_test(probe_files, gallery_files, model):
    P, p_labels , p_paths = embed_files(model, probe_files)
    G, g_labels, g_paths = embed_files(model, gallery_files)

    sims = P @ G.t() #similaritati cosinus intre probe si galerie
    nn_idx = sims.argmax(1)
    preds = g_labels[nn_idx]
    acc = (preds == p_labels).float().mean().item()
    return acc, sims, nn_idx, (P, p_labels, p_paths), (G, g_labels, g_paths)

#-----------------ok aici sunt functiile de debug si vizualizare---------------
def show_random_matches(k, sims, nn_idx, probe_paths, p_labels, gallery_paths, g_labels):
    k = min(k, len(probe_paths))
    idxs = random.sample(range(len(probe_paths)), k)
    print("\n===Random top-1 matches===")
    for i in idxs:
        cos = sims[i, nn_idx[i]].item()
        print(f"[{i:4d}] TRUE={p_labels[i]} PRED={g_labels[nn_idx[i]]} cos={cos:.3f}")

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(Image.open(probe_paths[i]).convert("L"), cmap="gray")
        ax[0].set_title(f"Probe (ID={p_labels[i]})")
        ax[1].imshow(Image.open(gallery_paths[nn_idx[i]]).convert("L"), cmap='gray'); ax[1].set_title(f"Top-1 (ID={g_labels[nn_idx[i]]})")
        for a in ax: a.axis('off')
        plt.tight_layout(); plt.show()

def per_view_accuracy(sims, nn_idx, probe_paths, p_labels, g_labels):
    ok_by_angle, tot_by_angle = {}, {}
    for i in range(len(probe_paths)):
        a = angle_from_path(probe_paths[i])
        correct = int(p_labels[i]) == int(g_labels[nn_idx[i]])
        tot_by_angle[a] = tot_by_angle.get(a, 0) + 1
        if correct: ok_by_angle[a] = ok_by_angle.get(a, 0) + 1
    angles = sorted(tot_by_angle.keys())
    accs = [ok_by_angle.get(a,0)/tot_by_angle[a] for a in angles]
    print("\n=== Accuracy by view (probe) ===")
    for a,acc in zip(angles, accs):
        print(f"view {a:03d}: acc={acc:.3f}  (n={tot_by_angle[a]})")
    # bar chart
    plt.figure()
    plt.bar([str(a) for a in angles], accs)
    plt.xlabel("View angle"); plt.ylabel("Accuracy"); plt.title("Rank-1 by view (probe)")
    plt.ylim(0,1); plt.tight_layout(); plt.show()


def show_failures(m, sims, nn_idx, probe_paths, p_labels, gallery_paths, g_labels):
    # sortăm greșitele după scor (cele mai “încrezătoare” greșeli primele)
    wrong = [i for i in range(len(probe_paths)) if int(p_labels[i]) != int(g_labels[nn_idx[i]])]
    if not wrong:
        print("\nNicio greșeală în probe — nice! 🎉")
        return
    cos_scores = [sims[i, nn_idx[i]].item() for i in wrong]
    order = np.argsort(cos_scores)[::-1]  # descrescător
    wrong = [wrong[i] for i in order[:m]]
    print(f"\n=== Top {len(wrong)} failures (by highest wrong cosine) ===")
    for i in wrong:
        cos = sims[i, nn_idx[i]].item()
        print(f"[{i:4d}] TRUE={p_labels[i]}  PRED={g_labels[nn_idx[i]]}  view={angle_from_path(probe_paths[i])}  cos={cos:.3f}")
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(Image.open(probe_paths[i]).convert("L"), cmap='gray'); ax[0].set_title(f"Probe (ID={p_labels[i]})")
        ax[1].imshow(Image.open(gallery_paths[nn_idx[i]]).convert("L"), cmap='gray'); ax[1].set_title(f"Wrong Top-1 (ID={g_labels[nn_idx[i]]})")
        for a in ax: a.axis('off')
        plt.tight_layout(); plt.show()

def tsne_plot(P, p_labels, G, g_labels):
    try:
        from sklearn.manifold import TSNE
    except Exception:
        print("[NOTE] scikit-learn nu e instalat (pip install scikit-learn) — sar peste t-SNE.")
        return
    Z_all = torch.cat([G, P], 0).numpy()
    Y_all = torch.cat([g_labels, p_labels], 0).numpy()
    Z2 = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto").fit_transform(Z_all)
    plt.figure()
    plt.scatter(Z2[:,0], Z2[:,1], c=Y_all, s=6, cmap='tab20')
    plt.title("t-SNE of GEI embeddings (gallery+probe)")
    plt.xticks([]); plt.yticks([]); plt.tight_layout(); plt.show()


# def main():
#     gallery = list_nm_pngs(DATA_ROOT, TEST_IDS, NM_GALLERY)
#     probe = list_nm_pngs(DATA_ROOT, TEST_IDS, NM_PROBE)
#     print(f"Gallery: {len(gallery)} | Probe: {len(probe)}")

#     if not gallery or not probe:
#         print("[EROARE] Nu am găsit fișiere NM pentru test. Verifică structura: data/CASIA-B-GEI/<075..124>/nm-0*/<view>.png")
#         return
#     model = load_module()
#     acc = rank1_test(probe, gallery, model)
#     print(f"Rank-1 accuracy: {acc:.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-matches", type=int, default=0, help="arată N potriviri Top-1 random (cu imagini)")
    ap.add_argument("--per-view", action="store_true", help="afișează acuratețe pe view (bar chart)")
    ap.add_argument("--show-fails", type=int, default=0, help="arată N cazuri greșite (cu imagini)")
    ap.add_argument("--tsne", action="store_true", help="desenează t-SNE al embedding-urilor")
    args = ap.parse_args()

    gallery_files = list_nm_pngs(DATA_ROOT, TEST_IDS, NM_GALLERY)
    probe_files   = list_nm_pngs(DATA_ROOT, TEST_IDS, NM_PROBE)
    print(f"Gallery: {len(gallery_files)} | Probe: {len(probe_files)}")

    if not gallery_files or not probe_files:
        print("[EROARE] Nu am găsit fișiere NM pentru test. Aștept structura data/CASIA-B-GEI/<075..124>/nm-0*/<view>.png")
        sys.exit(1)

    model = load_module()
    acc, sims, nn_idx, (P, p_labels, p_paths), (G, g_labels, g_paths) = rank1_test(probe_files, gallery_files, model)
    print(f"Rank-1 (NM-only): {acc:.3f}")

    if args.show_matches > 0:
        show_random_matches(args.show_matches, sims, nn_idx, p_paths, p_labels, g_paths, g_labels)

    if args.per_view:
        per_view_accuracy(sims, nn_idx, p_paths, p_labels, g_labels)

    if args.show_fails > 0:
        show_failures(args.show_fails, sims, nn_idx, p_paths, p_labels, g_paths, g_labels)

    if args.tsne:
        tsne_plot(P, p_labels, G, g_labels)

if __name__ == "__main__":
    main()