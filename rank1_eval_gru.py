# rank1_eval_seq.py — Rank-1 pe CASIA-B (secvențe), CNN+GRU
# Galerie: nm-01..nm-04 (subiecți 075..124)
# Probe:   nm-05..nm-06  SAU  bg-01..02  SAU  cl-01..02  ( cu --probe)
#
# Intrare: datasets/output/<SID>/<seq_kind>/<view>/*.png
# Model:   models/cnn_gru_nm.pth  (antrenat pe 001..074, nm-01..06)

import os, sys, glob, argparse, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT_FRAMES = "datasets/output"  # <-- SECVENTELE (NU GEI!)
MODEL_PATH  = "models/cnn_gru_nm.pth"

TEST_IDS    = [f"{i:03d}" for i in range(75, 125)]
NM_GALLERY  = [f"nm-{i:02d}" for i in range(1, 5)]  # nm-01..04
NM_PROBE    = [f"nm-{i:02d}" for i in range(5, 7)]  # nm-05..06
BG_PROBE    = [f"bg-{i:02d}" for i in range(1, 3)]  # bg-01..02
CL_PROBE    = [f"cl-{i:02d}" for i in range(1, 3)]  # cl-01..02


def subject_from_seqdir(seq_dir: str):
    # .../<root>/<SID>/<seq_kind>/<view_dir>
    parts = Path(seq_dir).parts
    # [..., ROOT_FRAMES, SID, seq_kind, view] -> SID = -3
    return parts[-3] if len(parts) >= 3 else None

def angle_from_seqdir(seq_dir: str):
    # view_dir e gen '000', '018'
    try:
        return int(Path(seq_dir).name)
    except:
        return -1

def list_seq_dirs(root, subject_ids, seq_kinds):
    seq_dirs = []
    for sid in subject_ids:
        for sk in seq_kinds:
            base = os.path.join(root, sid, sk)
            if not os.path.isdir(base):
                continue
            for view_dir in sorted(os.listdir(base)):
                full = os.path.join(base, view_dir)
                if os.path.isdir(full):
                    # păstrăm doar dacă are png-uri
                    if glob.glob(os.path.join(full, "*.png")):
                        seq_dirs.append(full)
    return seq_dirs

_transform = T.Compose([
    T.Resize((128, 88)),
    T.ToTensor(),
    T.Normalize([0.5],[0.5])
])

class EvalSeqDataset(Dataset):
    def __init__(self, seq_dirs, L=20):
        self.seq_dirs = seq_dirs
        self.L = L

    def __len__(self):
        return len(self.seq_dirs)

    def __getitem__(self, idx):
        seq_dir = self.seq_dirs[idx]
        sid = int(subject_from_seqdir(seq_dir)) - 1  # 0..49
        files = sorted(glob.glob(os.path.join(seq_dir, "*.png")))
        Ttot = len(files)
        if Ttot >= self.L:
            idxs = np.linspace(0, Ttot-1, num=self.L).round().astype(int)
        else:
            idxs = list(range(Ttot)) + [Ttot-1] * (self.L - Ttot)

        frames = []
        for j in idxs:
            img = Image.open(files[j]).convert("L")
            frames.append(_transform(img))
        x = torch.stack(frames, dim=0)  # [L,1,128,88]
        return x, sid, seq_dir

def load_model():
    from models.cnn_gru import CNN_GRU
    model = CNN_GRU(num_classes=74).to(DEVICE)  # 74 subs în train
    sd = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model

@torch.no_grad()
def embed_sequences(model, seq_dirs, L=20, batch=8):
    ds = EvalSeqDataset(seq_dirs, L=L)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)

    Z, Y, P = [], [], []
    for xb, yb, pb in dl:
        xb = xb.to(DEVICE)                    # [B,L,1,128,88]
        z  = model.forward_features(xb)       # [B,128]  — embedding de secv
        z  = F.normalize(z, dim=1)            # L2-norm pt. cosine
        Z.append(z.cpu()); Y.extend(yb.tolist()); P.extend(pb)
    return torch.cat(Z, 0), torch.tensor(Y), P

# --- Rank-1: cosine NN ---
def rank1(probe_dirs, gallery_dirs, model, L=20):
    P, p_labels, p_paths = embed_sequences(model, probe_dirs, L=L)
    G, g_labels, g_paths = embed_sequences(model, gallery_dirs, L=L)

    sims  = P @ G.t()                # cosine (deoarece am normalizat)
    nn_ix = sims.argmax(1)           # top-1 index din galerie pt. fiecare probe
    preds = g_labels[nn_ix]
    acc   = (preds == p_labels).float().mean().item()
    return acc, sims, nn_ix, (P, p_labels, p_paths), (G, g_labels, g_paths)


def show_random_matches(k, sims, nn_idx, probe_dirs, p_labels, gallery_dirs, g_labels):
    k = min(k, len(probe_dirs))
    idxs = random.sample(range(len(probe_dirs)), k)
    print("\n=== Random top-1 matches ===")
    for i in idxs:
        cos = sims[i, nn_idx[i]].item()
        print(f"[{i:4d}] TRUE={p_labels[i]}  PRED={g_labels[nn_idx[i]]}  cos={cos:.3f}  view={angle_from_seqdir(probe_dirs[i])}")

        probe_first  = sorted(glob.glob(os.path.join(probe_dirs[i], "*.png")))[0]
        gallery_first= sorted(glob.glob(os.path.join(gallery_dirs[nn_idx[i]], "*.png")))[0]
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(Image.open(probe_first).convert("L"), cmap="gray"); ax[0].set_title(f"Probe (ID={p_labels[i]})")
        ax[1].imshow(Image.open(gallery_first).convert("L"), cmap="gray"); ax[1].set_title(f"Top-1 (ID={g_labels[nn_idx[i]]})")
        for a in ax: a.axis("off")
        plt.tight_layout(); plt.show()

def per_view_accuracy(sims, nn_idx, probe_dirs, p_labels, g_labels):
    ok_by, tot_by = {}, {}
    for i in range(len(probe_dirs)):
        a = angle_from_seqdir(probe_dirs[i])
        correct = int(p_labels[i]) == int(g_labels[nn_idx[i]])
        tot_by[a] = tot_by.get(a, 0) + 1
        if correct: ok_by[a] = ok_by.get(a, 0) + 1
    angles = sorted(tot_by.keys())
    accs   = [ok_by.get(a,0)/tot_by[a] for a in angles]
    print("\n=== Accuracy by view (probe) ===")
    for a, acc in zip(angles, accs):
        print(f"view {a:03d}: acc={acc:.3f} (n={tot_by[a]})")
    plt.figure()
    plt.bar([str(a) for a in angles], accs)
    plt.xlabel("View angle"); plt.ylabel("Accuracy"); plt.title("Rank-1 by view (probe, seq)")
    plt.ylim(0,1); plt.tight_layout(); plt.show()


def tsne_plot(P, p_labels, G, g_labels, out_path):
    try:
        from sklearn.manifold import TSNE
    except Exception:
        print("[NOTE] scikit-learn nu e instalat (pip install scikit-learn) — sar peste t-SNE.")
        return
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    Z_all = torch.cat([G, P], 0).numpy()
    Y_all = torch.cat([g_labels, p_labels], 0).numpy()

    Z2 = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto").fit_transform(Z_all)

    plt.figure()
    plt.scatter(Z2[:,0], Z2[:,1], c=Y_all, s=6, cmap='tab20')
    plt.title("t-SNE of sequence embeddings (gallery + probe)")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"[OK] t-SNE salvat la: {out_path}")

# --- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", choices=["nm","bg","cl"], default="nm",
                    help="ce tip de probe: nm (default), bg sau cl")
    ap.add_argument("--L", type=int, default=20, help="nr cadre per secv (subsampling/pad)")
    ap.add_argument("--show-matches", type=int, default=0, help="arata N potriviri Top-1 random (imagini)")
    ap.add_argument("--per-view", action="store_true", help="afis acuratete pe view (bar chart)")
    ap.add_argument("--tsne", action="store_true", help="desenează t-SNE al embedding-urilor și salvează PNG")

    args = ap.parse_args()

    probe_list = NM_PROBE if args.probe=="nm" else (BG_PROBE if args.probe=="bg" else CL_PROBE)

    gallery_dirs = list_seq_dirs(ROOT_FRAMES, TEST_IDS, NM_GALLERY)
    probe_dirs   = list_seq_dirs(ROOT_FRAMES, TEST_IDS, probe_list)

    print(f"Gallery seqs: {len(gallery_dirs)} | Probe({args.probe}) seqs: {len(probe_dirs)}")
    if not gallery_dirs or not probe_dirs:
        print("[EROARE] Nu am gasit secv trb folders: datasets/output/<075..124>/<nm-xx|bg-xx|cl-xx>/<view>/*.png")
        sys.exit(1)

    model = load_model()
    acc, sims, nn_idx, (P, p_labels, p_paths), (G, g_labels, g_paths) = rank1(probe_dirs, gallery_dirs, model, L=args.L)
    print(f"Rank-1 ({args.probe}) = {acc:.3f}")


    if args.tsne:
        tsne_out = f"reports/tsne_seq_{args.probe}.png"
        tsne_plot(P, p_labels, G, g_labels, tsne_out)

    if args.show_matches > 0:
        show_random_matches(args.show_matches, sims, nn_idx, p_paths, p_labels, g_paths, g_labels)

    if args.per_view:
        per_view_accuracy(sims, nn_idx, p_paths, p_labels, g_labels)

if __name__ == "__main__":
    main()
