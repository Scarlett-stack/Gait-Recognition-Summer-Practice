# eval_rank1_nm.py — Rank-1 pe CASIA-B (NM-only): gallery nm-01..04, probe nm-05..06, subiecti 075–124

import os, glob, sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

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
    P, p_labels , _ = embed_files(model, probe_files)
    G, g_labels, _ = embed_files(model, gallery_files)

    sims = P @ G.t() #similaritati cosinus intre probe si galerie
    nn_idx = sims.argmax(1)
    preds = g_labels[nn_idx]
    acc = (preds == p_labels).float().mean().item()
    return acc

def main():
    gallery = list_nm_pngs(DATA_ROOT, TEST_IDS, NM_GALLERY)
    probe = list_nm_pngs(DATA_ROOT, TEST_IDS, NM_PROBE)
    print(f"Gallery: {len(gallery)} | Probe: {len(probe)}")

    if not gallery or not probe:
        print("[EROARE] Nu am găsit fișiere NM pentru test. Verifică structura: data/CASIA-B-GEI/<075..124>/nm-0*/<view>.png")
        return
    model = load_module()
    acc = rank1_test(probe, gallery, model)
    print(f"Rank-1 accuracy: {acc:.3f}")

if __name__ == "__main__":
    main()