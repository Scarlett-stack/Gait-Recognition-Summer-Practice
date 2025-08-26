# eval_rank1_nm.py — Rank-1 pe CASIA-B (NM-only): gallery nm-01..04, probe nm-05..06, subiecti 075–124

import os, glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

DATA_ROOT = "data/CASIA-B-GEI"
MODEL_PATH = "models/gei_cnn_nm.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_IDS = [f"nm-{i:02d}" for i in range(75, 125)]
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
        x = tfm(Image.open(path).convert("L"))
        y = int(subject_from_path(path)) - 1 #labels 0..49

        return x, y, path

