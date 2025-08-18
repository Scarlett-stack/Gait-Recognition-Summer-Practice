#entru fiecare secvență (ex. .../001/nm-01/000/) 
# returnează un tensor de formă [L, 1, 128, 88] și eticheta (ID subiect).
import os, glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

#secventele trebuiesc sortate ca sa stim ordinea buna

transform = T.Compose(
    [
        T.Resize((128, 88)),
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ]
)

class SequenceDataset(Dataset):
    def __init__(self, root, subject_ids, seq_ids, L=20):
        #for now o sa luam 20 de cadre intr-o secventa putem modifica L daca e
        self.samples = []
        self.L = L
        self.root = root

        for sid in subject_ids: #['001', '002', ..]
            for seq in seq_ids: #['nm-01', 'nm-02', ..]
                seq_dir = os.path.join(root, sid, seq)
                if not os.path.isdir(seq_dir):
                    continue
                #acum iau viewurile
                for view_dir in sorted(os.listdir(seq_dir)):
                    full_dir = os.path.join(seq_dir, view_dir)
                    if not os.path.isdir(full_dir):
                        continue
                    files = sorted(glob.glob(os.path.join(full_dir, "*.png")))
                    if len(files) == 0:
                        continue
                    self.samples.append((sid, files)) # (subject_id, [frame_paths])
        if not self.samples:
            raise RuntimeError(f"Nu am gasit secvente in {root} pentru subiectii {subject_ids} si secventele {seq_ids}")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sid, files = self.samples[idx]
        total = len(files)
        if total >= self.L:
            idxs = np.linspace(0, total-1, self.L, dtype=int).round().astype(int)
        else:
            idxs = list(range(total)) + [total-1]*(self.L - total)

        frames = []
        picked_paths = []
        for j in idxs:
            p = files[j]
            picked_paths.append(p)
            img = Image.open(files[j]).convert("L")
            x = transform(img)
            frames.append(x)
        x = torch.stack(frames, dim=0)  # [L, 1, 128, 88]
        y = int(sid) - 1
        return x, y, picked_paths[:5]