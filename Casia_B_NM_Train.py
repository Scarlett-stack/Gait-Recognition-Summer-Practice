from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from nm_loader import build_nm_protocol_files, list_nm_view, TRAIN_IDS, NM_ALL
import os
"""
pytorch dataloader are nevoie de clasa Dataset, plus ca vreau sa fac shuffle
si le baga si etichete ce cute
"""
class Casia_B_NM_Train(Dataset):
    #antrenam pe GEI
    def __init__(self, root):
        self.root = root
        self.files = list_nm_view(root, TRAIN_IDS, NM_ALL) #lista cu toate fisierele train
        if not self.files:
            raise RuntimeError(f"Nu am gasit fisiere in {root}")
        self.transform = T.Compose([
            T.Resize((128,88)), #height si width
            T.ToTensor(),  #tensor pytorch [1, 128, 88], 1 e nr de canale = grayscale
            T.Normalize([0.5],[0.5]) # x_norm = (x - mean) / std -> [-1, 1]
        ])
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("L") #L pt grayscale
        x = self.transform(img) #aplic transformarea
        sid = os.path.basename(os.path.dirname(os.path.dirname(path)))
        #am uitat ca eu le-am aduagat 1 din greseala
        y = int(sid) - 1
        return x, y, path  #tensor , eticheta, path pt imagine 