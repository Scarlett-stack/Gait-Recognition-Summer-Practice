import os
import glob
import cv2
import numpy as np

INPUT_ROOT  = "/home/daria/Documents/PRACTICA-CLEMENTIN/COD/datasets/output" 
OUTPUT_ROOT = "data/CASIA-B-GEI" 
H, W = 128, 88                     # redimensionare standard 
THRESH = 1                         # prag pt. binarizare ma rog daca e 0 ramane 0 daca e 255 sau intre ramane 1

def is_sequence_dir(d):
    #un 'sequence dir' e un folder care contine imagini PNG."""
    if glob.glob(os.path.join(d, "*.png")):
        return True
    return False

def list_sequence_dirs(root):
    
    #gaseste recursiv directoarele care contin imagini (secvente).
    
    seq_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        # daca acest folder contine imagini, e o secventa
        if is_sequence_dir(dirpath):
            seq_dirs.append(dirpath)
    return sorted(seq_dirs)

def read_sequence_images(seq_dir):
    #citesc alea png, le sortez si le bag in lista
    files = glob.glob(os.path.join(seq_dir, "*.png"))
    files = sorted(files)
    return files

def compute_gei_for_sequence(seq_dir):
    """
    Cum calculez GEI pt o secventa gen pt un om?:
    - citesc cadrele
    - resize (88x128) -> vezi ca cv2.resize e (W,H)
    - binarizare (0/1)
    - media pe axa timp
    - intoarce imagine uint8 (0..255)
    """
    files = read_sequence_images(seq_dir)
    if len(files) == 0:
        return None

    acc = None #matrice/imagine in care adun toate secventele pixel cu pixel
    count = 0

    for fp in files:
        im = cv2.imread(fp, cv2.IMREAD_GRAYSCALE) #o facem greyscale sa iau silueta
        if im is None:
            continue
        # resize la (W,H)
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_NEAREST) #deci asta cu interplare era metoda cea mai simpla, gen ia cel mai apropiat pixel

        # binarizare (0/1)
        bin_im = (im > THRESH).astype(np.float32)
        
        # aici e a primul cadru in care nu am nici o imagine 
        if acc is None:
            acc = bin_im
        else:
            acc += bin_im
        count += 1

    if count == 0:
        return None

    gei = acc / float(count)           # media apartine de [0..1]
    #si transformam la loc in imagine gen [0..255]
    #clip e pentru siguranta gen daca e vreodata mai mare ca 255 ma asigur ca ramane in interval
    gei_u8 = (gei * 255.0).clip(0,255).astype(np.uint8)
    return gei_u8

def make_output_path(seq_dir):
    rel = os.path.relpath(seq_dir, INPUT_ROOT)
    head, tail = os.path.split(rel)
    # folderul de iesire:
    out_dir = os.path.join(OUTPUT_ROOT, head)
    os.makedirs(out_dir, exist_ok=True)
    # fisierul GEI:
    out_path = os.path.join(out_dir, f"{tail}.png")
    return out_path

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    seq_dirs = list_sequence_dirs(INPUT_ROOT)
    if not seq_dirs:
        print(f"[MI-E RAAU] NU AM SECEVNTE!!: {INPUT_ROOT}")
        return

    total = len(seq_dirs)
    ok = 0
    print(f"[INFO] AM GASIT {total} secvente!!!!")

    for i, seq_dir in enumerate(seq_dirs, 1):
        gei = compute_gei_for_sequence(seq_dir)
        if gei is None:
            print(f"[WARN] Secventa fara cadre valide????: {seq_dir}")
            continue
        out_path = make_output_path(seq_dir)
        cv2.imwrite(out_path, gei)
        ok += 1

        if i % 50 == 0 or i == total:
            print(f"  - saliuut: {i}/{total} (generate: {ok})")

    print(f"[DONE] AVEM GEI {ok}/{total} IN: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
