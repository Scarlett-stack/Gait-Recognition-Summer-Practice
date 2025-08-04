import os, glob

#construim tiparele pt fisiere
#pt train 70% si restul test
TRAIN_IDS = [f"{i:03d}" for i in range(1, 75)]   # 001 - 074
TEST_IDS = [f"{i:03d}" for i in range(75, 125)]   # 075 - 124
#deocamdata vom face cu secvenetele de normal , fara bg si cl
NM_ALL = [f"nm-0{i}" for i in range(1,7)]
NM_GALLERY = [f"nm-0{i}" for i in range(1, 5)]
NM_PROBE = [f"nm-0{i}" for i in range(5, 7)]
VIEWS = [f"{v:03d}.png" for v in range(0, 181, 18)]

#construim pt fiecare view cate un path ca sa stiu unde il pun
"""
root = folderul radacine unde am GEI 
subject_ids = [001, 002 bla bla]
nm_list = ce secvente aleg 
"""
def list_nm_view(root, subject_ids, nm_list):
    paths = []
    for subject in subject_ids:
        for nm in nm_list:
            dir_secv = os.path.join(root, subject, nm) #verific daca exista, slabe sanse sa nu
            if not os.path.exists(dir_secv):
                continue
            for view in VIEWS:
                p = os.path.join(root, subject, nm, view)
                if os.path.exists(p):
                    paths.append(p)
                else:
                    print(f"Warning: nu exista {p}")
    return paths

"""
fisierele de test : 075..124
->gallery : nm-01 .. nm-04
->probe : nm-05 .. nm-06
si retunez (gallery_files, probe_files) ca liste de cai catre acele png-uri
"""
def build_nm_protocol_files(root):
    gallery = list_nm_view(root, TEST_IDS, NM_GALLERY)
    probe = list_nm_view(root, TEST_IDS, NM_PROBE)
    return (gallery, probe)