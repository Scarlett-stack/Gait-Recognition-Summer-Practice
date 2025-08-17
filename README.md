ca sa rulez ca modul si sa nu am probleme cu importurile:

```
python3 -m models.cnn_gei

```

o duce in 58-56% pentru 20 epoci si lr 1e-3 care orc e cam micut 

Ce urmeaza:

- imbunatatire gei: experimentat cu diferite loss functions , LR schedule (StepLR, CosineAnnealingLR)
- trb scris testarea

(ia-labs-env) daria@Daria-Katana-17-B13VFK:~/Documents/PRACTICA-CLEMENTIN/COD$ python3 rank1-eval.py --show-matches 6 --per-view --show-fails 6 --tsne
Gallery: 2187 | Probe: 1100
Rank-1 (NM-only): 0.965

===Random top-1 matches===
[ 228] TRUE=84 PRED=84 cos=0.998
[  51] TRUE=76 PRED=76 cos=0.997
[ 563] TRUE=99 PRED=99 cos=0.998
[ 501] TRUE=96 PRED=96 cos=0.998
[ 457] TRUE=94 PRED=94 cos=0.995
[ 285] TRUE=86 PRED=86 cos=0.998

=== Accuracy by view (probe) ===
view 000: acc=0.970  (n=100)
view 018: acc=0.990  (n=100)
view 036: acc=0.970  (n=100)
view 054: acc=0.960  (n=100)
view 072: acc=0.920  (n=100)
view 090: acc=0.950  (n=100)
view 108: acc=0.960  (n=100)
view 126: acc=0.960  (n=100)
view 144: acc=0.980  (n=100)
view 162: acc=1.000  (n=100)
view 180: acc=0.960  (n=100)

=== Top 6 failures (by highest wrong cosine) ===
[ 257] TRUE=85  PRED=76  view=72  cos=0.998
[ 404] TRUE=92  PRED=114  view=144  cos=0.998
[ 354] TRUE=90  PRED=77  view=36  cos=0.997
[1082] TRUE=123  PRED=104  view=72  cos=0.997
[ 324] TRUE=88  PRED=93  view=90  cos=0.997
[1007] TRUE=119  PRED=113  view=108  cos=0.997