# model secvential pt gait cu cnn (pe fiecare frame) +  gru (pe timp)
#in: [B, L, 1, 128, 88] tensor
#out : logit: [B , num_classes]

#cnn extrage features din fiecare cadru , gru invata ritm/ordine pasi in timp

import torch
import torch.nn as nn   
import torch.nn.functional as F

class CNN_GRU(nn.Module):
    def __init__(
            self, 
            num_classes:int, 
            feat_dim: int = 128, #dim vectorului frame extras de cnn
            gru_hidden:int = 128, #dim starii ascunse pt gru
            num_layers:int = 1,   #nr de straturi gru 
            bidirectional: bool = False,  #for now nu il facem true
            proj_to_feat: bool = True, #proiectam iesirea gru inapoi in spatiul de caracteristici
            dropout: float = 0.0 #dropout dupa embedding (anti-overfit)
    ):
        super().__init__()
        # ENCODER DE FRAME (CNN)
        # Refolosim ideea din CNN_GEI: 3 blocuri conv + BN + ReLU (+ MaxPool),
        # apoi AdaptiveAvgPool2d(1) ca să obt 1 singur vector per frame.
        # Outputul va fi [B*L, 128, 1, 1] -> flatten la [B*L, 128]

        self.frame_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), #64x44
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), #32x22
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), #16x11
            nn.AdaptiveAvgPool2d(1) #-> [B*L, 128, 1, 1]
        )

        #GRU pe timp
        # PrimeSte secvenTa de vectori de frame (dim = feat_dim) și da stari ascunse
        # batch_first=True -> asteapta [B, L, D] în loc de [L, B, D]
        self.gru = nn.GRU(
            input_size=feat_dim, 
            hidden_size=gru_hidden, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=bidirectional
        )

        out_dim = gru_hidden * (2 if bidirectional else 1)

        #proiectie la dimensiunea embedding vrem 126-dim
        if proj_to_feat:
            self.proj = nn.Linear(out_dim, feat_dim)
            self.embed_dim = feat_dim
        else:
            self.proj = nn.Identity()
            self.embed_dim = out_dim
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        #head de clasificare pt train
        self.head = nn.Linear(self.embed_dim, num_classes)


    def _encode_frames(self, x:torch.Tensor) -> torch.Tensor:
            B, L, C, H, W = x.shape
            x = x.view(B*L, C, H, W)  #combina B si L
            x = self.frame_encoder(x) #[B*L, feat_dim, 1, 1]
            x = x.view(B, L, -1)      #[B, L, feat_dim]
            return x

    #features/embedding de secventa fara clasificator
    #return un vector pe secventa (B, embed_dim) pt rank-1 eval

    def forward_features(self, x_seq: torch.Tensor) -> torch.Tensor:
            #x_seq: [B, L, 1, 128, 88]
            #cnn pt fiecare frame
            f_seq = self._encode_frames(x_seq) #[B, L, feat_dim]
            
            # 2) GRU -> h_n (forma: [num_layers * num_directions, B, hidden])
            # Nu avem nevoie de output-toti-timpii; iau DOAR ultima stare
            # De ce? h_T (= "rezumatul" secvenței) e un descriptor bun al mersului
            _, h_n = self.gru(f_seq)

            #3 extrag ultima stare a ultimului start 
            # - unidirecțional: h_n[-1] ∈ [B, H]
            # - bidirecțional: h_n are 2 direcții pe strat: [..., fwd, bwd]
            if self.gru.bidirectional:
                num_dirs = 2
                h_n = h_n.view(self.gru.num_layers, num_dirs, x_seq.size(0), self.gru.hidden_size)
                # ultimul strat, concat forward și backward
                h_last = torch.cat([h_n[-1, 0], h_n[-1, 1]], dim=1)   # [B, 2H]
            else:
                h_last = h_n[-1]  # [B, H]

            # 4) Proiecție la dim. embedding (de obicei 128) + (optional) dropout
            emb = self.proj(h_last)          # [B, embed_dim]
            emb = self.dropout(emb)
            return emb

        # ------------- forward de antrenare (cu clasificare) -------------
    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        emb = self.forward_features(x_seq)   # [B, embed_dim]
        logits = self.head(emb)              # [B, num_classes]
        return logits  


#sanity check
# def main():
#     B, L = 2, 20
#     x = torch.randn(B, L, 1, 128, 88)  # batch de 2 secvențe, câte 20 cadre

#     model = CNN_GRU(num_classes=74)    # 74 subiecți în train
#     with torch.no_grad():
#         emb = model.forward_features(x)
#         logits = model(x)

#     print("emb shape:", emb.shape)      # aștept [2, 128]
#     print("logits shape:", logits.shape)  # aștept [2, 74]

# if __name__ == "__main__":
#     main()