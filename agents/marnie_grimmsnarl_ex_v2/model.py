"""MarnieValueNet: **solo value** (+ testa ausiliaria Phi).

    value (B,1)  -- valore dello stato per il lato codificato, in [-1,1]
    phi   (B,8)  -- le 8 componenti di Phi, testa ausiliaria

Niente policy head: la policy e' l'euristica (vedi encoding.py per il perche').

Dimensioni, e perche' sono queste. teo_2 ha 7.2M parametri e su CPU non riesce
a vedere abbastanza partite per allenarli; qui l'obiettivo e' un modello che si
alleni davvero nel budget disponibile, non uno piu' espressivo:

| voce                        | teo_2   | qui   |
|-----------------------------|---------|-------|
| embedding carte             | 325k    | ~8k   |
| embedding attacchi          | 399k    | 0     |
| encoder                     | 3.2M    | ~180k |
| decoder azioni + policy     | 2.4M    | 0     |
| teste value/phi             | ~600k   | ~90k  |
| **totale**                  | 7.2M    | ~290k |

La testa Phi resta perche' e' l'auxiliary task che paga di piu': costringe
l'encoder a costruire internamente prize-race, copertura energetica e stato
delle evoluzioni, invece di sperare che emergano dal solo esito della partita
(un bit per partita). Su un dataset piccolo e' la differenza tra un value head
che impara e uno che predice la media.

Il **vocabolario viaggia nel checkpoint**: `save()` scrive pesi + lista di card
ID, `load()` ricostruisce il modello con la dimensione giusta. Un checkpoint e
il suo vocabolario non possono disallinearsi.
"""

import numpy as np
import torch
import torch.nn as nn

from cards import Vocab
from encoding import N_PHI, N_TOKEN_TYPES, SLOT_FEAT_DIM

MODEL_CONFIG = dict(
    d_model=96,
    n_heads=4,
    n_layers=2,
    d_ff=256,
    dropout=0.1,
)


def _masked_mean(x, mask, dim, eps=1e-6):
    m = mask.unsqueeze(-1)
    return (x * m).sum(dim) / (m.sum(dim) + eps)


class MarnieValueNet(nn.Module):
    def __init__(self, vocab_size, d_model=96, n_heads=4, n_layers=2, d_ff=256,
                 dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Vista "memorizzazione": una riga per carta *del vocabolario ridotto*.
        # La riga 0 e' "carta fuori vocabolario" ed e' allenata come le altre:
        # e' un vero simbolo ("qualcosa che non conosco"), non un padding.
        self.card_emb = nn.Embedding(vocab_size, d_model)
        self.token_type_emb = nn.Embedding(N_TOKEN_TYPES, d_model)

        # Vista "generalizzazione": attributi statici + stato dinamico.
        self.slot_proj = nn.Sequential(
            nn.Linear(SLOT_FEAT_DIM, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.slot_ln = nn.LayerNorm(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.enc_ln = nn.LayerNorm(d_model)

        pooled = 2 * d_model      # mean + max
        self.value_head = nn.Sequential(
            nn.Linear(pooled, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.GELU(), nn.Linear(d_model, 1),
        )
        self.phi_head = nn.Sequential(
            nn.Linear(pooled, d_model), nn.GELU(), nn.Linear(d_model, N_PHI)
        )

    def forward(self, slot_feats, card_idx, card_mask, token_types):
        emb = self.card_emb(card_idx)                       # (B,T,K,d)
        cards = _masked_mean(emb, card_mask, dim=2)         # (B,T,d)
        x = self.slot_proj(slot_feats) + cards + self.token_type_emb(token_types)
        x = self.enc_ln(self.encoder(self.slot_ln(x)))
        pooled = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=-1)
        return torch.tanh(self.value_head(pooled)), torch.tanh(self.phi_head(pooled))


def build_model(vocab, **overrides):
    cfg = dict(MODEL_CONFIG)
    cfg.update(overrides)
    return MarnieValueNet(len(vocab), **cfg)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Checkpoint: pesi + vocabolario + config, sempre insieme
# ---------------------------------------------------------------------------

def save(path, model, vocab, extra=None):
    blob = {
        "model": model.state_dict(),
        "vocab": vocab.to_list(),
        "model_config": MODEL_CONFIG,
    }
    if extra:
        blob.update(extra)
    torch.save(blob, path)


def load(path, map_location="cpu"):
    """Ritorna (model, vocab, blob). Solleva se il file non e' un checkpoint v2."""
    blob = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(blob, dict) or "vocab" not in blob:
        raise ValueError(
            f"{path}: non e' un checkpoint di marnie_grimmsnarl_ex_v2 "
            "(manca il vocabolario). I checkpoint di teo_2 non sono compatibili."
        )
    vocab = Vocab(blob["vocab"])
    model = build_model(vocab, **blob.get("model_config", {}))
    model.load_state_dict(blob["model"])
    model.eval()
    return model, vocab, blob


# ---------------------------------------------------------------------------
# Adattatori numpy -> torch
# ---------------------------------------------------------------------------

def state_tensors(encs, device):
    slot = torch.from_numpy(np.stack([e.slot_feats for e in encs])).to(device)
    cidx = torch.from_numpy(np.stack([e.card_idx for e in encs])).to(device)
    cmask = torch.from_numpy(np.stack([e.card_mask for e in encs])).to(device)
    ttypes = torch.from_numpy(np.stack([e.token_types for e in encs])).to(device)
    return slot, cidx, cmask, ttypes


@torch.inference_mode()
def evaluate_batch(model, encs, device):
    """Valuta piu' stati in un colpo solo. Ritorna un array (B,) in [-1,1].

    Batch e non uno alla volta perche' la ricerca chiede sempre K stati
    insieme (uno per candidato): su CPU la differenza tra 4 forward da 1 e 1
    forward da 4 e' quasi un fattore 4.
    """
    if not encs:
        return np.zeros(0, dtype=np.float32)
    slot, cidx, cmask, ttypes = state_tensors(encs, device)
    value, _phi = model(slot, cidx, cmask, ttypes)
    return value.squeeze(-1).cpu().numpy()
