"""TeoNet: rete value + policy + testa ausiliaria Phi.

Tre uscite:

  value  (B,1)  -- valore dello stato per il giocatore di turno, in [-1,1]
  phi    (B,8)  -- le 8 componenti di Phi (testa *ausiliaria*)
  policy (B,A)  -- logit per ogni azione candidata

La testa Phi non serve a giocare: serve ad allenare. Predire componenti che
sappiamo gia' calcolare esattamente sembra inutile, ma e' un classico
auxiliary task: costringe l'encoder a costruire internamente le
rappresentazioni di prize-race, energia ed evoluzioni invece di sperare che
emergano dal solo segnale di vittoria (che arriva una volta a partita ed e'
estremamente sparso). E' il modo piu' economico per accelerare il value head.

Differenze architetturali rispetto a teo_1 (d_model=128, 1 layer encoder + 1
decoder, 2 teste di attenzione, ~1M parametri):

  - doppia vista sulle carte: embedding per ID *piu'* proiezione degli
    attributi statici, cosi' generalizza a carte mai viste (vedi cards.py);
  - encoder pre-LN piu' profondo (4 layer, 8 teste, d_model 256);
  - le azioni si attenzionano *tra loro* prima di essere valutate: la scelta
    di una mossa e' intrinsecamente comparativa, e una policy che vede le
    alternative sceglie molto meglio di una che le valuta in isolamento;
  - pooling mean+max invece del solo mean.
"""

import numpy as np
import torch
import torch.nn as nn

from cards import MAX_ATTACK_ID, MAX_CARD_ID
from encoding import (
    ACTION_FEAT_DIM,
    N_TOKEN_TYPES,
    SLOT_FEAT_DIM,
)
from reward import COMPONENT_ORDER

N_PHI_COMPONENTS = len(COMPONENT_ORDER)

MODEL_CONFIG = dict(
    d_model=256,
    n_heads=8,
    n_enc_layers=4,
    n_dec_layers=2,
    d_ff=1024,
    dropout=0.1,
)


def _masked_mean(x, mask, dim, eps=1e-6):
    """Media di `x` lungo `dim` pesata da `mask` (broadcast sull'ultima dim)."""
    m = mask.unsqueeze(-1)
    return (x * m).sum(dim) / (m.sum(dim) + eps)


class CrossBlock(nn.Module):
    """Un blocco del decoder: self-attention tra azioni + cross-attention sullo
    stato + feed-forward. Pre-LN, residuale."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, memory, action_pad_mask):
        # action_pad_mask: True dove l'azione e' padding (da ignorare).
        h = self.ln_self(x)
        a, _ = self.self_attn(
            h, h, h, key_padding_mask=action_pad_mask, need_weights=False
        )
        # Le righe di padding possono uscire NaN se l'intera riga e'
        # mascherata; le azzeriamo esplicitamente.
        x = x + torch.nan_to_num(a)

        kv = self.ln_kv(memory)
        c, _ = self.cross_attn(self.ln_q(x), kv, kv, need_weights=False)
        x = x + torch.nan_to_num(c)

        return x + self.ff(self.ln_ff(x))


class TeoNet(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_enc_layers=4, n_dec_layers=2,
                 d_ff=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # --- vista "memorizzazione": embedding per card / attack ID ---------
        self.card_emb = nn.Embedding(MAX_CARD_ID, d_model, padding_idx=0)
        self.attack_emb = nn.Embedding(MAX_ATTACK_ID, d_model, padding_idx=0)
        self.token_type_emb = nn.Embedding(N_TOKEN_TYPES, d_model)

        # --- vista "generalizzazione": attributi statici ---------------------
        self.slot_proj = nn.Sequential(
            nn.Linear(SLOT_FEAT_DIM, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.slot_ln = nn.LayerNorm(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, n_enc_layers, enable_nested_tensor=False
        )
        self.enc_ln = nn.LayerNorm(d_model)

        pooled_dim = 2 * d_model  # mean + max
        self.value_head = nn.Sequential(
            nn.Linear(pooled_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.phi_head = nn.Sequential(
            nn.Linear(pooled_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, N_PHI_COMPONENTS),
        )

        self.action_proj = nn.Sequential(
            nn.Linear(ACTION_FEAT_DIM, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.action_ln = nn.LayerNorm(d_model)
        self.decoder = nn.ModuleList(
            [CrossBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_dec_layers)]
        )
        self.policy_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    # -----------------------------------------------------------------------

    def encode_state(self, slot_feats, card_ids, card_mask, token_types):
        # Somma degli embedding delle carte presenti nel token, mediata.
        emb = self.card_emb(card_ids)                       # (B,T,K,d)
        pooled_cards = _masked_mean(emb, card_mask, dim=2)  # (B,T,d)

        x = self.slot_proj(slot_feats) + pooled_cards + self.token_type_emb(token_types)
        x = self.slot_ln(x)
        return self.enc_ln(self.encoder(x))

    def forward(self, slot_feats, card_ids, card_mask, token_types,
                action_feats, action_card_ids, action_card_mask,
                action_attack_ids, action_mask):
        memory = self.encode_state(slot_feats, card_ids, card_mask, token_types)

        pooled = torch.cat([memory.mean(dim=1), memory.max(dim=1).values], dim=-1)
        value = torch.tanh(self.value_head(pooled))
        phi = torch.tanh(self.phi_head(pooled))

        a_emb = self.card_emb(action_card_ids)                        # (B,A,2,d)
        a_cards = _masked_mean(a_emb, action_card_mask, dim=2)        # (B,A,d)
        a = self.action_proj(action_feats) + a_cards + self.attack_emb(action_attack_ids)
        a = self.action_ln(a)

        pad = action_mask < 0.5
        # Se una riga del batch non ha nessuna azione valida, MultiheadAttention
        # produce NaN: teniamo almeno la prima posizione visibile.
        all_pad = pad.all(dim=1)
        if all_pad.any():
            pad = pad.clone()
            pad[all_pad, 0] = False

        for block in self.decoder:
            a = block(a, memory, pad)

        logits = self.policy_head(a).squeeze(-1)                      # (B,A)
        logits = logits.masked_fill(action_mask < 0.5, float("-inf"))
        return value, phi, logits


def build_model():
    return TeoNet(**MODEL_CONFIG)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Adattatori numpy -> torch
# ---------------------------------------------------------------------------

def state_tensors(state_encs, device):
    """Impila una lista di StateEncoding in tensori batch."""
    slot = torch.from_numpy(np.stack([s.slot_feats for s in state_encs])).to(device)
    cids = torch.from_numpy(np.stack([s.card_ids for s in state_encs])).to(device)
    cmask = torch.from_numpy(np.stack([s.card_mask for s in state_encs])).to(device)
    ttypes = torch.from_numpy(
        np.stack([s.token_types for s in state_encs])
    ).to(device)
    return slot, cids, cmask, ttypes


def action_tensors(action_encs, device, max_actions=None):
    """Impila una lista di ActionEncoding, con padding a lunghezza comune."""
    if max_actions is None:
        max_actions = max(len(a.actions) for a in action_encs)
    max_actions = max(1, max_actions)
    b = len(action_encs)

    feats = np.zeros((b, max_actions, ACTION_FEAT_DIM), dtype=np.float32)
    cids = np.zeros((b, max_actions, 2), dtype=np.int64)
    cmask = np.zeros((b, max_actions, 2), dtype=np.float32)
    aids = np.zeros((b, max_actions), dtype=np.int64)
    amask = np.zeros((b, max_actions), dtype=np.float32)

    for i, enc in enumerate(action_encs):
        k = min(len(enc.actions), max_actions)
        if k == 0:
            continue
        feats[i, :k] = enc.feats[:k]
        cids[i, :k] = enc.card_ids[:k]
        cmask[i, :k] = enc.card_mask[:k]
        aids[i, :k] = enc.attack_ids[:k]
        amask[i, :k] = 1.0

    return (
        torch.from_numpy(feats).to(device),
        torch.from_numpy(cids).to(device),
        torch.from_numpy(cmask).to(device),
        torch.from_numpy(aids).to(device),
        torch.from_numpy(amask).to(device),
    )


@torch.inference_mode()
def evaluate(model, state_enc, action_enc, device):
    """Inferenza su un singolo stato. Ritorna (value, phi, priors).

    `priors` e' la softmax dei logit sulle sole azioni legali: e' la
    distribuzione che la MCTS usa come prior nel PUCT.
    """
    slot, cids, cmask, ttypes = state_tensors([state_enc], device)
    af, acid, acmask, aaid, amask = action_tensors([action_enc], device)
    value, phi, logits = model(
        slot, cids, cmask, ttypes, af, acid, acmask, aaid, amask
    )
    n = len(action_enc.actions)
    priors = torch.softmax(logits[0, :n], dim=-1).cpu().numpy()
    return float(value[0, 0]), phi[0].cpu().numpy(), priors
