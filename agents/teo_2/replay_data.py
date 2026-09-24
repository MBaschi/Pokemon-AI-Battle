"""Formato dei sample estratti dai replay, condiviso fra builder e pretrain.

Deve stare in un modulo suo e non nello script che lo crea: pickle registra la
classe col suo modulo di definizione, quindi una classe definita in `__main__`
di `build_replay_dataset.py` non e' ricaricabile da `pretrain_from_replays.py`
(il cui `__main__` e' un altro file).
"""

import numpy as np


class ReplaySample:
    """Un (stato, mossa dell'esperto, esito) pronto per il training.

    Stesso contratto di `train_selfplay.Sample`, cosi' il pretrain puo' riusare
    la stessa pipeline di batching.

    Gli array sono in float16: a ~14 KB per sample in float32 un dataset da
    500k occuperebbe 7 GB. La precisione non serve, sono feature normalizzate
    in [0,1]; la conversione a float32 avviene al caricamento del batch.
    """

    __slots__ = ("slot_feats", "card_ids", "card_mask",
                 "a_feats", "a_card_ids", "a_card_mask", "a_attack_ids",
                 "policy", "value", "phi_target")

    def __init__(self, st, ae, policy, value, phi_target):
        self.slot_feats = st.slot_feats.astype(np.float16)
        self.card_ids = st.card_ids.astype(np.int32)
        self.card_mask = st.card_mask.astype(np.float16)
        self.a_feats = ae.feats.astype(np.float16)
        self.a_card_ids = ae.card_ids.astype(np.int32)
        self.a_card_mask = ae.card_mask.astype(np.float16)
        self.a_attack_ids = ae.attack_ids.astype(np.int32)
        self.policy = policy.astype(np.float16)
        self.value = float(value)
        self.phi_target = phi_target.astype(np.float16)
