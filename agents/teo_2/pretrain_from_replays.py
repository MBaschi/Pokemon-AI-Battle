"""Pre-allena TeoNet in supervisionato sui replay reali (behavioral cloning).

E' il passo di AlphaGo, non di AlphaGo Zero: policy supervisionata da partite
di giocatori forti, poi RL a partire da li'. AlphaGo Zero ha mostrato che il
bootstrap supervisionato non e' *necessario*, ma serviva compute enorme. Con
poche ore di CPU e' invece la leva piu' efficace, e i numeri di questo progetto
lo confermano: 24 h di RL puro danno 12% contro `gio_v1`, mentre un'euristica
scritta in un'ora ne fa 20%.

Cosa impara, e cosa no:

  - la **policy** e' il collo di bottiglia di teo_2 (KL ferma a 0.23 per 120
    iterazioni), ed e' esattamente cio' che i replay insegnano meglio: il
    target e' one-hot sulla mossa dell'esperto, molto piu' netto della
    distribuzione di visite di una MCTS a poche simulazioni;
  - il **value** impara da esiti veri invece che da self-play fra due copie
    deboli;
  - il tetto e' la forza dei maestri. Se sono euristici, il clone arriva al
    livello euristico: e' il fine-tuning RL successivo che deve superarlo.

Dopo questo script:
    python train_selfplay.py --resume out/pretrained.pth [...]

Esempi:
    python pretrain_from_replays.py --data data/replays --epochs 3
"""

import argparse
import glob
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import cgpath  # noqa: F401

from encoding import TOKEN_TYPES
from model import build_model, count_parameters
from replay_data import ReplaySample  # noqa: F401  -- serve a pickle per il load


def load_shard(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def make_batch(samples, device):
    """Impila i sample in tensori. Gli array su disco sono float16 (per non
    occupare 7 GB); la rete lavora in float32, quindi si converte qui."""
    b = len(samples)
    max_a = max(len(s.policy) for s in samples)

    slot = np.stack([s.slot_feats for s in samples]).astype(np.float32)
    cids = np.stack([s.card_ids for s in samples]).astype(np.int64)
    cmask = np.stack([s.card_mask for s in samples]).astype(np.float32)
    ttypes = np.stack([TOKEN_TYPES for _ in samples])

    a_feats = np.zeros((b, max_a, samples[0].a_feats.shape[1]), dtype=np.float32)
    a_cids = np.zeros((b, max_a, 2), dtype=np.int64)
    a_cmask = np.zeros((b, max_a, 2), dtype=np.float32)
    a_aids = np.zeros((b, max_a), dtype=np.int64)
    a_mask = np.zeros((b, max_a), dtype=np.float32)
    policy = np.zeros((b, max_a), dtype=np.float32)

    for i, s in enumerate(samples):
        k = len(s.policy)
        a_feats[i, :k] = s.a_feats[:k]
        a_cids[i, :k] = s.a_card_ids[:k]
        a_cmask[i, :k] = s.a_card_mask[:k]
        a_aids[i, :k] = s.a_attack_ids[:k]
        a_mask[i, :k] = 1.0
        policy[i, :k] = s.policy[:k]

    value = np.array([[s.value] for s in samples], dtype=np.float32)
    phi = np.stack([s.phi_target for s in samples]).astype(np.float32)

    t = lambda x: torch.from_numpy(x).to(device)  # noqa: E731
    return (t(slot), t(cids), t(cmask), t(ttypes),
            t(a_feats), t(a_cids), t(a_cmask), t(a_aids), t(a_mask),
            t(value), t(policy), t(phi))


def evaluate(model, samples, device, batch_size):
    """Loss e **accuratezza top-1** su un set di validazione.

    L'accuratezza e' la metrica leggibile: "quanto spesso la rete sceglie la
    stessa mossa dell'esperto". La cross-entropy da sola non e' confrontabile
    fra dataset con numeri di azioni diversi (lezione gia' pagata in questo
    progetto)."""
    model.eval()
    correct = total = 0
    vloss = 0.0
    nb = 0
    with torch.inference_mode():
        for i in range(0, len(samples), batch_size):
            chunk = samples[i: i + batch_size]
            if not chunk:
                continue
            (slot, cids, cmask, tt, af, ac, acm, aa, am,
             value, policy, phi) = make_batch(chunk, device)
            v, ph, logits = model(slot, cids, cmask, tt, af, ac, acm, aa, am)
            pred = logits.masked_fill(am < 0.5, float("-inf")).argmax(dim=-1)
            tgt = policy.argmax(dim=-1)
            correct += int((pred == tgt).sum())
            total += len(chunk)
            vloss += float(F.huber_loss(v, value, delta=0.5))
            nb += 1
    return (correct / max(total, 1)), (vloss / max(nb, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/replays")
    ap.add_argument("--out", default="out/pretrained.pth")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--policy-weight", type=float, default=1.0)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--phi-weight", type=float, default=0.5)
    ap.add_argument("--val-samples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    shards = sorted(glob.glob(str(Path(args.data) / "shard_*.pkl")))
    if not shards:
        print(f"nessuno shard in {args.data}: lancia prima build_replay_dataset.py",
              file=sys.stderr)
        return 1
    print(f"shard trovati: {len(shards)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    if args.resume and Path(args.resume).exists():
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"ripreso da {args.resume}")
    print(f"device: {device} | parametri: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Validazione presa dall'ultimo shard e mai usata in training: senza,
    # l'accuratezza misurata sarebbe quella di memorizzazione.
    val = load_shard(shards[-1])[: args.val_samples]
    train_shards = shards[:-1] if len(shards) > 1 else shards
    print(f"validazione: {len(val)} sample | shard di training: {len(train_shards)}")

    outpath = Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0

    for epoch in range(args.epochs):
        t0 = time.time()
        order = list(range(len(train_shards)))
        random.shuffle(order)
        model.train()
        seen = 0
        tot = {"policy": 0.0, "value": 0.0, "phi": 0.0}
        nb = 0

        for si in order:
            samples = load_shard(train_shards[si])
            random.shuffle(samples)
            for i in range(0, len(samples) - args.batch_size + 1, args.batch_size):
                chunk = samples[i: i + args.batch_size]
                (slot, cids, cmask, tt, af, ac, acm, aa, am,
                 value, policy, phi) = make_batch(chunk, device)

                optimizer.zero_grad(set_to_none=True)
                v, ph, logits = model(slot, cids, cmask, tt, af, ac, acm, aa, am)

                logp = torch.log_softmax(
                    logits.masked_fill(am < 0.5, float("-inf")), dim=-1
                )
                logp = torch.nan_to_num(logp, neginf=0.0)
                loss_policy = -(policy * logp).sum(dim=-1).mean()
                loss_value = F.huber_loss(v, value, delta=0.5)
                loss_phi = F.mse_loss(ph, phi)
                loss = (args.policy_weight * loss_policy
                        + args.value_weight * loss_value
                        + args.phi_weight * loss_phi)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                tot["policy"] += loss_policy.item()
                tot["value"] += loss_value.item()
                tot["phi"] += loss_phi.item()
                nb += 1
                seen += len(chunk)
            del samples
            sys.stderr.write(f"\r  epoca {epoch}: {seen} sample   ")
            sys.stderr.flush()
        sys.stderr.write("\n")

        acc, vloss = evaluate(model, val, device, args.batch_size)
        print(f"epoca {epoch}: policy={tot['policy'] / max(nb, 1):.4f} "
              f"value={tot['value'] / max(nb, 1):.4f} phi={tot['phi'] / max(nb, 1):.4f} "
              f"| VAL accuratezza={100 * acc:.1f}% value_loss={vloss:.4f} "
              f"[{time.time() - t0:.0f}s]", flush=True)

        # Si salva sul migliore in validazione, non sull'ultima epoca: il
        # behavioral cloning va in overfitting presto.
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), outpath)
            print(f"  nuovo migliore -> {outpath}")

    print(f"fatto. migliore accuratezza in validazione: {100 * best_acc:.1f}%")
    print(f"ora: python train_selfplay.py --resume {outpath} [...]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
