"""Allena il ramo *value* di ``encoding.MyModel`` (stesso modello di teo_1) sul
dataset prodotto da ``build_replay_dataset.py``: apprendimento supervisionato
sull'esito reale della partita, invece che via self-play MCTS.

Il decoder (ramo policy) non viene allenato qui: gli passiamo un input
"vuoto" fisso solo per rispettare la firma di ``MyModel.forward`` — il ramo
encoder/value viene calcolato indipendentemente dal decoder, quindi non ha
alcun effetto sul valore appreso.

Uso:
    python build_replay_dataset.py --max-games 20
    python train_value_from_replays.py --data data/replay_value_samples.pkl
"""

from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

import torch
import torch.optim

from encoding import LearnInput, MyModel, SparseVector, build_model, dummy_decoder_input

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data" / "replay_value_samples.pkl"
DEFAULT_OUT = HERE / "out" / "value_from_replays.pth"


def _sv_from_sample(s: dict) -> SparseVector:
    sv = SparseVector()
    sv.index = s["index"]
    sv.value = s["value"]
    sv.offset = s["offset"]
    return sv


def group_split(samples: list[dict], test_frac: float, seed: int):
    """Split per partita (episode_id), cosi' nessuna partita finisce sia in
    train che in test — snapshot della stessa partita sono correlati."""
    episodes = sorted({s["episode_id"] for s in samples})
    rng = random.Random(seed)
    rng.shuffle(episodes)
    n_test = max(1, int(len(episodes) * test_frac))
    test_episodes = set(episodes[:n_test])
    train = [s for s in samples if s["episode_id"] not in test_episodes]
    test = [s for s in samples if s["episode_id"] in test_episodes]
    return train, test


def phase_of(turn_frac: float) -> str:
    if turn_frac < 1 / 3:
        return "early"
    if turn_frac < 2 / 3:
        return "mid"
    return "late"


def make_batch(batch: list[dict], device):
    input_enc = LearnInput()
    input_dec = LearnInput()
    labels = []
    for s in batch:
        input_enc.add(_sv_from_sample(s))
        input_dec.add(dummy_decoder_input())
        labels.append(s["label"])
    label_tensor = torch.tensor(labels, dtype=torch.float32, device=device).view(-1, 1)
    return (
        torch.tensor(input_enc.index, dtype=torch.int32, device=device),
        torch.tensor(input_enc.value, dtype=torch.float32, device=device),
        torch.tensor(input_enc.offset, dtype=torch.int32, device=device),
        torch.tensor(input_dec.index, dtype=torch.int32, device=device),
        torch.tensor(input_dec.value, dtype=torch.float32, device=device),
        torch.tensor(input_dec.offset, dtype=torch.int32, device=device),
        label_tensor,
    )


@torch.inference_mode()
def evaluate(model: MyModel, samples: list[dict], device, batch_size: int):
    model.eval()
    by_phase: dict[str, list[tuple[float, float]]] = {
        "early": [],
        "mid": [],
        "late": [],
    }
    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        *enc_dec_args, label_tensor = make_batch(batch, device)
        out_enc, _ = model(*enc_dec_args)
        preds = out_enc.view(-1).tolist()
        for s, pred in zip(batch, preds):
            by_phase[phase_of(s["turn_frac"])].append((pred, s["label"]))

    print("Valutazione (held-out per partita):")
    all_pairs = []
    for phase in ("early", "mid", "late"):
        pairs = by_phase[phase]
        all_pairs += pairs
        if not pairs:
            continue
        mse = sum((p - y) ** 2 for p, y in pairs) / len(pairs)
        decided = [(p, y) for p, y in pairs if y != 0.0]
        acc = (
            sum(1 for p, y in decided if (p > 0) == (y > 0)) / len(decided)
            if decided
            else float("nan")
        )
        print(f"  fase {phase:5s}: n={len(pairs):5d}  mse={mse:.4f}  acc={acc:.3f}")
    if all_pairs:
        mse = sum((p - y) ** 2 for p, y in all_pairs) / len(all_pairs)
        decided = [(p, y) for p, y in all_pairs if y != 0.0]
        acc = (
            sum(1 for p, y in decided if (p > 0) == (y > 0)) / len(decided)
            if decided
            else float("nan")
        )
        print(f"  totale     : n={len(all_pairs):5d}  mse={mse:.4f}  acc={acc:.3f}")


def train(
    data_path: Path,
    out_path: Path,
    init_checkpoint: Path | None,
    epochs: int,
    batch_size: int,
    lr: float,
    test_frac: float,
    seed: int,
):
    with open(data_path, "rb") as f:
        dataset = pickle.load(f)
    samples = dataset["samples"]
    print(f"Sample totali: {len(samples)}")

    train_samples, test_samples = group_split(samples, test_frac, seed)
    print(f"Train: {len(train_samples)}  Test: {len(test_samples)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    if init_checkpoint is not None:
        model.load_state_dict(torch.load(init_checkpoint, map_location=device))
        print(f"Pesi iniziali caricati da {init_checkpoint}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.HuberLoss(delta=0.2)

    print("--- prima del training ---")
    evaluate(model, test_samples, device, batch_size)

    rng = random.Random(seed)
    for epoch in range(epochs):
        model.train()
        rng.shuffle(train_samples)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(train_samples) - batch_size + 1, batch_size):
            batch = train_samples[i : i + batch_size]
            *enc_dec_args, label_tensor = make_batch(batch, device)
            optimizer.zero_grad()
            out_enc, _ = model(*enc_dec_args)
            loss = loss_fn(out_enc, label_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"Epoch {epoch + 1}/{epochs}  loss={total_loss / max(n_batches, 1):.4f}")

    print("--- dopo il training ---")
    evaluate(model, test_samples, device, batch_size)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Modello salvato: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="parti da un checkpoint esistente (es. out/model4.pth del self-play)",
    )
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    train(
        a.data,
        a.out,
        a.init_checkpoint,
        a.epochs,
        a.batch_size,
        a.lr,
        a.test_frac,
        a.seed,
    )


if __name__ == "__main__":
    main()
