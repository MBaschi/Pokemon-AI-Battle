"""Training della value net di marnie_grimmsnarl_ex_v2, a scala di avversari.

Cosa si allena, esattamente: **una funzione V(stato) -> [-1,1]**, dove lo stato
e' un *confine di turno* (il tratto e' appena passato). Niente policy, niente
MCTS, niente target di visite.

Perche' questo e' molto piu' facile del training di teo_2:

  - il target e' un solo scalare per stato, e viene dall'esito reale della
    partita (Monte Carlo puro), non da un bootstrap su una rete non allenata --
    che e' il bug che aveva reso teo_2 *piu' debole* dei pesi casuali;
  - i sample sono ~20-40 per partita invece di 150-250, e sono esattamente
    quelli che la ricerca interroghera' in partita. Nessuno scarto tra la
    distribuzione di training e quella di uso;
  - la rete ha ~290k parametri invece di 7.2M, con un vocabolario di ~60 carte
    invece di 1268;
  - e soprattutto: **anche una V mediocre non peggiora l'agente**, perche' puo'
    solo riordinare mosse che l'euristica giudica gia' equivalenti. Non c'e' la
    fase "il training danneggia l'agente" da attraversare.

La scala di avversari (`--ladder`), nell'ordine richiesto:

    0 random     -> pavimento di sanita'
    1 self       -> l'ibrido contro se stesso
    2 euristica  -> marnie_grimmsnarl_ex puro: il vero banco di prova, perche'
                    misura *l'apporto della rete* a parita' di tutto il resto
    3 gio_v1
    4 gio_v2

Si sale di gradino quando la win rate sulla finestra recente supera
`--promote-at`. Il gradino 2 e' il piu' informativo di tutti: e' l'unico
confronto in cui l'unica variabile e' la rete.

Reward: ritorno Monte Carlo con potential-based shaping su Phi. Con gamma=1 la
somma degli shaping lungo l'episodio **telescopa**, quindi non serve nessun
ciclo all'indietro:

    G_t = esito + scale * (Phi_finale - Phi_t)

E' la stessa quantita' che teo_2 calcola con TD(lambda=1), scritta in chiuso.

Esempi:
    python train.py --iterations 20 --games 120
    python train.py --iterations 60 --games 200 --resume out/marnie_v2_resume.pth
    python train.py --rung 2 --no-promote --games 300     # solo vs euristica
"""

import argparse
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import cgpath  # noqa: F401  -- deve precedere qualsiasi import di cg.*

from cg.api import to_observation_class
from cg.game import battle_finish, battle_select, battle_start

import heuristic
from cards import ATTACK_TABLE, CARD_TABLE, build_vocab
from encoding import encode_state
from model import (
    build_model,
    count_parameters,
    load as load_ckpt,
    save as save_ckpt,
    state_tensors,
)
from reward import phi as phi_fn, phi_vector
from search import HybridAgent, SearchConfig, TurnSearch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

MY_DECK = list(heuristic.DECK_LIST)


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------


class Sample:
    __slots__ = ("enc", "value", "phi_target")

    def __init__(self, enc, phi_target):
        self.enc = enc
        self.value = 0.0  # riempito a fine partita
        self.phi_target = phi_target


# ---------------------------------------------------------------------------
# Avversari
# ---------------------------------------------------------------------------


def load_deck(path):
    rows = Path(path).read_text().split("\n")
    deck = [int(r) for r in rows if r.strip()][:60]
    if len(deck) != 60:
        raise ValueError(f"{path}: il mazzo ha {len(deck)} carte, ne servono 60")
    return deck


def random_opponent(rng):
    """Risposta legale a caso. Il pavimento: perderci significa un bug."""

    def act(obs_dict):
        sel = obs_dict.get("select")
        if sel is None:
            return None  # il chiamante mette il mazzo
        opts = sel.get("option") or []
        n = len(opts)
        minc = sel.get("minCount") or 0
        maxc = min(sel.get("maxCount") or 0, n)
        if maxc <= 0:
            return []
        k = rng.randint(max(1, minc), maxc) if maxc >= max(1, minc) else maxc
        return rng.sample(range(n), k)

    return act


def repo_agent(main_py):
    """Carica un agente del repo (gio_v1, gio_v2, ...) isolandone i moduli."""
    sys.path.insert(0, str(REPO_ROOT))
    from benchmark_agents import load_agent

    loaded = load_agent(Path(main_py))
    return loaded.fn, loaded.deck


# Gradini in cui la win rate attesa e' 50% *per costruzione*, perche' i due
# lati sono la stessa policy sullo stesso mazzo. Applicargli una soglia di
# promozione bloccherebbe la scala per sempre: si passa oltre appena la
# finestra e' piena.
SYMMETRIC_RUNGS = frozenset({"self"})


def build_ladder(names, rng):
    """(nome, factory) per ogni gradino richiesto.

    La factory e' pigra: gio_v2 carica torch e un checkpoint suo, e non ha
    senso pagarlo se il run non arriva mai a quel gradino.
    """
    catalog = {
        "random": lambda: (random_opponent(rng), MY_DECK),
        "self": lambda: ("self", MY_DECK),
        "euristica": lambda: (heuristic.heuristic_agent, MY_DECK),
        "gio_v1": lambda: repo_agent(REPO_ROOT / "agents" / "gio_v1" / "main.py"),
        "gio_v2": lambda: repo_agent(REPO_ROOT / "agents" / "gio_v2" / "main.py"),
    }
    out = []
    for n in names:
        if n not in catalog:
            raise SystemExit(f"gradino sconosciuto: {n} (validi: {', '.join(catalog)})")
        out.append((n, catalog[n]))
    return out


def ladder_decks(names):
    """I mazzi che entreranno nel vocabolario: il nostro piu' quelli dei
    gradini che esistono su disco."""
    decks = [MY_DECK]
    for n in names:
        p = REPO_ROOT / "agents" / n / "deck.csv"
        if p.exists():
            try:
                decks.append(load_deck(p))
            except (OSError, ValueError):
                pass
    return decks


# ---------------------------------------------------------------------------
# Raccolta dati
# ---------------------------------------------------------------------------


def _visible_hand(state, player):
    ps = state.players[player]
    if ps.hand:
        return [c.id for c in ps.hand]
    return [] if ps.handCount == 0 else None


def collect_game(
    me_agent, opp_act, decks, my_slot, rng, max_turns=200, record_opponent_side=False
):
    """Gioca una partita e ritorna (sample, esito_per_me, turni).

    I sample si prendono ai **confini di turno**, cioe' esattamente dove la
    ricerca chiede una valutazione. Da entrambe le prospettive quando il lato
    avversario e' governato dalla stessa policy (self-play): contro un random
    non ha senso, perche' quelle posizioni non le incontreremo mai.

    La mano viene tracciata a mano (`last_hand`) perche' il motore la mostra
    solo a chi ha il tratto, e al confine di turno il tratto e' gia' passato:
    senza tracciamento la rete vedrebbe sempre una mano vuota proprio dove il
    contenuto della mano decide il turno successivo. Vedi encoding.encode_state.
    """
    obs_dict, start = battle_start(decks[0], decks[1])
    if start.errorPlayer is not None and start.errorPlayer >= 0:
        battle_finish()
        raise ValueError(f"mazzo non valido (tipo {start.errorType})")

    me_agent.reset()
    samples = []  # (slot, Sample, phi_t)
    last_hand = [None, None]
    prev_turn = None
    prev_state = None

    try:
        while obs_dict["current"]["result"] < 0:
            if obs_dict["current"]["turn"] > max_turns:
                break
            obs = to_observation_class(obs_dict)
            state = obs.current
            slot = state.yourIndex

            for p in (0, 1):
                h = _visible_hand(state, p)
                if h is not None:
                    last_hand[p] = h

            # Confine di turno: il contatore e' cambiato rispetto alla
            # decisione precedente. `prev_state` e' lo stato in cui il tratto
            # e' appena passato.
            if prev_turn is not None and state.turn != prev_turn:
                sides = (0, 1) if record_opponent_side else (my_slot,)
                for side in sides:
                    enc = encode_state(
                        _ObsShim(prev_state, obs),
                        decks[side],
                        me_agent.searcher.vocab,
                        me_idx=side,
                        hand_ids=last_hand[side],
                    )
                    ph = phi_vector(prev_state, side, CARD_TABLE, ATTACK_TABLE)
                    samples.append(
                        (
                            side,
                            Sample(enc, np.array(ph, dtype=np.float32)),
                            phi_fn(prev_state, side, CARD_TABLE, ATTACK_TABLE),
                        )
                    )

            prev_turn = state.turn
            prev_state = state

            if slot == my_slot:
                action = me_agent.select(obs)
            else:
                action = opp_act(obs_dict)
            obs_dict = battle_select(list(action))

        result = obs_dict["current"]["result"]
        final_state = to_observation_class(obs_dict).current
    finally:
        battle_finish()

    for side, sample, phi_t in samples:
        if result == 2 or result < 0:
            outcome = 0.0
        else:
            outcome = 1.0 if result == side else -1.0
        phi_final = phi_fn(final_state, side, CARD_TABLE, ATTACK_TABLE)
        sample.value = _shaped_target(outcome, phi_t, phi_final)

    if result == 2 or result < 0:
        my_outcome = 0.0
    else:
        my_outcome = 1.0 if result == my_slot else -1.0
    return [s for _, s, _ in samples], my_outcome, obs_dict["current"]["turn"]


SHAPING_SCALE = 0.25


def _shaped_target(outcome, phi_t, phi_final):
    """Ritorno Monte Carlo + shaping potential-based, in forma chiusa.

    Con gamma=1 la somma dei reward di shaping r_k = scale*(Phi_{k+1} - Phi_k)
    telescopa a scale*(Phi_T - Phi_t): non serve nessuna ricorsione all'indietro
    e non c'e' nessun lambda da sbagliare.

    La divisione per (1 + 2*scale) e' una normalizzazione, non un clip: il
    ritorno vive in [esito - 2*scale, esito + 2*scale] e un clip secco
    saturerebbe quasi tutti i target a +-1, buttando via proprio
    l'informazione dello shaping.
    """
    g = outcome + SHAPING_SCALE * (phi_final - phi_t)
    return float(np.clip(g / (1.0 + 2.0 * SHAPING_SCALE), -1.0, 1.0))


class _ObsShim:
    """Observation minimale: uno `state` passato e il `select` corrente.

    encode_state legge solo `.current`; il campo `select` c'e' per non
    sorprendere chi legge, ma non viene mai usato dall'encoder.
    """

    __slots__ = ("current", "select")

    def __init__(self, state, obs):
        self.current = state
        self.select = getattr(obs, "select", None)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_steps(
    model, optimizer, buffer, device, steps, batch_size, phi_weight, grad_clip=1.0
):
    if len(buffer) < batch_size:
        return None
    model.train()
    totals = {"value": 0.0, "phi": 0.0}
    for _ in range(steps):
        batch = [buffer[i] for i in random.sample(range(len(buffer)), batch_size)]
        slot, cidx, cmask, ttypes = state_tensors([s.enc for s in batch], device)
        v_t = torch.tensor(
            [[s.value] for s in batch], dtype=torch.float32, device=device
        )
        phi_t = torch.from_numpy(np.stack([s.phi_target for s in batch])).to(device)

        optimizer.zero_grad(set_to_none=True)
        value, phi_pred = model(slot, cidx, cmask, ttypes)
        loss_v = F.huber_loss(value, v_t, delta=0.5)
        loss_phi = F.mse_loss(phi_pred, phi_t)
        (loss_v + phi_weight * loss_phi).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        totals["value"] += loss_v.item()
        totals["phi"] += loss_phi.item()
    model.eval()
    return {k: v / steps for k, v in totals.items()}


@torch.inference_mode()
def calibration(model, buffer, device, n=512):
    """Correlazione tra V predetto e target, e dispersione delle predizioni.

    Le due misure che contano piu' della loss. Una value head puo' avere loss
    bassissima e predire *sempre la media*: succede se i target sono poco
    dispersi, ed e' invisibile guardando solo la loss che scende. `pred_std`
    vicino a zero significa esattamente questo.
    """
    if len(buffer) < 32:
        return None
    idx = random.sample(range(len(buffer)), min(n, len(buffer)))
    batch = [buffer[i] for i in idx]
    slot, cidx, cmask, ttypes = state_tensors([s.enc for s in batch], device)
    pred = model(slot, cidx, cmask, ttypes)[0].squeeze(-1).cpu().numpy()
    tgt = np.array([s.value for s in batch], dtype=np.float32)
    if pred.std() < 1e-6 or tgt.std() < 1e-6:
        return {"corr": 0.0, "pred_std": float(pred.std()), "tgt_std": float(tgt.std())}
    return {
        "corr": float(np.corrcoef(pred, tgt)[0, 1]),
        "pred_std": float(pred.std()),
        "tgt_std": float(tgt.std()),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--games", type=int, default=120, help="partite per iterazione")
    ap.add_argument("--steps-per-iter", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--buffer", type=int, default=60000)
    ap.add_argument("--phi-loss-weight", type=float, default=0.5)
    ap.add_argument(
        "--explore-eps",
        type=float,
        default=0.25,
        help="probabilita' di giocare un candidato a caso *dentro la "
        "banda di ambivalenza*. Serve a diversificare i dati: a 0 "
        "due agenti deterministici rigiocano quasi la stessa partita",
    )
    ap.add_argument("--rel-margin", type=float, default=0.15)
    ap.add_argument("--abs-margin", type=float, default=25.0)
    ap.add_argument("--max-candidates", type=int, default=4)
    ap.add_argument(
        "--phi-weight",
        type=float,
        default=1.0,
        help="peso iniziale di Phi nel valore di foglia. Parte a 1 "
        "(la rete non sa ancora niente) e decade a 0",
    )
    ap.add_argument("--phi-decay-iters", type=int, default=15)
    ap.add_argument(
        "--ladder",
        nargs="*",
        default=["random", "self", "euristica", "gio_v1", "gio_v2"],
    )
    ap.add_argument("--rung", type=int, default=0, help="gradino di partenza")
    ap.add_argument(
        "--no-promote",
        action="store_true",
        help="resta sul gradino iniziale (utile per un A/B mirato)",
    )
    ap.add_argument(
        "--promote-at",
        type=float,
        default=0.60,
        help="win rate sulla finestra oltre la quale si sale di gradino",
    )
    ap.add_argument(
        "--promote-window",
        type=int,
        default=200,
        help="partite minime prima di poter promuovere",
    )
    ap.add_argument(
        "--rung-patience",
        type=int,
        default=10,
        help="iterazioni massime su un gradino prima di salire comunque. "
        "Serve a non restare bloccati per sempre su un avversario "
        "che non si riesce a battere: i dati di un gradino piu' "
        "duro valgono comunque piu' di altre 50 iterazioni contro "
        "uno gia' saturo. 0 = nessun limite",
    )
    ap.add_argument("--max-turns", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out")
    ap.add_argument("--log-file", default="train_log.txt")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        """Stampa e appende al log. Append e non write: un --resume non deve
        cancellare la storia del run precedente. Il file viene riaperto a ogni
        riga, cosi' e' leggibile mentre il training gira."""
        print(msg, flush=True)
        if args.log_file:
            try:
                with open(args.log_file, "a", encoding="utf-8") as fh:
                    fh.write(msg + "\n")
            except OSError as exc:
                print(f"  [!] scrittura log fallita: {exc}", file=sys.stderr)

    device = torch.device("cpu")  # la rete e' piccola: la GPU non ripaga il transfer

    # Il vocabolario si fissa *una volta sola*, all'inizio, e include i mazzi di
    # tutta la scala: allargarlo a run iniziato cambierebbe la dimensione
    # dell'embedding e renderebbe incompatibili i checkpoint gia' salvati.
    vocab = build_vocab(MY_DECK, ladder_decks(args.ladder))
    # eval() esplicito: la raccolta dati interroga la rete, e con il dropout
    # attivo valuterebbe ogni foglia con una sottorete diversa -- rumore che
    # non si vede da nessuna metrica perche' train_steps rimette eval() alla
    # fine, quindi sparirebbe da tutte le misure tranne che dai dati raccolti.
    model = build_model(vocab).to(device).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    start_iter = 0

    if args.resume and Path(args.resume).exists():
        loaded, loaded_vocab, blob = load_ckpt(args.resume, map_location=device)
        if len(loaded_vocab) != len(vocab):
            raise SystemExit(
                f"{args.resume}: vocabolario da {len(loaded_vocab)} voci, "
                f"quello corrente ne ha {len(vocab)}. Usa la stessa --ladder del "
                "run originale, oppure riparti da zero."
            )
        model = loaded.to(device)
        vocab = loaded_vocab
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        if "optimizer" in blob:
            optimizer.load_state_dict(blob["optimizer"])
        start_iter = int(blob.get("iteration", 0)) + 1
        args.rung = int(blob.get("rung", args.rung))
        log(f"ripreso da {args.resume} (iterazione {start_iter}, gradino {args.rung})")

    log(f"vocabolario: {len(vocab)} voci | parametri: {count_parameters(model):,}")
    log(f"scala: {' -> '.join(args.ladder)}")

    ladder = build_ladder(args.ladder, rng)
    rung = min(args.rung, len(ladder) - 1)

    searcher = TurnSearch(
        model=model,
        vocab=vocab,
        device=device,
        cfg=SearchConfig(
            rel_margin=args.rel_margin,
            abs_margin=args.abs_margin,
            max_candidates=args.max_candidates,
            phi_weight=args.phi_weight,
            explore_eps=args.explore_eps,
        ),
        rng=rng,
        my_deck=MY_DECK,
    )
    me = HybridAgent(searcher)
    # L'avversario "self" ha il suo TurnSearch: condividerne uno solo
    # significherebbe condividere anche le carte osservate, cioe' far vedere a
    # ciascun lato l'informazione dell'altro.
    self_searcher = TurnSearch(
        model=model,
        vocab=vocab,
        device=device,
        cfg=searcher.cfg,
        rng=random.Random(args.seed + 7),
        my_deck=MY_DECK,
    )
    self_agent = HybridAgent(self_searcher)

    buffer = []
    opp_name, opp_factory = ladder[rung]
    opp_act, opp_deck = opp_factory()
    # Finestra scorrevole di esiti *per partita* (1 = vinta), non per iterazione:
    # e' l'unico modo perche' --promote-window significhi davvero "partite".
    window = deque(maxlen=args.promote_window)
    rung_iters = 0

    for it in range(start_iter, start_iter + args.iterations):
        t0 = time.time()
        # Phi come stampella nel valore di foglia: serve finche' la rete e'
        # rumore, poi deve togliersi di mezzo.
        decay = max(0.0, 1.0 - it / max(1, args.phi_decay_iters))
        searcher.cfg.phi_weight = args.phi_weight * decay

        wins = losses = draws = 0
        turns = []
        new = []
        for g in range(args.games):
            my_slot = g % 2  # lati alternati: confronto appaiato
            decks = [None, None]
            decks[my_slot] = MY_DECK
            decks[1 - my_slot] = opp_deck
            is_self = opp_name == "self"
            if is_self:
                self_agent.reset()
                other = lambda o: self_agent.select(
                    to_observation_class(o)
                )  # noqa: E731
            else:
                other = opp_act
            try:
                samples, outcome, nturns = collect_game(
                    me,
                    other,
                    decks,
                    my_slot,
                    rng,
                    args.max_turns,
                    record_opponent_side=is_self,
                )
            except BaseException as exc:
                print(
                    f"\n  partita {g} saltata: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                continue
            new.extend(samples)
            turns.append(nturns)
            if outcome > 0:
                wins += 1
                window.append(1)
            elif outcome < 0:
                losses += 1
                window.append(0)
            else:
                draws += 1  # patta/abort: fuori dal conteggio, come nel benchmark
            sys.stderr.write(
                f"\r  {opp_name}: {g + 1}/{args.games} "
                f"({wins}-{losses}-{draws}, {len(new)} sample)   "
            )
            sys.stderr.flush()
        sys.stderr.write("\n")

        buffer.extend(new)
        if len(buffer) > args.buffer:
            buffer = buffer[-args.buffer :]

        stats = train_steps(
            model,
            optimizer,
            buffer,
            device,
            args.steps_per_iter,
            args.batch_size,
            args.phi_loss_weight,
        )
        cal = calibration(model, buffer, device)

        decisive = wins + losses
        wr = wins / decisive if decisive else 0.0
        dec_sum = len(window)
        wwr = (sum(window) / dec_sum) if dec_sum else 0.0

        save_ckpt(outdir / "marnie_v2_latest.pth", model, vocab)
        save_ckpt(
            outdir / "marnie_v2_resume.pth",
            model,
            vocab,
            extra={"optimizer": optimizer.state_dict(), "iteration": it, "rung": rung},
        )

        s = searcher.stats
        rate = (100.0 * s["searched"] / s["decisions"]) if s["decisions"] else 0.0
        changed = (100.0 * s["changed"] / s["searched"]) if s["searched"] else 0.0
        line = (
            f"iter {it} [{opp_name}] wr={100 * wr:.0f}% (finestra {100 * wwr:.0f}%"
            f" su {dec_sum}) buffer={len(buffer)} phi_w={searcher.cfg.phi_weight:.2f}"
        )
        if stats:
            line += f" v={stats['value']:.4f} phi={stats['phi']:.4f}"
        if cal:
            line += f" corr={cal['corr']:+.2f} pred_std={cal['pred_std']:.3f}"
            if cal["pred_std"] < 0.05:
                line += " [!] la value head predice quasi una costante"
        line += f" | ricerca {rate:.0f}% delle scelte, cambia {changed:.0f}%"
        if turns:
            line += f" | turni {np.mean(turns):.0f}"
        line += f" [{time.time() - t0:.0f}s]"
        log(line)
        searcher.stats = {k: 0 for k in searcher.stats}

        rung_iters += 1
        window_full = dec_sum >= args.promote_window
        earned = window_full and (wwr >= args.promote_at or opp_name in SYMMETRIC_RUNGS)
        stalled = args.rung_patience > 0 and rung_iters >= args.rung_patience
        if not args.no_promote and rung < len(ladder) - 1 and (earned or stalled):
            rung += 1
            rung_iters = 0
            opp_name, opp_factory = ladder[rung]
            opp_act, opp_deck = opp_factory()
            window.clear()
            why = "soglia raggiunta" if earned else "pazienza esaurita"
            log(f"  --> gradino {rung}: {opp_name} ({why})")

    log(f"fatto. checkpoint: {outdir / 'marnie_v2_latest.pth'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
