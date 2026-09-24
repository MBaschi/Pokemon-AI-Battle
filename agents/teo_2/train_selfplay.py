"""Training self-play (RL puro) per teo_2.

Ciclo AlphaZero-like:

  1. self-play: la MCTS guidata dalla rete gioca contro se stessa e produce,
     per ogni decisione, un target di policy (distribuzione delle visite) e
     uno stato da valutare;
  2. i target di value vengono calcolati a ritroso con TD(lambda), combinando
     l'esito finale della partita, il valore alla radice della MCTS e il
     *reward shaping* potential-based basato su Phi (reward.py);
  3. si allena la rete su value + policy + testa ausiliaria Phi;
  4. si ripete.

Nessun dato umano, nessuna euristica cablata nella policy: l'unica conoscenza
di dominio iniettata e' Phi, e lo e' in una forma (potential-based shaping)
che dimostrabilmente non cambia la politica ottima.

Generalita' fra mazzi: il self-play pesca a ogni partita una coppia di mazzi
dal pool (--decks). Poiche' l'encoding e' per *attributi* di carta e non per
card ID (vedi cards.py), un checkpoint allenato su piu' mazzi gioca anche
liste che non ha mai visto.

Esempi:
    python train_selfplay.py --iterations 10 --games 40 --sims 24
    python train_selfplay.py --decks ../gio_v1/deck.csv ../gharchomp_ex/deck.csv
"""

import argparse
import multiprocessing
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import cgpath  # noqa: F401  -- deve precedere qualsiasi import di cg.*

from cards import ATTACK_TABLE, CARD_TABLE
from encoding import encode_actions, encode_state, enumerate_actions
from mcts import SearchConfig, run_mcts
from model import action_tensors, build_model, count_parameters, state_tensors
from reward import COMPONENT_ORDER, phi as phi_fn

from cg.api import to_observation_class
from cg.game import battle_finish, battle_select, battle_start

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Dati di training
# ---------------------------------------------------------------------------

class Sample:
    __slots__ = ("state_enc", "action_enc", "policy", "value", "phi_target")

    def __init__(self, state_enc, action_enc, policy, phi_target):
        self.state_enc = state_enc
        self.action_enc = action_enc
        self.policy = policy
        self.value = 0.0          # riempito a fine partita
        self.phi_target = phi_target


def load_deck(path):
    rows = Path(path).read_text().split("\n")
    deck = [int(r) for r in rows if r.strip()][:60]
    if len(deck) != 60:
        raise ValueError(f"{path}: il mazzo ha {len(deck)} carte, ne servono 60")
    return deck


def default_decks():
    """Pool di default: tutti i deck.csv presenti nel repo."""
    candidates = [
        HERE / "deck.csv",
        REPO_ROOT / "agents" / "gio_v1" / "deck.csv",
        REPO_ROOT / "agents" / "teo_1" / "deck.csv",
        REPO_ROOT / "agents" / "gharchomp_ex" / "deck.csv",
    ]
    return [p for p in candidates if p.exists()]


# ---------------------------------------------------------------------------
# Self-play
# ---------------------------------------------------------------------------

def play_selfplay_game(decks, model, device, cfg, rng, max_turns=200):
    """Una partita della rete contro se stessa. Ritorna i sample delle due parti.

    `cfg` e' la TrainConfig: serve sia la SearchConfig annidata (per la MCTS)
    sia i parametri TD(lambda)/shaping usati per i target di value.
    """
    search_cfg = cfg.search
    deck0 = rng.choice(decks)
    deck1 = rng.choice(decks)
    obs, start_data = battle_start(deck0, deck1)
    if start_data.errorPlayer is not None and start_data.errorPlayer >= 0:
        battle_finish()
        raise ValueError(f"deck non valido (tipo {start_data.errorType})")

    decks_by_slot = (deck0, deck1)
    samples = ([], [])
    phis = ([], [])     # Phi(s_t) per ogni decisione, prospettiva del giocatore
    roots = ([], [])    # valore MCTS alla radice
    # Carte avversarie gia' osservate: migliora la determinizzazione.
    seen = ([], [])

    try:
        while obs["current"]["result"] < 0:
            if obs["current"]["turn"] > max_turns:
                break
            slot = obs["current"]["yourIndex"]
            o = to_observation_class(obs)
            my_deck = decks_by_slot[slot]

            action, policy, root_value = run_mcts(
                o, my_deck, model, device, search_cfg, rng, opponent_known=seen[1 - slot]
            )

            actions = enumerate_actions(o.select, search_cfg.max_actions)
            if len(actions) > 1:
                st = encode_state(o, my_deck)
                ae = encode_actions(o, actions)
                phi_target = np.array(
                    [
                        c
                        for c in _phi_components_list(o.current, slot)
                    ],
                    dtype=np.float32,
                )
                samples[slot].append(Sample(st, ae, policy, phi_target))
                phis[slot].append(phi_fn(o.current, slot, CARD_TABLE, ATTACK_TABLE))
                roots[slot].append(root_value)

            # Traccia le carte dell'avversario che diventano visibili.
            _observe(seen, o, slot)

            obs = battle_select(list(action))

        result = obs["current"]["result"]
        final_state = to_observation_class(obs).current
    finally:
        battle_finish()

    for slot in (0, 1):
        if result == 2 or result < 0:
            outcome = 0.0
        else:
            outcome = 1.0 if result == slot else -1.0
        phi_final = phi_fn(final_state, slot, CARD_TABLE, ATTACK_TABLE)
        _assign_value_targets(
            samples[slot], phis[slot], roots[slot], outcome, phi_final, cfg
        )

    return samples[0] + samples[1], result


def _phi_components_list(state, slot):
    from reward import phi_components

    comps = phi_components(state, slot, CARD_TABLE, ATTACK_TABLE)
    return [comps[k] for k in COMPONENT_ORDER]


def _observe(seen, obs, slot):
    """Registra le carte avversarie visibili (campo + scarti)."""
    opp = obs.current.players[1 - slot]
    bucket = seen[1 - slot]
    for c in opp.discard:
        bucket.append(c.id)
    for p in [x for x in opp.active if x] + [x for x in opp.bench if x]:
        bucket.append(p.id)
    if len(bucket) > 120:
        del bucket[:-120]


def _assign_value_targets(samples, phis, roots, outcome, phi_final, cfg):
    """TD(lambda) con reward shaping potential-based.

    r_t = gamma * Phi(s_{t+1}) - Phi(s_t)  (con Phi(s_T) = phi_final)

    Il ritorno telescopa: con gamma=1 la somma degli r_t lungo l'episodio vale
    Phi(s_T) - Phi(s_0), cioe' una costante rispetto alla politica. E' proprio
    questa proprieta' a garantire che lo shaping non introduca ottimi spuri:
    accelera l'apprendimento senza cambiarne il punto di arrivo.
    """
    n = len(samples)
    if n == 0:
        return
    gamma, lam, scale = cfg.gamma, cfg.td_lambda, cfg.shaping_scale

    # ATTENZIONE al valore di lambda in questo gioco. Una partita ha 150-250
    # *decisioni* (non mosse): ogni turno ne contiene molte. Con lambda=0.8 il
    # segnale terminale viene moltiplicato per 0.8 a ogni passo all'indietro,
    # quindi dopo 20 passi vale 0.8^20 = 0.012 ed e' di fatto invisibile al 90%
    # degli stati; il bootstrap, che a inizio training viene da un value net
    # non allenato, lo sostituisce con rumore centrato in zero. Risultato
    # misurato: target con std 0.16 tutti schiacciati sullo zero, value head
    # che impara a predire una costante, MCTS senza segnale e agente allenato
    # *piu' debole* di uno con pesi casuali (5-35 head-to-head).
    # Il default e' quindi lambda=1 (ritorno Monte Carlo puro, come AlphaZero).
    g = outcome
    for t in reversed(range(n)):
        phi_next = phi_final if t == n - 1 else phis[t + 1]
        r = scale * (gamma * phi_next - phis[t])
        if t == n - 1:
            g = r + gamma * outcome
        else:
            bootstrap = (1.0 - lam) * roots[t + 1] + lam * g
            g = r + gamma * bootstrap
        # Normalizzazione invece del solo clip: con lambda=1 il ritorno vive in
        # [outcome - 2*scale, outcome + 2*scale], quindi un clip secco
        # saturerebbe quasi tutti i target a +-1 buttando via proprio
        # l'informazione dello shaping. Dividere per l'ampiezza teorica la
        # conserva e usa tutto il codominio della tanh.
        samples[t].value = float(np.clip(g / (1.0 + 2.0 * scale), -1.0, 1.0))


# ---------------------------------------------------------------------------
# Self-play parallelo
# ---------------------------------------------------------------------------
#
# Deve essere per *processi*, non per thread, per due motivi indipendenti:
#
#   1. l'engine `cg` tiene il puntatore alla battaglia in una globale di modulo
#      (cg.sim.Battle.battle_ptr). Due partite nello stesso processo si
#      calpesterebbero a vicenda;
#   2. il GIL renderebbe comunque inutile il threading su lavoro CPU-bound.
#
# Su Windows il metodo di avvio e' `spawn`: ogni worker re-importa il modulo da
# zero, quindi tutto cio' che gli serve va passato esplicitamente e le funzioni
# devono stare a livello di modulo per essere picklabili.

_WORKER = {}


def _selfplay_worker_init(decks, cfg, threads):
    """Inizializza un processo worker. I pesi NON si caricano qui: arrivano col
    primo task, cosi' il pool sopravvive al cambio di generazione."""
    # Senza questo ogni worker aprirebbe i suoi 32 thread BLAS e i processi si
    # contenderebbero i core, rendendo il parallelismo piu' lento del seriale.
    torch.set_num_threads(max(1, threads))
    _WORKER["model"] = build_model()
    _WORKER["model"].eval()
    _WORKER["gen"] = -1
    _WORKER["decks"] = decks
    _WORKER["cfg"] = cfg
    _WORKER["device"] = torch.device("cpu")


def _selfplay_worker_run(task):
    """Gioca una partita. Non solleva mai: un worker che muore non deve
    abbattere il training, quindi l'errore torna come valore."""
    seed, ckpt_path, generation, phi_weight = task
    try:
        # Ricarica i pesi solo quando cambia la generazione: mantenere vivo il
        # pool fra le iterazioni evita di ripagare ~10-15 s di avvio processi
        # (import di torch + engine cg) a ogni giro.
        if _WORKER["gen"] != generation:
            _WORKER["model"].load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            _WORKER["model"].eval()
            _WORKER["gen"] = generation
        _WORKER["cfg"].search.phi_weight = phi_weight

        rng = random.Random(seed)
        with torch.inference_mode():
            samples, result = play_selfplay_game(
                _WORKER["decks"], _WORKER["model"], _WORKER["device"],
                _WORKER["cfg"], rng,
            )
        return samples, result, None
    except BaseException as exc:
        return [], -1, f"{type(exc).__name__}: {exc}"


def make_selfplay_pool(decks, cfg, workers, threads):
    ctx = multiprocessing.get_context("spawn")
    return ctx.Pool(
        processes=workers,
        initializer=_selfplay_worker_init,
        initargs=(decks, cfg, threads),
    )


def collect_selfplay_parallel(pool, model, cfg, base_seed, games, generation, tmpdir):
    """Genera `games` partite sul pool persistente. Ritorna la lista di sample.

    I pesi passano via file invece che come argomento del task: serializzarli
    per ogni partita significherebbe ri-picklare ~29 MB a giro.
    """
    ckpt = os.path.join(tmpdir, f"weights_gen{generation}.pth")
    torch.save(model.state_dict(), ckpt)

    tasks = [
        (base_seed * 100003 + i, ckpt, generation, cfg.search.phi_weight)
        for i in range(games)
    ]
    out = []
    done = 0
    for samples, _result, err in pool.imap_unordered(_selfplay_worker_run, tasks):
        done += 1
        if err:
            print(f"\n  partita saltata: {err}", file=sys.stderr)
        else:
            out.extend(samples)
        sys.stderr.write(f"\r  self-play {done}/{games} ({len(out)} sample)   ")
        sys.stderr.flush()
    sys.stderr.write("\n")
    try:
        os.remove(ckpt)  # una generazione per volta: non accumulare 29 MB a iterazione
    except OSError:
        pass
    return out


def collect_selfplay_sequential(model, decks, cfg, rng, games):
    """Versione a processo singolo, usata con --workers 1 e per il debug
    (le eccezioni restano visibili e i breakpoint funzionano)."""
    out = []
    model.eval()
    with torch.inference_mode():
        for g in range(games):
            try:
                samples, _result = play_selfplay_game(decks, model, device_of(model), cfg, rng)
            except Exception as exc:
                print(f"\n  partita {g} saltata: {exc}", file=sys.stderr)
                continue
            out.extend(samples)
            sys.stderr.write(f"\r  self-play {g + 1}/{games} ({len(out)} sample)   ")
            sys.stderr.flush()
    sys.stderr.write("\n")
    return out


def device_of(model):
    return next(model.parameters()).device


def resolve_workers(requested):
    """0 = automatico. Il tetto a 8 non e' il numero di core ma la RAM: ogni
    worker e' un processo torch completo (~0.7 GB), quindi 16 worker sarebbero
    ~11 GB solo di self-play."""
    if requested and requested > 0:
        return requested
    cpu = os.cpu_count() or 1
    return max(1, min(8, cpu // 2))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, optimizer, buffer, device, cfg, steps=0):
    """Un passo di training sul replay buffer.

    Con `steps > 0` esegue esattamente `steps` minibatch estratti a caso
    (approccio AlphaZero), invece di passate complete sul buffer. E' quello che
    serve su run lunghi: il costo per iterazione diventa **costante** invece di
    crescere col buffer, e non si ri-allena ripetutamente sugli stessi sample
    vecchi solo perche' il buffer si e' allungato.
    """
    model.train()
    if len(buffer) < cfg.batch_size:
        return None

    if steps > 0:
        batches = [
            random.sample(range(len(buffer)), cfg.batch_size) for _ in range(steps)
        ]
    else:
        order = list(range(len(buffer)))
        random.shuffle(order)
        n = len(order) // cfg.batch_size
        batches = [
            order[b * cfg.batch_size: (b + 1) * cfg.batch_size] for b in range(n)
        ]
    if not batches:
        return None
    n_batches = len(batches)

    totals = {"loss": 0.0, "value": 0.0, "policy": 0.0, "phi": 0.0,
              "target_entropy": 0.0, "kl": 0.0, "n_actions": 0.0}
    for idx in batches:
        batch = [buffer[i] for i in idx]

        slot, cids, cmask, ttypes = state_tensors([s.state_enc for s in batch], device)
        max_a = max(len(s.action_enc.actions) for s in batch)
        af, acid, acmask, aaid, amask = action_tensors(
            [s.action_enc for s in batch], device, max_a
        )

        value_t = torch.tensor(
            [[s.value] for s in batch], dtype=torch.float32, device=device
        )
        phi_t = torch.from_numpy(
            np.stack([s.phi_target for s in batch])
        ).to(device)

        policy_t = np.zeros((len(batch), max_a), dtype=np.float32)
        for i, s in enumerate(batch):
            k = min(len(s.policy), max_a)
            policy_t[i, :k] = s.policy[:k]
            total = policy_t[i].sum()
            if total > 0:
                policy_t[i] /= total
        policy_t = torch.from_numpy(policy_t).to(device)

        optimizer.zero_grad(set_to_none=True)
        value, phi_pred, logits = model(
            slot, cids, cmask, ttypes, af, acid, acmask, aaid, amask
        )

        loss_value = F.huber_loss(value, value_t, delta=0.5)
        # Cross-entropy contro la distribuzione delle visite della MCTS.
        logp = torch.log_softmax(logits.masked_fill(amask < 0.5, float("-inf")), dim=-1)
        logp = torch.nan_to_num(logp, neginf=0.0)
        loss_policy = -(policy_t * logp).sum(dim=-1).mean()
        loss_phi = F.mse_loss(phi_pred, phi_t)

        loss = loss_value + cfg.policy_weight * loss_policy + cfg.phi_loss_weight * loss_phi
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        totals["loss"] += loss.item()
        totals["value"] += loss_value.item()
        totals["policy"] += loss_policy.item()
        totals["phi"] += loss_phi.item()

        # La cross-entropy da sola non e' confrontabile fra iterazioni: cresce
        # con ln(n_azioni), e il numero di azioni legali cambia man mano che
        # l'agente sopravvive piu' a lungo e raggiunge posizioni piu' ricche.
        # KL = cross-entropy - entropia(target) isola la qualita' del fit ed e'
        # l'unica delle due che va davvero guardata.
        with torch.no_grad():
            tgt_ent = -(policy_t * torch.log(policy_t.clamp_min(1e-9))).sum(dim=-1).mean()
            totals["target_entropy"] += tgt_ent.item()
            totals["kl"] += loss_policy.item() - tgt_ent.item()
            totals["n_actions"] += amask.sum(dim=-1).mean().item()

    return {k: v / n_batches for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Valutazione
# ---------------------------------------------------------------------------

def random_agent_factory(rng):
    def act(obs_dict):
        o = to_observation_class(obs_dict)
        sel = o.select
        n = len(sel.option)
        k = min(sel.maxCount, n)
        if k <= 0:
            return []
        return rng.sample(range(n), max(sel.minCount, 1) if sel.minCount > 0 else k)

    return act


def evaluate_vs_random(decks, model, device, cfg, rng, games=20, max_turns=200):
    """Win rate contro un agente casuale: il pavimento minimo di sanita'."""
    model.eval()
    opponent = random_agent_factory(rng)
    wins = losses = 0
    eval_cfg = SearchConfig(
        n_simulations=cfg.eval_sims,
        c_puct=cfg.search.c_puct,
        dirichlet_eps=0.0,       # niente rumore in valutazione
        temperature=0.0,         # deterministico
        phi_weight=cfg.search.phi_weight,
        max_actions=cfg.max_actions,
    )
    with torch.inference_mode():
        for g in range(games):
            deck = rng.choice(decks)
            me = g % 2
            obs, _ = battle_start(deck, deck)
            try:
                while obs["current"]["result"] < 0:
                    if obs["current"]["turn"] > max_turns:
                        break
                    slot = obs["current"]["yourIndex"]
                    if slot == me:
                        o = to_observation_class(obs)
                        action, _, _ = run_mcts(o, deck, model, device, eval_cfg, rng)
                        obs = battle_select(list(action))
                    else:
                        obs = battle_select(opponent(obs))
                result = obs["current"]["result"]
            finally:
                battle_finish()
            if result == me:
                wins += 1
            elif result >= 0 and result != 2:
                losses += 1
    decisive = wins + losses
    return wins / decisive if decisive else 0.0


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoints(outdir, model, optimizer, it, args):
    """Salva i checkpoint dell'iterazione, senza mai poter abbattere il run.

    Due protezioni, entrambe imparate a caro prezzo su questo progetto:

      - la cartella viene ricreata a ogni salvataggio invece che una volta
        all'avvio. Se sparisce a run in corso (cancellata per errore, unita di
        rete che si stacca, pulizia automatica) un run da 24 ore non deve
        morire alla prima `torch.save`;
      - un errore di scrittura viene segnalato ma non propagato: perdere un
        checkpoint costa un'iterazione, perdere il run le costa tutte.
    """
    try:
        outdir.mkdir(parents=True, exist_ok=True)
        # `teo2_latest.pth` resta un state_dict puro: e' il file che carica
        # main.py, e non va cambiato di formato.
        torch.save(model.state_dict(), outdir / "teo2_latest.pth")
        # Checkpoint numerati diradati: su un run da 200 iterazioni salvarli
        # tutti sarebbero ~6 GB di disco per nulla.
        if args.ckpt_every > 0 and (it % args.ckpt_every == 0
                                    or it == args.iterations - 1):
            torch.save(model.state_dict(), outdir / f"teo2_iter{it:03d}.pth")
        # Stato completo per riprendere senza perdere i momenti di Adam: su un
        # run lungo un crash a meta' non deve costare il riavvio da zero.
        torch.save(
            {"model": model.state_dict(),
             "optimizer": optimizer.state_dict(),
             "iteration": it},
            outdir / "teo2_resume.pth",
        )
        return True
    except BaseException as exc:
        print(f"\n  [!] salvataggio checkpoint fallito ({exc}); il training prosegue",
              file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Config + main
# ---------------------------------------------------------------------------

class TrainConfig:
    def __init__(self, args):
        self.gamma = args.gamma
        self.td_lambda = args.td_lambda
        self.shaping_scale = args.shaping_scale
        self.batch_size = args.batch_size
        self.policy_weight = args.policy_weight
        self.phi_loss_weight = args.phi_loss_weight
        self.grad_clip = 1.0
        self.max_actions = args.max_actions
        self.eval_sims = max(1, args.sims // 2)
        self.search = SearchConfig(
            n_simulations=args.sims,
            c_puct=args.c_puct,
            dirichlet_eps=args.dirichlet_eps,
            temperature=args.temperature,
            phi_weight=args.phi_weight,
            max_actions=args.max_actions,
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--games", type=int, default=30, help="partite self-play per iterazione")
    ap.add_argument("--epochs", type=int, default=2, help="epoche di training per iterazione")
    ap.add_argument("--steps-per-iter", type=int, default=0,
                    help="minibatch casuali per iterazione invece di epoche complete "
                         "(0 = usa --epochs). Consigliato sui run lunghi: rende il "
                         "costo per iterazione costante al crescere del buffer")
    ap.add_argument("--sims", type=int, default=24, help="simulazioni MCTS per decisione")
    ap.add_argument("--eval-games", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--td-lambda", type=float, default=1.0,
                    help="1.0 = ritorno Monte Carlo puro. NON abbassarlo senza "
                         "controllare la std dei target di value: le partite qui "
                         "hanno 150-250 decisioni e valori < 1 cancellano il "
                         "segnale terminale (vedi _assign_value_targets)")
    ap.add_argument("--shaping-scale", type=float, default=0.25,
                    help="ampiezza del reward shaping da Phi")
    ap.add_argument("--phi-weight", type=float, default=0.3,
                    help="peso iniziale di Phi nel valore di foglia MCTS")
    ap.add_argument("--phi-decay-iters", type=int, default=20,
                    help="iterazioni su cui phi-weight decade a 0 (scala assoluta, "
                         "indipendente da --iterations)")
    ap.add_argument("--phi-loss-weight", type=float, default=0.5)
    ap.add_argument("--policy-weight", type=float, default=1.0)
    ap.add_argument("--c-puct", type=float, default=1.4)
    ap.add_argument("--dirichlet-eps", type=float, default=0.25)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-actions", type=int, default=64)
    ap.add_argument("--buffer", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=0,
                    help="processi per il self-play (0 = automatico). Il collo di "
                         "bottiglia e' la RAM, non i core: ogni worker e' un "
                         "processo torch completo (~0.7 GB)")
    ap.add_argument("--worker-threads", type=int, default=1,
                    help="thread BLAS per worker. 1 e' quasi sempre giusto: piu' "
                         "thread mettono i processi in contesa sugli stessi core")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decks", nargs="*", default=None)
    ap.add_argument("--out", default="out")
    ap.add_argument("--log-file", default="",
                    help="file dove appendere le righe di riepilogo per iterazione. "
                         "Evita di dover incanalare l'output nella shell (`tee` non "
                         "esiste in cmd.exe) e sopravvive alla chiusura del terminale")
    ap.add_argument("--ckpt-every", type=int, default=10,
                    help="ogni quante iterazioni salvare un checkpoint numerato "
                         "(teo2_latest.pth e teo2_resume.pth sono sempre aggiornati)")
    ap.add_argument("--resume", default=None,
                    help="path a teo2_resume.pth (ripristina anche l'optimizer) "
                         "oppure a un qualunque state_dict (solo i pesi)")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    def log(msg):
        """Stampa a schermo e, se richiesto, appende al file di log.

        Append e non write: un `--resume` non deve cancellare la storia del run
        precedente. Il file viene riaperto e chiuso a ogni riga, cosi' il log e'
        leggibile mentre il training gira e sopravvive a un crash del processo.
        """
        print(msg, flush=True)
        if args.log_file:
            try:
                with open(args.log_file, "a", encoding="utf-8") as fh:
                    fh.write(msg + "\n")
            except OSError as exc:
                print(f"  [!] scrittura log fallita: {exc}", file=sys.stderr)

    deck_paths = args.decks or default_decks()
    if not deck_paths:
        print("nessun mazzo trovato: passa --decks path/a/deck.csv", file=sys.stderr)
        return 1
    decks = [load_deck(p) for p in deck_paths]
    log(f"pool di mazzi: {len(decks)}")
    for p in deck_paths:
        log(f"  - {p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.resume and os.path.exists(args.resume):
        blob = torch.load(args.resume, map_location=device)
        if isinstance(blob, dict) and "model" in blob:
            model.load_state_dict(blob["model"])
            optimizer.load_state_dict(blob["optimizer"])
            log(f"ripreso da {args.resume} (pesi + optimizer, iterazione {blob.get('iteration')})")
        else:
            model.load_state_dict(blob)
            log(f"ripreso da {args.resume} (solo pesi)")

    workers = resolve_workers(args.workers)
    log(f"device: {device} | parametri: {count_parameters(model):,} | worker self-play: {workers}")
    cfg = TrainConfig(args)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    tmpdir_ctx = tempfile.TemporaryDirectory(prefix="teo2_")
    tmpdir = tmpdir_ctx.name

    # Il pool si crea una volta sola e vive per tutto il training: i pesi
    # aggiornati arrivano ai worker via file a ogni generazione.
    pool = make_selfplay_pool(decks, cfg, workers, args.worker_threads) if workers > 1 else None

    buffer = []
    try:
        for it in range(args.iterations):
            t0 = time.time()
            # Il peso di Phi nella foglia decade: serve da stampella iniziale,
            # poi la rete deve reggersi da sola. Il decadimento e' su una scala
            # *assoluta* (--phi-decay-iters), NON su args.iterations: legarlo
            # alla lunghezza del run renderebbe un probe da 3 iterazioni
            # (decadimento istantaneo) non rappresentativo di un run da 40, che
            # e' esattamente il modo in cui un run di prova mente sul risultato.
            decay = max(0.0, 1.0 - it / max(1, args.phi_decay_iters))
            cfg.search.phi_weight = args.phi_weight * decay

            model.eval()
            if pool is not None:
                new_samples = collect_selfplay_parallel(
                    pool, model, cfg, base_seed=args.seed + it,
                    games=args.games, generation=it, tmpdir=tmpdir,
                )
            else:
                new_samples = collect_selfplay_sequential(model, decks, cfg, rng, args.games)
            buffer.extend(new_samples)

            if len(buffer) > args.buffer:
                buffer = buffer[-args.buffer:]

            stats = None
            if args.steps_per_iter > 0:
                stats = train_epoch(
                    model, optimizer, buffer, device, cfg, steps=args.steps_per_iter
                )
            else:
                for _ in range(args.epochs):
                    stats = train_epoch(model, optimizer, buffer, device, cfg)

            save_checkpoints(outdir, model, optimizer, it, args)

            # Diagnostica che avrebbe intercettato subito il bug del lambda: se
            # i target di value sono tutti schiacciati sullo zero non c'e' nulla
            # da imparare, per quanto bene scenda la loss. Sotto ~0.3 e' allarme.
            tstd = float(np.std([s.value for s in buffer])) if buffer else 0.0
            line = (
                f"iter {it}: buffer={len(buffer)} phi_w={cfg.search.phi_weight:.2f} "
                f"v_std={tstd:.3f}"
            )
            if tstd < 0.3:
                line += " [!] target di value poco dispersi"
            if stats:
                line += (
                    f" v={stats['value']:.4f} phi={stats['phi']:.4f}"
                    # KL e' la metrica di fit da seguire; H e n_act servono a
                    # capire se un movimento della policy loss e' reale o solo
                    # uno spostamento della distribuzione del numero di azioni.
                    f" KL={stats['kl']:.4f} (H={stats['target_entropy']:.3f}"
                    f" n_act={stats['n_actions']:.1f})"
                )
            if args.eval_games > 0:
                wr = evaluate_vs_random(decks, model, device, cfg, rng, args.eval_games)
                line += f" wr_vs_random={100 * wr:.0f}%"
            line += f" [{time.time() - t0:.0f}s]"
            log(line)
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        tmpdir_ctx.cleanup()

    log(f"fatto. checkpoint finale: {outdir / 'teo2_latest.pth'}")
    return 0


if __name__ == "__main__":
    # Obbligatorio con il metodo di avvio `spawn` (default su Windows): senza
    # questa guardia ogni worker ri-eseguirebbe il training all'import.
    multiprocessing.freeze_support()
    raise SystemExit(main())
