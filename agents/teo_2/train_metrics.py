"""Lettura e lettura *critica* dei log di training di teo_2.

Il training non stampa tensorboard: stampa una riga per iterazione (vedi
`train_selfplay.log`). Questo modulo la trasforma in un DataFrame, la disegna e
prova a dire cosa sta succedendo, perche' su un run di 250 iterazioni la
domanda non e' "la loss scende?" (scende quasi sempre) ma "sta imparando
qualcosa che serve a vincere?".

Cosa significa ogni colonna, e cosa ci si aspetta di vedere:

  v_std   deviazione standard dei target di value nel buffer. E' il *primo*
          numero da guardare: se i target sono tutti uguali non c'e' segnale
          da imparare, per quanto bene scenda la loss. Sotto ~0.3 e' allarme
          rosso (e' il bug del TD(lambda) documentato nel README).
  KL      divergenza fra la policy della rete e la distribuzione di visite
          della MCTS: quanto la rete "e' d'accordo" con la ricerca. E' la vera
          metrica di fit della policy. La cross-entropy grezza no: cresce con
          ln(n_azioni), quindi si muove anche quando il fit non cambia.
  H       entropia del *target* (le visite MCTS). Scende se la ricerca diventa
          decisa; ma scende anche se le partite cambiano forma, quindi va
          letta insieme a n_act.
  n_act   numero medio di azioni legali per decisione. Cresce quando l'agente
          sopravvive piu' a lungo e raggiunge posizioni piu' ricche: e' un
          proxy indiretto di "gioca meglio", ed e' il motivo per cui KL non e'
          confrontabile fra iterazioni senza guardarlo.
  v, phi  loss della testa value e della testa ausiliaria Phi.
  phi_w   peso di Phi nel valore di foglia della MCTS: decade a 0 per
          costruzione, non e' una metrica di apprendimento.
  wr      win rate contro l'agente casuale, se `--eval-games > 0`. E' il
          pavimento di sanita', non una misura di forza: batterlo al 100% non
          dice niente sul confronto con un avversario vero.

API:
    parse_train_log(path)      -> DataFrame per iterazione
    parse_pretrain_log(path)   -> DataFrame per epoca (behavioral cloning)
    diagnose(df)               -> list[str], osservazioni in italiano
    plot_training(df)          -> figure matplotlib con i pannelli principali
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

# Palette: blu = serie principale, arancio = serie di confronto, grigio =
# riferimenti (soglie, medie). Poche tinte, sempre le stesse, mai per rango.
C_MAIN = "#2a78d6"
C_ALT = "#eb6834"
C_THIRD = "#1baf7a"
C_REF = "#8a8a85"
C_BAD = "#e34948"

HERE = Path(__file__).resolve().parent

# `iter 12: buffer=30000 phi_w=0.21 v_std=0.709 v=0.0295 phi=0.0084
#  KL=0.2260 (H=1.322 n_act=7.4) wr_vs_random=45% [338s]`
_ITER_RE = re.compile(r"^iter\s+(\d+):")
_KV_RE = re.compile(r"([A-Za-z_]+)=(-?\d+(?:\.\d+)?)")
_TIME_RE = re.compile(r"\[(\d+(?:\.\d+)?)s\]")

_COLUMNS = {
    "buffer": "buffer",
    "phi_w": "phi_w",
    "v_std": "v_std",
    "v": "loss_value",
    "phi": "loss_phi",
    "KL": "kl",
    "H": "entropy",
    "n_act": "n_actions",
    "wr_vs_random": "wr_vs_random",
}


def parse_train_log(path=None):
    """Legge un log di `train_selfplay.py` e ritorna un DataFrame per iterazione.

    Il file puo' contenere piu' run appesi uno dopo l'altro (`--resume` appende
    invece di sovrascrivere, di proposito): la colonna `run` li separa,
    riconoscendoli da un contatore di iterazione che torna indietro. Tenerli
    distinti conta, perche' una curva che "riparte" e' quasi sempre un resume e
    non un peggioramento del modello.
    """
    path = Path(path) if path else HERE / "train_log.txt"
    rows = []
    run = 0
    prev_iter = None

    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        m = _ITER_RE.match(line.strip())
        if not m:
            continue
        it = int(m.group(1))
        if prev_iter is not None and it <= prev_iter:
            run += 1
        prev_iter = it

        row = {"run": run, "iter": it, "warn": "[!]" in line}
        for key, value in _KV_RE.findall(line):
            col = _COLUMNS.get(key)
            if col:
                row[col] = float(value)
        mt = _TIME_RE.search(line)
        if mt:
            row["seconds"] = float(mt.group(1))
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Asse x continuo anche con piu' run appesi: `step` e' il progressivo
    # globale, `iter` resta quello stampato dal training.
    df["step"] = np.arange(len(df))
    df["hours"] = df.get("seconds", pd.Series(0.0, index=df.index)).fillna(0).cumsum() / 3600.0
    return df


# `epoca 0: policy=1.2345 value=0.0456 phi=0.0123 | VAL accuratezza=41.2%
#  value_loss=0.0501 [812s]`
_EPOCH_RE = re.compile(r"^epoca\s+(\d+):")
_ACC_RE = re.compile(r"accuratezza=(\d+(?:\.\d+)?)%")


def parse_pretrain_log(path):
    """Legge il log di `pretrain_from_replays.py` (una riga per epoca)."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        m = _EPOCH_RE.match(line.strip())
        if not m:
            continue
        row = {"epoch": int(m.group(1))}
        for key, value in _KV_RE.findall(line):
            if key in ("policy", "value", "phi", "value_loss"):
                row[key] = float(value)
        ma = _ACC_RE.search(line)
        if ma:
            row["val_accuracy"] = float(ma.group(1)) / 100.0
        mt = _TIME_RE.search(line)
        if mt:
            row["seconds"] = float(mt.group(1))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Diagnosi
# ---------------------------------------------------------------------------

def _slope(series, window):
    """Pendenza per iterazione sugli ultimi `window` punti (regressione lineare)."""
    y = pd.Series(series).dropna().to_numpy()[-window:]
    if len(y) < 3:
        return 0.0
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def diagnose(df, window=40):
    """Osservazioni sul run, in italiano, ordinate da "grave" a "informativa".

    Sono euristiche, non verdetti: servono a sapere *dove guardare*. Le soglie
    sono quelle documentate nel README di teo_2, pagate con run veri.
    """
    if df is None or df.empty:
        return ["Nessuna iterazione trovata nel log."]

    out = []
    last = df.iloc[-1]
    n = len(df)
    w = min(window, n)

    if "v_std" in df:
        v_std = float(last["v_std"])
        if v_std < 0.3:
            out.append(
                f"[!] v_std={v_std:.3f} < 0.30: i target di value sono schiacciati "
                "sullo zero. La value head sta imparando una costante e la MCTS "
                "resta senza segnale — controlla --td-lambda (vedi README)."
            )
        else:
            out.append(f"v_std={v_std:.3f}: i target di value sono ben dispersi, ok.")

    if "kl" in df:
        s = _slope(df["kl"], w)
        kl = float(last["kl"])
        if abs(s) * w < 0.01:
            out.append(
                f"KL ferma a {kl:.3f} (variazione {s * w:+.3f} sulle ultime {w} "
                "iterazioni): la policy non si avvicina piu' alla MCTS. E' il "
                "collo di bottiglia tipico di teo_2 — piu' simulazioni per "
                "decisione o pretraining sui replay, non piu' iterazioni."
            )
        elif s < 0:
            out.append(f"KL in calo ({s * w:+.3f} su {w} iterazioni, ora {kl:.3f}): "
                       "la policy sta seguendo la ricerca, e' il segnale che vuoi.")
        else:
            out.append(f"KL in aumento ({s * w:+.3f} su {w} iterazioni, ora {kl:.3f}): "
                       "la ricerca si allontana dalla policy; guarda se n_act sta "
                       "crescendo (partite piu' lunghe = decisioni piu' difficili).")

    if "n_actions" in df:
        s = _slope(df["n_actions"], w)
        out.append(
            f"n_act {float(df['n_actions'].iloc[0]):.1f} -> {float(last['n_actions']):.1f} "
            f"({s * w:+.1f} sulle ultime {w}): "
            + ("le partite raggiungono posizioni piu' ricche, di solito buon segno."
               if s > 0 else "posizioni piu' povere o partite piu' corte.")
        )

    if "entropy" in df:
        out.append(
            f"H {float(df['entropy'].iloc[0]):.3f} -> {float(last['entropy']):.3f}: "
            + ("la ricerca e' piu' decisa che a inizio run."
               if float(last["entropy"]) < float(df["entropy"].iloc[0])
               else "la ricerca non e' piu' decisa di prima.")
        )

    if "loss_value" in df:
        s = _slope(df["loss_value"], w)
        out.append(f"loss value {float(last['loss_value']):.4f} "
                   f"({s * w:+.4f} sulle ultime {w} iterazioni).")

    if "wr_vs_random" in df and df["wr_vs_random"].notna().any():
        out.append(f"win rate vs random: {float(df['wr_vs_random'].dropna().iloc[-1]):.0f}%. "
                   "E' il pavimento di sanita': serve a scoprire un agente rotto, "
                   "non a stimare la forza.")
    else:
        out.append("Nessun win rate nel log (--eval-games 0): l'unica misura di "
                   "forza vera resta benchmark_agents.py contro un altro agente.")

    if "seconds" in df:
        out.append(f"{n} iterazioni, {df['seconds'].sum() / 3600:.1f} h totali, "
                   f"{df['seconds'].mean():.0f} s/iterazione in media.")
    if df["run"].nunique() > 1:
        out.append(f"Il log contiene {df['run'].nunique()} run appesi (resume): "
                   "la colonna `run` li separa.")
    return out


# ---------------------------------------------------------------------------
# Grafici
# ---------------------------------------------------------------------------

def _panel(ax, df, col, title, ylabel, color=C_MAIN, threshold=None, thr_label=None):
    if col not in df or df[col].dropna().empty:
        ax.set_axis_off()
        ax.set_title(f"{title} — assente nel log", fontsize=10, color=C_REF)
        return
    ax.plot(df["step"], df[col], color=color, linewidth=2)
    if threshold is not None:
        ax.axhline(threshold, color=C_BAD, linewidth=1, linestyle="--")
        ax.text(df["step"].iloc[0], threshold, f" {thr_label or threshold}",
                color=C_BAD, fontsize=8, va="bottom")
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.15, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # Etichetta diretta sull'ultimo punto invece di una legenda per un solo
    # valore: il numero che interessa e' sempre quello corrente.
    last = df[col].dropna()
    if not last.empty:
        ax.annotate(f"{last.iloc[-1]:.4g}",
                    (df["step"].loc[last.index[-1]], last.iloc[-1]),
                    textcoords="offset points", xytext=(4, 0), fontsize=9, color=color)


def plot_training(df, figsize=(13, 9)):
    """Griglia di pannelli con le metriche per iterazione.

    Un asse per grandezza (mai due scale sullo stesso grafico) e una tinta per
    ruolo: e' l'unico modo per cui "la curva sale" significhi sempre la stessa
    cosa quando si passa da un pannello all'altro.
    """
    import matplotlib.pyplot as plt

    if df is None or df.empty:
        raise ValueError("DataFrame vuoto: nessuna iterazione da disegnare.")

    fig, axes = plt.subplots(3, 2, figsize=figsize)
    _panel(axes[0][0], df, "kl", "KL(policy ‖ visite MCTS) — fit della policy",
           "KL", C_MAIN)
    _panel(axes[0][1], df, "v_std", "Dispersione dei target di value",
           "std", C_THIRD, threshold=0.3, thr_label="0.30 = allarme")
    _panel(axes[1][0], df, "loss_value", "Loss value (Huber)", "loss", C_MAIN)
    _panel(axes[1][1], df, "loss_phi", "Loss testa ausiliaria Φ", "loss", C_ALT)
    _panel(axes[2][0], df, "entropy", "Entropia del target (H)", "nat", C_MAIN)
    _panel(axes[2][1], df, "n_actions", "Azioni legali per decisione", "n", C_ALT)

    for ax in axes[2]:
        ax.set_xlabel("iterazione", fontsize=9)
    # Confine fra run appesi: senza, un resume sembra un salto inspiegabile.
    if df["run"].nunique() > 1:
        for r in sorted(df["run"].unique())[1:]:
            x = float(df[df["run"] == r]["step"].iloc[0])
            for row in axes:
                for ax in row:
                    if ax.has_data():
                        ax.axvline(x, color=C_REF, linewidth=1, linestyle=":")
    fig.suptitle("teo_2 — self-play, metriche per iterazione", fontsize=13)
    fig.tight_layout()
    return fig


def plot_progress(df, figsize=(14, 3.4)):
    """Pannelli "di contorno": buffer, peso di Phi, tempo, win rate vs random.

    I primi tre non sono metriche di apprendimento — servono a capire il *costo*
    del run e a spiegare gli scalini nelle altre curve (il buffer che satura, Phi
    che si spegne). Il quarto compare solo se il training e' stato lanciato con
    `--eval-games > 0`, ed e' un pavimento di sanita': scopre un agente rotto,
    non misura la forza.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=figsize)
    _panel(axes[0], df, "buffer", "Sample nel replay buffer", "sample", C_MAIN)
    _panel(axes[1], df, "phi_w", "Peso di Φ nella foglia MCTS", "phi_w", C_ALT)
    _panel(axes[2], df, "seconds", "Durata iterazione", "s", C_THIRD)
    _panel(axes[3], df, "wr_vs_random", "Win rate vs agente casuale", "%", C_MAIN)
    for ax in axes:
        ax.set_xlabel("iterazione", fontsize=9)
    fig.tight_layout()
    return fig


def plot_pretrain(df, figsize=(11, 3.4)):
    """Curve del behavioral cloning: loss policy e accuratezza in validazione."""
    import matplotlib.pyplot as plt

    if df is None or df.empty:
        raise ValueError("Nessuna epoca di pretraining nel log.")
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    d = df.rename(columns={"epoch": "step"})
    _panel(axes[0], d, "policy", "Loss policy (cross-entropy)", "loss", C_MAIN)
    _panel(axes[1], d, "value", "Loss value", "loss", C_ALT)
    _panel(axes[2], d, "val_accuracy", "Accuratezza top-1 in validazione",
           "frazione", C_THIRD)
    for ax in axes:
        ax.set_xlabel("epoca", fontsize=9)
    fig.tight_layout()
    return fig


def plot_match_eval(decisions_df, names=("giocatore 0", "giocatore 1"), result=None,
                    figsize=(13, 4)):
    """Curva di valutazione di una partita (`Match.to_dataframe()`).

    Come la barra di valutazione di un motore di scacchi: sopra il 50% sta
    meglio il giocatore 0. La curva e' la stima della *rete di teo_2*, quindi
    dice cosa pensa il modello, non cosa e' vero; i punti evidenziati sono le
    decisioni dopo cui la stima e' scesa di piu' per chi aveva appena mosso.

    `result` (`Match.result`: 0, 1 o 2) aggiunge in fondo l'esito reale. Vale la
    pena passarlo: la distanza fra l'ultima stima e l'esito e' il modo piu'
    rapido per accorgersi che la rete non ha visto arrivare la sconfitta.
    """
    import matplotlib.pyplot as plt

    d = decisions_df.dropna(subset=["win%_p0"])
    if d.empty:
        raise ValueError("Nessuna valutazione: la partita e' stata giocata senza evaluator.")

    fig, ax = plt.subplots(figsize=figsize)
    x = d["step"].to_numpy()
    y = d["win%_p0"].to_numpy()
    if result in (0, 1, 2):
        x = np.append(x, x[-1] + 1)
        y = np.append(y, 100.0 if result == 0 else 0.0 if result == 1 else 50.0)
    ax.axhline(50, color=C_REF, linewidth=1, linestyle="--")
    ax.fill_between(x, 50, y, where=(y >= 50), color=C_MAIN, alpha=0.25, interpolate=True)
    ax.fill_between(x, 50, y, where=(y < 50), color=C_BAD, alpha=0.25, interpolate=True)
    ax.plot(x, y, color=C_MAIN, linewidth=2)

    worst = d[(d["n_azioni"] > 1) & d["equity_persa"].notna()].nlargest(5, "equity_persa")
    if not worst.empty:
        ax.scatter(worst["step"], worst["win%_p0"], s=42, facecolor="white",
                   edgecolor=C_BAD, linewidth=2, zorder=3)
        # Etichette alternate sopra/sotto: con due errori vicini si sovrappongono.
        for k, (_, r) in enumerate(worst.sort_values("step").iterrows()):
            ax.annotate(f"{r['scelta'][:26]}\n−{r['equity_persa']:.0f} pt",
                        (r["step"], r["win%_p0"]), textcoords="offset points",
                        xytext=(0, 10 if k % 2 == 0 else -26), fontsize=7,
                        ha="center", color=C_REF)
    ax.set_ylim(0, 100)
    ax.set_xlabel("decisione")
    ax.set_ylabel(f"win % stimata per {names[0]}")
    ax.set_title(f"Valutazione: {names[0]} (sopra) vs {names[1]} (sotto)", fontsize=12)
    ax.grid(True, alpha=0.15, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


def plot_calibration(cal_df, figsize=(5.2, 5)):
    """Affidabilita' del value head: predetto contro realizzato.

    Sulla diagonale = calibrato. Sotto la diagonale = la rete e' troppo
    ottimista, sopra = troppo pessimista. Il diametro dei punti e' quante
    decisioni cadono in quel bin: i bin quasi vuoti non vanno interpretati.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], color=C_REF, linewidth=1, linestyle="--")
    sizes = 30 + 220 * (cal_df["n"] / max(1, cal_df["n"].max()))
    ax.scatter(cal_df["predetto"], cal_df["reale"], s=sizes, color=C_MAIN, alpha=0.75)
    ax.plot(cal_df["predetto"], cal_df["reale"], color=C_MAIN, linewidth=1.5, alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("win probability predetta")
    ax.set_ylabel("frequenza di vittoria reale")
    ax.set_title("Calibrazione della value head", fontsize=12)
    ax.grid(True, alpha=0.15, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


def checkpoint_table(outdir=None):
    """Inventario dei checkpoint su disco: iterazione, dimensione, data.

    Serve a rispondere alla domanda pratica "quale file carico?" e a scegliere
    due generazioni distanti da far scontrare in arena.
    """
    outdir = Path(outdir) if outdir else HERE / "out"
    if not outdir.is_dir():
        return pd.DataFrame()
    rows = []
    for p in sorted(outdir.glob("*.pth")):
        m = re.search(r"iter(\d+)", p.name)
        stat = p.stat()
        rows.append({
            "file": p.name,
            "iter": int(m.group(1)) if m else None,
            "MB": round(stat.st_size / 1e6, 1),
            "modificato": pd.Timestamp(stat.st_mtime, unit="s", tz="UTC").tz_convert(
                "Europe/Rome").strftime("%Y-%m-%d %H:%M"),
            "path": str(p),
        })
    return pd.DataFrame(rows).sort_values(["iter", "file"], na_position="last")
