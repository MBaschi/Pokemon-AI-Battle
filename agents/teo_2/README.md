# teo_2

Agente **reinforcement learning puro**: impara a giocare via self-play, senza
dati umani e senza policy euristica cablata. Ispirato a `teo_1` (MCTS +
rete value/policy) ma con tre cambiamenti sostanziali:

1. **deck-agnostico** — l'encoding descrive le carte per *attributi*, non per
   card ID, quindi un checkpoint gioca anche liste mai viste in training;
2. **reward shaping con Φ** — una funzione potenziale a 8 componenti guida
   l'apprendimento senza alterarne il punto di arrivo;
3. **rete molto più grande** (7.2M parametri contro ~1M) con MCTS PUCT vera.

## File

| file | ruolo |
|---|---|
| `cgpath.py` | rende importabile il package `cg` (va importato per primo) |
| `reward.py` | **Φ**: le 8 componenti, il potenziale, il reward shaping |
| `cards.py` | tabella di feature statiche per carta/attacco — la chiave della generalità |
| `encoding.py` | stato → 18 token, azioni → feature; enumerazione delle azioni legali |
| `model.py` | `TeoNet`: value + policy + testa ausiliaria Φ |
| `mcts.py` | MCTS PUCT con determinizzazione, Dirichlet, budget temporale |
| `train_selfplay.py` | ciclo RL: self-play → target TD(λ) → training |
| `main.py` | **agente live** (`agent(obs_dict) -> list[int]`) |
| `deck.csv` | mazzo pilotato in submission (sostituibile: l'agente è generico) |
| `dashboard.ipynb` | **dashboard**: curve di training, replay annotati, calibrazione |
| `train_metrics.py` | legge `train_log.txt` → DataFrame, grafici, diagnosi del run |
| `replay_export.py` | gioca partite in locale ed esporta il replay HTML annotato |

## La funzione di reward Φ

Ogni componente è **antisimmetrica** (io − avversario) e a valori in `[-1, 1]`.
Φ è la media pesata, quindi anch'essa in `[-1, 1]` e antisimmetrica:
`Φ(s, io) == -Φ(s, avversario)`.

| componente | peso | cosa misura |
|---|---|---|
| `prize` | **1.00** | rilassamento continuo del contatore prize |
| `attack_ready` | 0.15 | copertura del costo del miglior attacco dell'attivo |
| `energy` | 0.10 | stessa frazione sui 2 Pokémon migliori del proprio lato |
| `evolution` | 0.08 | stage medio normalizzato su 2 |
| `board_safety` | 0.12 | numero di Pokémon in gioco, **non monotono** |
| `deckout` | 0.10 | `max(0, 6 − carte_in_mazzo)/6`, differenziale |
| `type_matchup` | 0.05 | +1 se il mio attivo colpisce la loro weakness |
| `conditions` | 0.03 | `(loro_condizioni − mie_condizioni)/3` |

Dettagli che contano:

- **`prize`** — per ogni Pokémon in gioco, `ko_frac = min(1, danno/HP_max)`
  pesata per i prize che quel Pokémon cede. Il **cap a 1 è essenziale**:
  l'overkill non deve valere più di un KO, altrimenti l'agente impara a
  caricare danno su un bersaglio già morto invece di spalmarlo. Il peso per
  `prize_value` riproduce la prize-trade math vera — danneggiare un ex vale
  il doppio.
- **`board_safety`** — `-1` con 1 solo Pokémon (un KO è sconfitta immediata),
  `0` con 2, cresce fino a 4, poi **piatta**. Over-benchare non va premiato:
  ogni Pokémon fragile in più è un bersaglio gratis per Boss's Orders.
- **`attack_ready` / `energy`** — la copertura energetica usa un matching
  greedy corretto (prima i simboli tipati, poi i `{C}` con quello che avanza),
  non il semplice conteggio delle energie: una `{F}` paga un `{C}` ma non
  viceversa.

### Perché lo shaping è sicuro

Φ non entra nella reward come bonus additivo, ma come **potential-based
shaping** (Ng–Harada–Russell 1999):

```
r(s, s') = γ · Φ(s') − Φ(s)
```

È l'unica forma di shaping che *dimostrabilmente non cambia la politica
ottima*: con γ=1 la somma lungo l'episodio telescopa a `Φ(s_T) − Φ(s_0)`,
una costante rispetto alla politica. Conseguenza pratica importante: **anche
se i pesi qui sopra fossero sbagliati, l'agente non può convergere a qualcosa
di peggio di quello che imparerebbe col solo segnale vittoria/sconfitta.**
I pesi cambiano *quanto in fretta* impara, non *cosa* impara — quindi si
possono tarare liberamente senza rischio.

Φ viene usata in tre punti:

1. **shaping della reward** durante il calcolo dei target TD(λ);
2. **valore di foglia della MCTS**, con peso che **decade a zero** durante il
   training (`--phi-weight`): a inizio training la rete è rumore puro e Φ dà
   un segnale sensato da subito, poi la rete deve reggersi da sola;
3. **feature di input** (token globale) e **target di una testa ausiliaria**.

## Perché è deck-agnostico

`teo_1` codifica ogni carta come indice one-hot su ~1300 card ID: una carta
mai vista ha un embedding non allenato, quindi il modello va ri-allenato per
ogni mazzo. `teo_2` usa **due viste in parallelo**:

- **embedding per card ID** → memorizza le carte che conosce;
- **proiezione degli attributi statici** (tipo, HP, stage, ritirata, weakness,
  danno/costo del miglior attacco…) → *generalizza* alle carte che non
  conosce. «Stage 2 da 330 HP con attacco da 260» resta interpretabile anche
  se quel card ID non è mai comparso.

Mano, mazzo e scarti sono descritti dalla **media** delle feature delle carte
che contengono: cattura la composizione senza dipendere da ordine o
dimensione. Il self-play pesca a ogni partita una coppia di mazzi dal pool
(`--decks`), così la generalità viene allenata, non solo sperata.

## Architettura della rete (`TeoNet`, 7.2M parametri)

```
18 token  ->  [proj attributi + embedding carte + embedding tipo-token]
          ->  Transformer encoder pre-LN  (4 layer, 8 teste, d_model 256)
          ->  pooling mean+max
              |-- value head   -> tanh, scalare in [-1,1]
              |-- phi head     -> 8 componenti (ausiliaria)
          ->  azioni: self-attention tra loro + cross-attention sullo stato
              (2 layer)  -> policy head -> 1 logit per azione
```

Token: 2 attivi + 10 panchina + 2 riepiloghi giocatore + mano + mazzo +
stadio + globale.

Due scelte che contano:

- **le azioni si attenzionano fra loro** prima di essere valutate. Scegliere
  una mossa è intrinsecamente comparativo: una policy che vede le alternative
  sceglie molto meglio di una che le valuta in isolamento. `teo_1` valuta
  ogni azione indipendentemente.
- **testa ausiliaria su Φ**. Predire qualcosa che sappiamo già calcolare
  esattamente sembra inutile, ma è un classico auxiliary task: costringe
  l'encoder a costruire internamente le rappresentazioni di prize-race,
  energia ed evoluzioni, invece di sperare che emergano dal solo segnale di
  vittoria — che arriva **una volta a partita** ed è estremamente sparso.

## MCTS

Rispetto a `teo_1`:

- **PUCT vero**: `Q + c_puct · P · √N_padre / (1 + N_figlio)`. `teo_1` usa per
  i figli non visitati il Q del *padre*, il che sottostima sistematicamente le
  mosse mai provate.
- **rumore di Dirichlet alla radice**: senza, il self-play collassa sulla
  mossa che la rete già preferisce e non esplora mai — è la causa singola più
  comune per cui un training self-play non decolla.
- **selezione per temperatura**: stocastica a inizio partita, deterministica
  dopo (`--temperature`, `temp_moves`).
- **budget temporale esplicito**, obbligatorio: il timeout è sconfitta
  immediata.

**Informazione nascosta.** `search_begin` pretende un'ipotesi *completa* su
mazzi, prize e mano avversaria, quindi ogni ricerca lavora su una singola
determinizzazione. È un'approssimazione nota (*strategy fusion*): l'agente
finge di conoscere carte che non conosce. Mitigata usando le carte
dell'avversario **già osservate** (campo + scarti) invece di riempire tutto
con un filler come fa `teo_1`, ma non eliminata — servirebbe una ricerca su
information set, fuori budget.

## Self-play parallelo

Il self-play è distribuito su più processi (`--workers`, 0 = automatico).
Deve essere per **processi** e non per thread, per due motivi indipendenti:
l'engine `cg` tiene il puntatore alla battaglia in una globale di modulo
(`cg.sim.Battle.battle_ptr`), quindi due partite nello stesso processo si
calpesterebbero; e il GIL renderebbe comunque inutile il threading su lavoro
CPU-bound.

Su Windows il metodo di avvio è `spawn`: ogni worker re-importa tutto da zero.
Due conseguenze progettuali:

- **il pool è persistente** per tutto il training, non ricreato a ogni
  iterazione. I pesi aggiornati arrivano via file e il worker li ricarica solo
  quando cambia la generazione. Ricreare il pool costerebbe ~10-15 s di avvio
  processi a ogni giro;
- **`torch.set_num_threads(1)` in ogni worker** (`--worker-threads`). Senza,
  ogni processo aprirebbe i suoi 32 thread BLAS e la contesa sui core
  renderebbe il parallelismo più lento del seriale.

Aspettative realistiche sullo speedup, misurate su 32 core:

| configurazione | 32 partite |
|---|---|
| sequenziale (32 thread BLAS) | ~196 s (estrapolato) |
| 8 worker × 1 thread | 76 s |
| 16 worker × 1 thread | 80 s |

Circa **2.6x**, non 8x, e 16 worker non sono meglio di 8. Il motivo è che il
sequenziale *stava già* usando tutti i core via threading BLAS: il guadagno
reale è solo la parte non parallelizzabile che si recupera. Il tetto
automatico a 8 worker è dettato dalla RAM (ogni worker è un processo torch
completo, ~0.7 GB), non dai core.

### Tempi per iterazione (misurati, 32 partite, 8 worker, 300 step, CPU)

| simulazioni | self-play | training | totale | iterazioni in 24 h |
|---|---|---|---|---|
| 32 | 312 s | 103 s | **415 s** (6.9 min) | ~208 |
| 64 | 571 s | 103 s | **674 s** (11.2 min) | ~130 |

Il costo del self-play è **sublineare** nelle simulazioni (2x sims → 1.83x
tempo), ma resta la voce dominante: il training è 103 s fissi grazie a
`--steps-per-iter`. Attenzione che il tempo per iterazione *cresce* durante il
run, perché un agente più forte fa partite più lunghe.

Dato che 12 → 128 simulazioni sposta l'entropia del target solo da 0.918 a
0.882, e che il segnale di apprendimento (`H` in calo) è stato osservato a 12
simulazioni, su un budget fisso conviene spendere in **iterazioni** più che in
simulazioni.

## Comandi

```bash
cd agents/teo_2

# training (i default sono conservativi: alza games/sims se hai GPU)
python train_selfplay.py --iterations 10 --games 30 --sims 24

# self-play parallelo (0 = automatico, min(8, core/2))
python train_selfplay.py --iterations 40 --games 32 --sims 64 --workers 8

# run lungo: passi fissi invece di epoche, cosi' il costo per iterazione non
# cresce col buffer (vedi --steps-per-iter)
python train_selfplay.py --iterations 120 --games 32 --sims 64 \
    --workers 8 --steps-per-iter 300 --batch-size 128 --buffer 30000 \
    --phi-decay-iters 40 --eval-games 0

# pool di mazzi esplicito -> generalità
python train_selfplay.py --decks deck.csv ../gio_v1/deck.csv ../teo_1/deck.csv

# riprendere
python train_selfplay.py --resume out/teo2_latest.pth

# valutare contro un altro agente del repo
cd ../.. && python benchmark_agents.py \
    --agent1 agents/teo_2/main.py --agent2 agents/gio_v1/main.py --games 20
```

`main.py` carica `out/teo2_latest.pth` (o `$TEO2_CHECKPOINT`).

## Dashboard: capire cosa sta imparando

`dashboard.ipynb` (da eseguire in `agents/teo_2`) mette insieme tre cose che le
righe di log da sole non danno.

**1. Curve per iterazione** — `train_metrics.parse_train_log()` trasforma
`train_log.txt` in un DataFrame e `plot_training()` lo disegna; `diagnose()`
prova a leggerlo per te (v_std sotto 0.3, KL piatta, n_act che cresce…). Le
soglie sono quelle pagate a caro prezzo e documentate qui sotto.

**2. Replay annotato** — `replay_export.py` gioca una partita in locale sul
motore `cg`, prende gli step dal motore stesso (`cg.game.visualize_data`) e li
passa al viewer condiviso di `view_replays/replay_render.py`, che sa già
disegnare tre campi opzionali per step:

| campo | cosa mostra |
|---|---|
| `eval_p0` | la barra di valutazione sotto la plancia, cliccabile |
| `agent_scores` | le mosse candidate della MCTS con valore (win %), visite `N`, prior `P` |
| `debug_out` | valore alla radice, Φ, quanto è cambiata la valutazione dopo la mossa |

```bash
# replay annotato contro un altro agente del repo
python replay_export.py --opponent ../gharchomp_ex/main.py --sims 64

# due generazioni a confronto
python replay_export.py --opponent self \
    --checkpoint out/teo2_iter000.pth --opponent-checkpoint out/teo2_latest.pth \
    --games 3 --sims 32
```

La valutazione viene sempre dalla value head del **lato 0**, anche quando muove
l'avversario: una curva stimata da due cervelli diversi avrebbe salti che non
significano niente. E resta la valutazione del modello che stiamo giudicando —
dice "la rete pensa di stare peggio", non "la mossa era oggettivamente cattiva".

`Match.blunders()` ordina le decisioni per equity persa (l'analogo della
centipawn loss): ci finisce dentro anche la risposta avversaria e la fortuna,
quindi serve a *trovare i punti interessanti*, non a dare voti.

**3. Le due misure che le loss non danno** — `arena()` fa scontrare due
checkpoint (l'unica misura di forza vera), `calibration()` confronta la win
probability predetta con l'esito reale. Una value head può avere loss bassissima
ed essere sistematicamente troppo sicura di sé: la loss misura l'errore medio, la
calibrazione se il numero *significa* quello che dice.

La strumentazione lato MCTS è il parametro opzionale `stats_out` di `run_mcts`:
riempie un dict con prior, visite e Q di ogni azione alla radice. È puramente
osservativo — training e agente live lo lasciano a `None` e non pagano nulla.

## Degradazione graduale

L'agente resta giocabile in ogni condizione, dalla più forte alla più debole:

1. checkpoint presente → MCTS con value/policy della rete;
2. **checkpoint assente → stessa MCTS ma valore di foglia = Φ**. Φ da sola è
   una valutazione di stato sensata, quindi l'agente è legale e non idiota
   anche senza allenamento;
3. torch assente o ricerca fallita → greedy statico;
4. qualunque eccezione → prima opzione legale.

## Stato attuale e aspettative oneste

Verificato end-to-end sull'engine reale: encoding, rete, MCTS, self-play,
training e agente live girano senza crash (0 crash nel benchmark del repo).

### Risultati del primo training (6 iterazioni, 12 partite, 12 sims, CPU)

| iter | value loss | phi loss | policy loss | wr vs random |
|---|---|---|---|---|
| 0 | 0.0268 | 0.0740 | 1.4687 | 83% |
| 1 | 0.0192 | 0.0300 | 1.4815 | 92% |
| 2 | 0.0157 | 0.0205 | 1.4943 | 58% |
| 3 | 0.0130 | 0.0167 | 1.4838 | 75% |
| 4 | 0.0131 | 0.0147 | 1.4914 | 83% |
| 5 | 0.0128 | 0.0130 | 1.4898 | 83% |

Contro `gio_v1`: **2/10** da allenato contro **0/4** da non allenato, con la
durata media delle partite salita da 8.0 a 14.4 turni. Campione piccolo, ma
la direzione è coerente.

### Bug trovato: TD(λ) cancellava il segnale terminale

Il primo run "approfondito" (3 iterazioni, 128 sims) ha prodotto un agente che
**perdeva 5-35 contro un modello con pesi casuali**. Non rumore: intervallo di
confidenza [2%, 22%]. Il training stava attivamente peggiorando l'agente.

Causa: `--td-lambda 0.8` con episodi da **150-250 decisioni**. Una partita di
questo gioco non ha ~100 mosse come una di Go: ogni turno contiene molte
selezioni separate. Con λ=0.8 il segnale terminale viene moltiplicato per 0.8 a
ogni passo all'indietro, quindi dopo 20 passi vale `0.8²⁰ = 0.012` ed è di
fatto invisibile al 90% degli stati. Al suo posto resta il bootstrap, che a
inizio training viene da un value net non allenato, cioè rumore centrato in
zero.

Misurato prima e dopo il fix (λ=1.0, ritorno Monte Carlo puro come AlphaZero,
più normalizzazione al posto del clip secco):

| | std dei target | quota con \|target\| > 0.5 |
|---|---|---|
| prima (λ=0.8, scale=0.5) | 0.0926 | 1% |
| dopo (λ=1.0, scale=0.25) | **0.6737** | **100%** |

Il value head stava imparando a predire una costante (`-0.05 ± 0.07`) perché
era esattamente ciò che i target gli chiedevano. La loss scendeva
regolarmente — **una loss che scende non dice nulla se il target è degenere.**

`train_selfplay.py` ora stampa `v_std` a ogni iterazione e segnala
`[!] target di value poco dispersi` sotto 0.3, così l'errore non può ripetersi
in silenzio.

**Effetto del fix, verificato head-to-head su 40 partite contro pesi casuali:**

| | risultato |
|---|---|
| prima del fix (3 iter) | 5-35 → **12%** — il training *danneggiava* l'agente |
| dopo il fix (4 iter) | 19-21 → **48%** — nessun effetto, ma nessun danno |

Il fix ha rimosso il danno attivo; non ha ancora prodotto un miglioramento,
il che con **40 partite di self-play totali** è del tutto atteso e non è
evidenza che il resto funzioni o non funzioni. Da questo punto in poi serve
compute, non altre patch.

### Il primo segnale positivo: `H` scende

Su un run con pool persistente (3 iterazioni × 16 partite):

```
iter 0: v=0.1511 phi=0.1071 KL=0.1705 (H=1.422)
iter 1: v=0.0974 phi=0.0653 KL=0.2864 (H=1.256)
iter 2: v=0.0888 phi=0.0480 KL=0.3249 (H=1.199)
```

**`H` (entropia del target MCTS) scende da 1.422 a 1.199.** Man mano che il
value head migliora, la ricerca concentra davvero le visite e i target di
policy diventano più informativi. È esattamente il contrario di quanto
affermato nella prima diagnosi («il target è bloccato al pavimento uniforme»),
ed è il primo indicatore che il ciclo di apprendimento gira come dovrebbe.

Osservazione ancora aperta: la `KL` cresce (0.17 → 0.32) mentre `H` scende.
La cross-entropy totale (`H + KL`) cala comunque, quindi la policy non sta
peggiorando in assoluto: sta **inseguendo un bersaglio mobile**. Il buffer in
questi test non è mai stato troncato (5620 < 20000), quindi contiene ancora i
target piatti della generazione 0 e la rete ne fitta la miscela. Da
riverificare a buffer saturo: se la KL continua a salire **con buffer
stabile**, lì c'è un problema vero — probabilmente un buffer troppo lungo che
tiene in vita target di policy stantii.

### Perché due diagnosi precedenti erano sbagliate

Vale la pena registrarle, perché sono trappole di misura generali:

1. **«La policy loss deve scendere»** — falso. La cross-entropy cresce con
   `ln(n_azioni)`, e il numero di mosse legali aumenta man mano che l'agente
   sopravvive più a lungo e raggiunge posizioni più ricche. La loss può salire
   mentre la policy migliora. La metrica corretta è la **KL**
   (`cross-entropy − entropia del target`), ora loggata.
2. **«L'entropia normalizzata del target è ~uniforme, quindi non c'è
   segnale»** — falso. L'entropia è dominata dalla coda, non dal picco: una
   distribuzione di visite `[45, 31, 22, 15, 15]` è chiaramente informativa
   (3x tra prima e ultima) ma ha comunque `H/ln(n) = 0.94`.

Entrambe nascevano dallo stesso errore di fondo: **valutare su 6-12 partite**,
dove la banda di rumore (±15-20%) ingoia qualsiasi effetto. Per confronti
usare il head-to-head appaiato su ≥40 partite, non il win rate contro random.

**Un agente RL non allenato perde**, e va detto chiaramente: senza
training serio `teo_2` perde contro `gio_v1`. Non è un bug, è la natura
dell'approccio. `gio_v1` è un'euristica scritta a mano da un giocatore che sa
cosa fa; batterla richiede *compute*, non solo architettura. Ordini di
grandezza indicativi:

| budget | attesa ragionevole |
|---|---|
| ~15 min CPU | batte l'agente casuale, perde contro le euristiche |
| ore su GPU | competitivo con le euristiche semplici |
| giorni su GPU | possibilità concreta di superare `gio_v1` |

Il punto della richiesta era «impari via self-play e possa superare i bot
euristici»: l'infrastruttura per farlo c'è ed è corretta, il superamento è
funzione del training che ci si mette. Il ciclo è progettato per essere
ripreso (`--resume`) e allungato senza rifare nulla.

### Come misurare, prima ancora di cosa tarare

L'errore più costoso finora non è stato un iperparametro sbagliato: è stato
**misurare male**. In ordine di importanza:

1. **Non fidarsi del win rate contro random su poche partite.** ±15-20% di
   rumore su 12 partite. Per qualsiasi confronto usare il **head-to-head
   appaiato su ≥40 partite** contro il checkpoint precedente (o contro pesi
   casuali, come test di sanità).
2. **Guardare `v_std`, non solo le loss.** Una loss che scende su un target
   degenere non significa niente. Sotto 0.3 il training non può funzionare,
   qualunque cosa dicano le altre metriche.
3. **Guardare `KL`, non la policy loss grezza.** La seconda cresce con
   `ln(n_azioni)` e si muove quando cambia la distribuzione delle posizioni,
   non quando cambia la qualità della policy.

### Cosa taro per primo

1. **`--sims`** (default 24). Resta il moltiplicatore di qualità più diretto
   della MCTS, ma con un'aspettativa ridimensionata dalla misura: passare da
   12 a 128 sposta l'entropia normalizzata del target solo da 0.918 a 0.882,
   al costo di 10x in tempo. 64 è un compromesso ragionevole; non aspettarsi
   miracoli dal salire oltre.
2. **`--shaping-scale`** (default 0.25). Quanto lo shaping pesa rispetto al
   segnale terminale. Segnale che è troppo alto: value loss bassissima e win
   rate piatto — sta imparando a predire Φ invece di vincere.
3. **`--phi-decay-iters`** (default 20). Per quanto a lungo tenere la
   stampella di Φ nel valore di foglia. Alzarlo se l'agente peggiora appena
   `phi_w` arriva a 0, segno che il value head non è ancora pronto a
   sostituirla.

Da **non** toccare per primo: `--td-lambda` (abbassarlo sotto 1 riapre
esattamente il bug documentato sopra), `--c-puct` (misurato bilanciato:
`spread(Q)/media(U) = 1.13`), la larghezza della rete.
