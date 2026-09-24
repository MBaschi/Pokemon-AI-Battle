# marnie_grimmsnarl_ex_v2

Ibrido tra `marnie_grimmsnarl_ex` (euristica a parametri tarati) e una **value
net** piccola, allenata per self-play su una scala di avversari.

La regola che tiene insieme le due metà, e da cui discende tutto il resto:

> **L'euristica decide sempre. La rete interviene solo quando l'euristica
> giudica due o più mosse equivalenti.**

Non l'inverso. Lasciare che una policy debole scavalchi un'euristica più forte
è un downgrade, e "identificare la mossa critica" è a sua volta un'euristica —
quindi non si può appaltare alla rete. Così invece il **pavimento è v1**: con la
banda di ambivalenza a zero questo agente *è* v1, mossa per mossa.

| file | ruolo |
|---|---|
| `cgpath.py` | rende importabile il package `cg` (va importato per primo) |
| `heuristic.py` | lo strato euristico: copia di v1 + l'API per la ricerca |
| `params.json` | i pesi tarati di v1 (unica superficie che il tuner tocca) |
| `reward.py` | **Φ**: potenziale a 8 componenti, shaping e valutazione di fallback |
| `cards.py` | feature statiche per carta + **vocabolario ridotto** degli embedding |
| `encoding.py` | stato → 18 token (nessun encoding delle azioni) |
| `model.py` | `MarnieValueNet`: solo value + testa ausiliaria Φ, ~290k parametri |
| `search.py` | candidati ambivalenti, rollout a completamento di turno, `HybridAgent` |
| `main.py` | **agente live** (`agent(obs_dict) -> list[int]`) |
| `train.py` | ciclo di training a scala di avversari |
| `deck.csv` | il mazzo (identico a v1) |

## Come decide una mossa

1. `heuristic.score_options` dà un punteggio a **ogni** opzione legale. È lo
   stesso codice che l'handler userebbe per decidere: la ricerca legge la
   classifica, non ne calcola una seconda. Due opinioni sulla stessa mossa sono
   il modo classico in cui un ibrido diventa peggiore di entrambe le sue metà.
2. Si prendono le opzioni entro una banda dal massimo
   (`abs_margin + rel_margin · |migliore|`, banda *relativa* perché i punteggi
   dell'euristica non hanno una scala comune tra contesti). Se ne resta una
   sola — il caso più frequente — si gioca quella e non si spende niente.
3. Per ciascun candidato si applica la mossa nel motore determinizzato e si
   **finisce il turno con l'euristica**, fino a quando passa il tratto.
4. Si valuta lo stato di confine con la rete e si tiene il migliore.

Nei log di training la riga `ricerca N% delle scelte, cambia M%` dice quanto
sta davvero lavorando la rete: tipicamente entra su ~35% delle decisioni
classificabili e ne cambia ~20%.

### Perché il rollout, e non la valutazione immediata

Valutare la posizione subito dopo una singola opzione di MAIN non dice quasi
niente: un turno qui contiene 5-15 decisioni, e «gioco Poffin» e «attacco»
portano a stati che non stanno sulla stessa scala. Completare il turno porta
tutti i candidati **allo stesso confine** — il momento in cui passa il tratto —
che è anche l'unico punto in cui «chi sta meglio?» ha una risposta ben
definita. Ed è esattamente la distribuzione su cui la rete viene allenata:
nessuno scarto tra come si allena e come viene interrogata.

Il rollout usa l'euristica, non la rete: costa microsecondi per decisione, e il
costo vero sono le chiamate al motore.

### La mano nascosta

Il motore mostra la mano solo a chi ha il tratto, e lo stato di confine è per
definizione uno stato in cui il tratto è **appena passato**: senza correttivi
la rete vedrebbe sempre una mano vuota proprio dove il contenuto della mano
decide il turno dopo (avere Rare Candy + Grimmsnarl ex in mano *è* metà del
piano del mazzo). Misurato su una partita reale: la mano risultava invisibile
in **21 valutazioni su 25**.

`search.complete_turn` traccia quindi la mano dall'ultimo stato in cui era
visibile e la passa a `encode_state(..., hand_ids=...)`; `train.collect_game`
fa la stessa cosa sui dati. Se un giorno le due parti divergessero, la rete
verrebbe allenata su una feature che in partita non esiste — è il tipo di bug
che non si vede da nessuna metrica di training.

## L'encoder ridotto

`teo_2` tiene un embedding per **ogni** card ID del gioco: 1268 × 256 = 325k
parametri, più 399k per gli attacchi. In una partita di questo mazzo se ne
aggiornano forse 40: tutti gli altri restano inizializzazione casuale e vengono
comunque letti dall'attenzione. È rumore pagato a peso pieno.

Qui il vocabolario è **chiuso e piccolo**: le 19 carte distinte del mazzo, più
quelle dei mazzi avversari della scala (32 voci in tutto), più l'indice 0 =
«carta fuori vocabolario» — che è un simbolo vero e allenato, non un padding.
Una carta fuori vocabolario non diventa invisibile: continua ad arrivare alla
rete attraverso il **vettore di attributi** (tipo, HP, stage, ritirata,
weakness, danno/costo del miglior attacco), che è la vista che generalizza.
Perde solo l'identità memorizzata, che per una carta mai vista non avrebbe
comunque significato.

| voce | teo_2 | qui |
|---|---|---|
| embedding carte | 325k | ~8k |
| embedding attacchi | 399k | 0 |
| encoder | 3.2M | ~180k |
| decoder azioni + policy head | 2.4M | **0** |
| teste value/Φ | ~600k | ~90k |
| **totale** | **7.2M** | **~290k** |

Il vocabolario **viaggia dentro il checkpoint** (`model.save`/`model.load`):
un checkpoint e il suo vocabolario non possono disallinearsi, che sarebbe
l'unico modo silenzioso di rompere tutto.

## Niente policy head

Sparisce l'intero ramo decoder/cross-attention di `teo_2` e sparisce il target
di policy. Il motivo non è il risparmio di parametri: è che **chi propone le
mosse è già l'euristica**, che su questo mazzo è molto più forte di qualsiasi
policy allenabile con un budget CPU. Alla rete resta il solo compito che una
rete piccola impara con poche migliaia di partite — «quanto vale questa
posizione» — e che è una regressione su uno scalare invece di una
classificazione su decine di azioni eterogenee.

Come effetto collaterale sparisce anche la parte più fragile del training di
`teo_2`: entropia del target, KL che insegue un bersaglio mobile, buffer che
tiene in vita target di policy stantii.

## Training

```bash
cd agents/marnie_grimmsnarl_ex_v2

# scala completa (default)
python train.py --iterations 40 --games 200

# solo contro l'euristica pura: l'unico A/B in cui la sola variabile è la rete
python train.py --rung 2 --no-promote --games 300 --iterations 20

# riprendere (ripristina anche l'optimizer e il gradino raggiunto)
python train.py --resume out/marnie_v2_resume.pth

# valutare
cd ../.. && python benchmark_agents.py \
    --agent1 agents/marnie_grimmsnarl_ex_v2/main.py \
    --agent2 agents/gio_v1/main.py --games 100
```

`main.py` carica `out/marnie_v2_latest.pth` (o `$MARNIE2_CHECKPOINT`).

### La scala di avversari

```
0 random     pavimento di sanità
1 self       l'ibrido contro se stesso
2 euristica  marnie_grimmsnarl_ex puro
3 gio_v1
4 gio_v2
```

Si sale quando la win rate sulla finestra supera `--promote-at`, **oppure**
quando scade `--rung-patience` (default 10 iterazioni). La pazienza non è
pigrizia: i dati di un gradino più duro valgono più di altre 50 iterazioni
contro un avversario già saturo.

Il gradino `self` è un caso a parte: la win rate è **50% per costruzione**
(stessa policy, stesso mazzo, lati alternati), quindi qualsiasi soglia lo
bloccherebbe per sempre. Si passa oltre appena la finestra è piena.

Il gradino 2 è il più informativo di tutti: è l'unico confronto in cui l'unica
variabile è la rete, perché euristica, pesi e mazzo sono identici sui due lati.

### Reward

Ritorno Monte Carlo con potential-based shaping su Φ. Con γ=1 la somma degli
shaping lungo l'episodio **telescopa**, quindi non serve nessuna ricorsione
all'indietro:

```
G_t = esito + scale · (Φ_finale − Φ_t)
```

È la stessa quantità che `teo_2` calcola con TD(λ=1), scritta in chiuso — e non
c'è nessun λ da sbagliare. (Il bug che aveva reso `teo_2` *più debole* dei pesi
casuali era esattamente λ<1 su episodi da 150-250 decisioni: qui il problema
non può ripresentarsi.)

### Esplorazione

`--explore-eps` (default 0.25) gioca un candidato a caso **dentro la banda di
ambivalenza**. Serve a diversificare i dati: a 0, due agenti deterministici
rigiocherebbero quasi la stessa partita e la rete vedrebbe sempre le stesse
posizioni. È sicura perché i candidati sono, per costruzione, mosse che
l'euristica giudica equivalenti — non è rumore sulla policy, è rumore su un
pareggio. In partita vale sempre 0.

### Niente multiprocessing, di proposito

`teo_2` ha un pool persistente di worker perché una sua partita di self-play
costa minuti (64 simulazioni MCTS × 7.2M parametri per decisione). Qui una
partita costa **~0.25 s**: 200 partite sono 50 secondi su un core. Il
parallelismo aggiungerebbe la complessità di `spawn`, la ricarica dei pesi via
file e ~0.7 GB di RAM per worker per comprimere un minuto. Se un giorno servisse,
il pattern da copiare è in `teo_2/train_selfplay.py`.

### Cosa guardare nei log

```
iter 7 [euristica] wr=54% (finestra 54% su 200) buffer=12040 phi_w=0.13
       v=0.0812 phi=0.0121 corr=+0.61 pred_std=0.284
       | ricerca 35% delle scelte, cambia 21% | turni 13 [58s]
```

- **`corr`** — correlazione tra V predetto e target. È la metrica di qualità,
  non la loss.
- **`pred_std`** — dispersione delle predizioni. Sotto 0.05 la value head sta
  predicendo quasi una costante, ed è invisibile guardando solo la loss che
  scende: una loss bassa su target poco dispersi non significa niente. Il log
  lo segnala esplicitamente.
- **`cambia M%`** — quanto spesso la rete ribalta la prima scelta
  dell'euristica. A 0% la rete non sta facendo niente; sopra ~40% la banda è
  probabilmente troppo larga.
- **`wr` sul gradino 2** — l'unica misura che isola l'apporto della rete.

## Degradazione graduale

Ogni livello è un agente completo, non un'emergenza:

1. checkpoint presente → ricerca con la value net;
2. checkpoint assente → stessa ricerca, valore di foglia = Φ;
3. torch assente / ricerca fallita / budget esaurito → **euristica pura (= v1)**;
4. qualunque eccezione → prima risposta legale.

Il livello 3 è il punto: il pavimento di questo agente è v1.

## Taratura

Il knob che regola *quanta* libertà ha la rete è la banda di ambivalenza,
modificabile senza ri-deploy per gli A/B:

| variabile | default | effetto |
|---|---|---|
| `MARNIE2_REL_MARGIN` | 0.15 | banda relativa al punteggio migliore |
| `MARNIE2_ABS_MARGIN` | 25 | banda minima assoluta |
| `MARNIE2_MAX_CANDIDATES` | 4 | tetto sui candidati valutati |
| `MARNIE2_CHECKPOINT` | — | checkpoint alternativo |

`MARNIE2_REL_MARGIN=0 MARNIE2_ABS_MARGIN=0` riproduce v1 esattamente (servono
**entrambe**: la banda è la somma dei due termini). È il modo giusto di
misurare il contributo della rete a parità di tutto il resto.

## Aspettative oneste

Baseline misurate su 100 partite, euristica pura:

| | win rate |
|---|---|
| `marnie_grimmsnarl_ex` vs `gio_v1` | 20% |
| `marnie_grimmsnarl_ex` vs `gio_v2` | 7% |

Va detto chiaramente: **questa architettura non è pensata per colmare un
divario del genere da sola.** Uno spread 20/80 non è un problema di spareggi tra
mosse equivalenti — è un problema strutturale (di piano di gioco, di pesi
euristici, o di mazzo). La ricerca sui candidati ambivalenti dà un guadagno
reale ma *incrementale*: il suo ruolo è prendere l'euristica migliore che
riesci a costruire e spremerne l'ultimo pezzo, in sicurezza.

Il posto dove cercare il resto è lo strato euristico, e `params.json` ha
almeno un valore che vale la pena rivedere prima di qualsiasi altra cosa:

- **`punk_up_count: 2.06`** — Punk Up cerca fino a **5** energie {D} nel mazzo,
  e con sole 10 energie in lista è l'unico motivo per cui il mazzo sta in piedi.
  Il tuner l'ha portato a 2. Se non è un artefatto del rumore, è mezzo motore
  spento.
- **`boss_ko_score: 22.8` sotto `boss_disruption_score: 18.1`** — i due valori
  sono a un soffio, mentre nel codice il primo dovrebbe dominare («Boss che
  tira fuori un bersaglio che uccido subito» vs «Boss generico»). Così la
  distinzione è di fatto collassata.
- **`prefer_go_first: 0`** — contro la logica dichiarata nel codice, che vuole
  negare all'avversario un turno di setup. Può essere giusto, ma va verificato.

Il modo corretto di verificarli è un A/B a `REL_MARGIN=0` (cioè su v1 puro), non
a occhio.
