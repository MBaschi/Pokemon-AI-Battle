# State Evaluator — classificatore "chi vincerà?"

Primo modello **data-driven** per la valutazione di uno stato di gioco della
competizione *Pokémon TCG AI Battle*. Dato lo stato della partita (carte in
campo, mano, scarti, prize, energie…) stima la **probabilità che il giocatore
di riferimento vinca**. È una *funzione di valutazione appresa*: lo stesso
mattone che, negli scacchi/Go, precedeva l'RL.

## A cosa serve (l'approccio è utile?)

Sì, ed è un buon primo passo. Usi concreti:

- **Ranking delle mosse (1-ply)**: per ogni azione legale, simula lo stato
  risultante e scegli quella con probabilità di vittoria stimata più alta.
- **Valutazione delle foglie in una ricerca** (MCTS / minimax poco profondo):
  sostituisce i rollout casuali con una stima appresa.
- **Supporto a bot euristici**: tie-breaker quando l'euristica è indecisa.
- **Base per l'RL**: è già di fatto un *critic* (value function) riutilizzabile.

Ti aspettavi "logiche molto semplici": è proprio quello che emerge (vedi i pesi
della logistica sotto). È il valore di partire da un modello interpretabile.

### Nota sull'autoencoder

Nella richiesta ipotizzavi un autoencoder per l'encoding. Per una **classificazione
supervisionata** su feature strutturate lo sconsiglio come pezzo centrale della
v1: un modello diretto (regressione logistica / gradient boosting) è più forte,
molto più interpretabile e meno affamato di dati. L'AE serve soprattutto a
**comprimere il vettore sparso "quali carte sono visibili"** o per pre-training
non supervisionato.

Compromesso adottato: le feature ingegnerizzate sono il cuore interpretabile; il
"bag of cards" alto-dimensionale viene compresso da un **autoencoder lineare
(PCA via SVD)** — zero dipendenze, gira ovunque. In [`encoding.py`](encoding.py)
c'è anche `TorchAutoencoder` (non lineare) come upgrade opzionale se installi
`torch`.

## Da dove vengono i dati

**Non** dalla `matches.db`: quella contiene solo la *sequenza* delle giocate, non
lo stato del board. Lo stato completo è nei **replay JSON**
(`meta_analysis/ptcg_data/replays/*.json`, già scaricati dallo scraper —
~8.800 partite), in `steps[0][0].visualize[k].current`: per ogni Pokémon in
campo `hp/maxHp`, `energies`, `tools`, `preEvolution`, `id`; più `hand`, `deck`,
`discard`, `prize` di entrambi i giocatori.

### Informazione realistica vs oracle

Nel replay si vede *tutto* (anche mano e mazzo avversari). Un bot vero **no**.
Per default estraiamo feature **`realistic`**: solo ciò che un bot osserva —
la propria mano, i due board, gli scarti (pubblici) e i *conteggi* di
mano/mazzo avversari. Così il modello è davvero utilizzabile in partita. La
modalità **`oracle`** (`--info-mode oracle`) aggiunge l'informazione nascosta:
utile come limite superiore per misurare quanto pesa non vedere le carte
avversarie.

## Struttura

| file | ruolo |
|---|---|
| [`state_features.py`](state_features.py) | parsing del replay → feature scalari + "bag of cards" da una prospettiva |
| [`build_dataset.py`](build_dataset.py) | campiona stati dai replay, li etichetta col vincitore, salva `.npz` + `.meta.json` |
| [`encoding.py`](encoding.py) | standardizzazione + encoder del bag of cards (PCA lineare; opz. AE torch) |
| [`model.py`](model.py) | regressione logistica e metriche, in puro numpy |
| [`train_classifier.py`](train_classifier.py) | split per-partita, training, metriche per fase, ablation, salvataggio modello |

## Uso

```bash
cd agents/state_evaluator

# 1) costruisci il dataset (tutte le partite disponibili)
python build_dataset.py --per-game 6 --vocab 300

#    smoke test veloce su poche partite:
python build_dataset.py --max-games 400

# 2) allena e valuta
python train_classifier.py --latent 16
```

Output: `data/state_samples.npz` + `.meta.json` (dataset) e `data/win_model.npz`
(modello riusabile da un bot). Dipendenze: solo `numpy`. Se installi
`scikit-learn`, il training aggiunge in automatico un **HistGradientBoosting**
(di norma più accurato della logistica).

## Come leggere i risultati (accortezze importanti)

Il codice affronta i punti dove questi modelli ingannano:

1. **Split per partita**, non per sample: snapshot dello stesso game sono
   correlati e con la stessa label → uno split casuale gonfierebbe l'accuracy.
2. **Metriche per fase (early / mid / late)**: a fine partita "chi ha meno prize
   vince" è banale. Il segnale *interessante* è predire da stati early/mid.
   Guarda l'AUC per fase, non solo quella complessiva.
3. **Simmetria A/B**: ogni stato genera due sample (prospettiva mia e avversaria)
   → label perfettamente bilanciate (0.50) e modello simmetrico.
4. **Ablation senza prize** (`diff_prizes_remaining` ecc. rimosse): mostra quanto
   il modello dipende dal punteggio vs dalle logiche di board.

Esempio reale (smoke test, **solo 400 partite** → tanti dati mancano):

```
[Logistica — test]  accuracy=0.674  auc=0.741
    fase early: acc=0.507  auc=0.482   <- ~casuale: early-game quasi indeciso
    fase mid  : acc=0.647  auc=0.695
    fase late : acc=0.802  auc=0.886   <- facile: la partita è quasi decisa
Logiche apprese:  diff_prizes_remaining (-)  diff_discard_count (-)  diff_total_energy (+) ...
Senza prize:      accuracy=0.654       <- cala poco: conta anche il board
```

Con l'intero corpus (~8.800 partite) i numeri, la calibrazione e soprattutto la
fase early migliorano nettamente.

## Integrare il modello in un bot

`data/win_model.npz` contiene pesi della logistica, scaler, componenti PCA e
vocabolario carte. Per valutare uno stato: costruisci le feature con
`state_features.state_to_features(current, me, card_meta(), info_mode)`,
applica scaler + PCA come in [`encoding.py`](encoding.py), poi
`sigmoid(w·x + b)`. (Un piccolo `predict.py` di comodo è un prossimo passo
naturale.)

⚠️ Le feature qui sono estratte dal *replay*. Per l'uso live vanno mappate
sull'**osservazione che il bot riceve dall'engine** (`agents/*/cg/`): stessi
campi, ma verifica i nomi/formati prima di fidarti delle predizioni in partita.

## Limiti e prossimi passi

- **Più dati**: gira `build_dataset.py` su tutte le partite; valuta più
  `--per-game` (attenzione alla correlazione intra-partita).
- **Modello più forte**: `pip install scikit-learn` (HistGB) o `torch` (MLP/AE
  non lineare).
- **Calibrazione**: Platt/isotonic sul validation, se ti serve una probabilità
  affidabile e non solo un ranking.
- **Target alternativi**: invece del vincitore finale (segnale lontano e
  rumoroso), predire il *vantaggio in prize a k turni* o la *bontà della mossa*
  dà un segnale più denso e più utile a un bot.
- **Allineamento all'engine**: definire le feature direttamente
  dall'osservazione `cg/` per un uso live senza rischio di disallineamento.
