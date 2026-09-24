# Agente AI per il Pokémon TCG

Il nostro agente per la **PTCG AI Battle Challenge** (Kaggle × The Pokémon Company × Matsuo Lab × HEROZ),
con gli strumenti per simulare le partite in locale e rivederle nel viewer ufficiale dei replay.

La competizione viene valutata su una ladder Elo automatica: la ladder è l'unica vera misura della
forza dell'agente. Tutto ciò che trovi qui serve allo **sviluppo locale**: scrivere l'agente,
testare i mazzi e fare debug visivo delle linee di gioco.

---

## Cosa contiene questo repository (e cosa no)

Versionato in git:

- `main.py` — l'agente: `agent(obs_dict) -> list[int]`.
- `deck.csv` — la nostra lista di 60 carte.
- `battle_test.py` — esegue una partita in self-play ed esporta un replay per il viewer.
- `setup-viewer.sh` — installa il viewer dei replay in locale.
- `requirements.txt`, questo `README`.

**Volutamente NON committato** (ciascun membro del team lo recupera in locale):

- **L'engine della competizione** (il package `cg/` e il suo `.so`). È la parte proprietaria e
  riservata della competizione: scaricala dalla pagina Kaggle della competizione e non
  ridistribuirla qui.
- **Il viewer** (`cabt-viewer/`). È un monorepo di sviluppo da diverse centinaia di MB;
  `setup-viewer.sh` lo recupera a un commit fissato. È in gitignore.
- `.venv/`, `__pycache__/`, ecc.

> **Il file `.gitignore` dovrebbe contenere almeno:**
> ```
> .venv/
> __pycache__/
> *.pyc
> /cabt-viewer/
> /cg/
> *.so
> ```

---

## 1. Prerequisiti

Servono **git**, **Python 3.11+** e **Node.js 18+** (il viewer è un'app Node/Vite).
Per la parte di simulazione è consigliato Python 3.11, perché è più vicino al runtime di Kaggle
rispetto alle release molto recenti: il codice che gira sulla ladder viene eseguito
dall'interprete di Kaggle, quindi evita di affidarti a sintassi troppo all'avanguardia.

### macOS

```bash
# Homebrew (https://brew.sh) se non ce l'hai, poi:
brew install git node python@3.11
corepack enable            # incluso in Node; fornisce la versione di pnpm fissata
```

### Windows

Due opzioni: scegline una e usala in modo coerente.

- **Git Bash (la più semplice):** installa [Git for Windows](https://git-scm.com/download/win)
  (include Git Bash) e [Node.js LTS](https://nodejs.org). Poi esegui lo script bash da un terminale
  Git Bash. In un terminale con privilegi elevati:
  ```powershell
  winget install Git.Git OpenJS.NodeJS.LTS Python.Python.3.11
  corepack enable
  ```
- **WSL (se lo usi già):** installa Ubuntu tramite WSL e segui i passi per **Linux** al suo interno.
  Avvia il viewer lì e apri `http://localhost:5173` nel browser di Windows.

Su Windows `corepack enable` potrebbe richiedere un terminale da **Amministratore**. Se non
funziona, ripiega su `npm install -g pnpm@9.15.3`.

### Linux

```bash
# Fedora
sudo dnf install -y git nodejs npm python3 python3-virtualenv

# Debian / Ubuntu  (il node di apt è spesso troppo vecchio: usa NodeSource per Node 18+)
sudo apt-get install -y git python3 python3-venv
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

corepack enable
```

Verifica: `node -v` deve stampare **v18** o superiore.

---

## 2. Recuperare l'engine della competizione

Dalla **pagina Kaggle della competizione** scarica i materiali iniziali (starter materials) e metti
il package `cg/` dell'engine nella **root del repository**, in modo che
`from cg.api import ...` (usato in `main.py`) venga risolto:

```
pokemon_tcg_agent/
├── main.py
├── cg/            <-- dagli starter materials di Kaggle (NON committato)
│   ├── api.py
│   └── ...
└── ...
```

Non committare `cg/` né il `.so`: sono già in gitignore qui sopra.

---

## 3. Ambiente Python

```bash
python3.11 -m venv .venv
# attivalo:
source .venv/bin/activate          # macOS / Linux / Git Bash
# .venv\Scripts\Activate.ps1       # Windows PowerShell

pip install -U pip
pip install -r requirements.txt
```

`requirements.txt` blocca **`kaggle-environments==1.30.1`**, la stessa versione che gira sulla
ladder. Tienila bloccata; non aggiornarla con leggerezza.

Test rapido (esegue una partita in self-play, senza bisogno del viewer):

```bash
python battle_test.py
```

Dovresti vedere stampati i reward e una riga `replay -> ...`.

---

## 4. Installare il viewer dei replay

Il viewer ufficiale mostra le partite esattamente come le visualizza Kaggle. Installalo una volta:

```bash
./setup-viewer.sh                  # macOS / Linux / Git Bash / WSL
```

Lo script clona il viewer in `./cabt-viewer` (commit fissato, clone shallow) ed esegue
`pnpm install`. L'installazione è la parte lenta — scarica un toolchain React/Vite — ma è una
tantum.

> Puoi mettere il viewer altrove con `VIEWER_DIR=/percorso/del/viewer ./setup-viewer.sh`. In tal
> caso imposta `CABT_REPLAY_PATH` di conseguenza (vedi sotto).

---

## 5. Il ciclo "guarda una partita"

`battle_test.py` scrive il JSON del replay di ogni partita nel file servito dal viewer. Di default è
`./cabt-viewer/.../replays/test-replay.json`; cambia la destinazione con `CABT_REPLAY_PATH` se il
tuo viewer si trova altrove.

**Terminale A — avvia il viewer una volta e lascialo in esecuzione:**

```bash
cd cabt-viewer/kaggle_environments/envs/cabt/visualizer/default
pnpm dev-with-replay
# apri l'URL stampato, es. http://localhost:5173
```

(`dev-with-replay` è multipiattaforma: gestisce per te la variabile d'ambiente del file di replay su
ogni sistema operativo.)

**Terminale B — esegui le partite:**

```bash
python battle_test.py     # sovrascrive il replay
```

Poi **aggiorna la scheda del browser** per vedere l'ultima partita. È tutto qui il ciclo:
esegui → aggiorna.

---

## Capire il viewer (importante)

Il viewer locale è una **canvas volutamente essenziale**: un tabellone nero in cui ogni carta è
disegnata come un rettangolo etichettato (nome, `HP`, energie attaccate). È una scelta di design,
non un bug di rendering: è sufficiente per seguire lo stato del tabellone e fare debug di una linea
di gioco.

Per il **replay grafico completo** (artwork reale delle carte, anteprima grande della carta), clicca
uno dei pulsanti **"Open Visualizer"** sulla canvas. Questo invia (POST) il replay al visualizer
ospitato da HEROZ (`ptcgvis.heroz.jp`) e lo apre in una nuova scheda.

> **Problema noto:** nella scheda di HEROZ alcune immagini delle carte possono apparire come
> rettangoli neri (compresa l'anteprima XL). Sembra un problema lato server di HEROZ, non qualcosa
> nel nostro setup, ed è puramente estetico. La canvas essenziale mostra comunque lo stato completo
> del tabellone.

---

## Risoluzione dei problemi

- **`pnpm: command not found`** — esegui `corepack enable` (terminale da Amministratore su Windows),
  oppure `npm install -g pnpm@9.15.3`.
- **`Node ... is too old`** — installa Node 18+ (macOS: `brew upgrade node`; Debian/Ubuntu: usa il
  passo NodeSource qui sopra; Windows: reinstalla la LTS).
- **Il viewer mostra "Failed to load replay file"** — non hai ancora eseguito una partita, oppure
  `CABT_REPLAY_PATH` punta a un file diverso da quello servito dal dev server. Esegui
  `python battle_test.py` e verifica che il percorso `replay -> ...` stampato coincida con
  `.../replays/test-replay.json` nella cartella del tuo viewer.
- **Porta 5173 già in uso** — `VITE_PORT=5174 pnpm dev-with-replay`, poi apri quella porta.
- **Il `git fetch` del commit fissato fallisce** — il tuo git potrebbe essere molto vecchio:
  aggiornalo, oppure riesegui lo script (riprova il fetch).

---

## Aggiornare il viewer

Il viewer è bloccato a un commit in `setup-viewer.sh` (`PINNED_COMMIT`). Per spostare tutto il team
a una versione più recente, aggiorna quel SHA, committa la modifica e fai rieseguire
`./setup-viewer.sh` ai membri del team.
