# meta_analysis

Analisi del *metagame* Pokémon TCG a supporto della **PTCG AI Battle Challenge**.

Due notebook complementari:

| Notebook | Cosa analizza | Fonte dati |
|----------|---------------|------------|
| [`01_competition_meta_analysis.ipynb`](01_competition_meta_analysis.ipynb) | Il **card pool della competizione**: carte più forti/efficienti, distribuzione per tipo, Trainer/Energy, composizione dei mazzi sample, (opz.) leaderboard Kaggle | `doc/pokemon-tcg-ai-battle/EN_Card_Data.csv`, `deck.csv`, Kaggle API |
| [`02_limitlesstcg_standard_meta.ipynb`](02_limitlesstcg_standard_meta.ipynb) | Il **meta reale dei tornei ufficiali** (formato Standard): archetipi più usati, concentrazione del meta, core cards, carte più giocate | scraping live di [limitlesstcg.com](https://limitlesstcg.com) |
| [`03_competition_matches_analysis.ipynb`](03_competition_matches_analysis.ipynb) | Le **partite reali della competizione**: classifica agent (winrate/Elo), scontri diretti, durata partite, carte più presenti/giocate, profilo mazzi per agente | DB SQLite prodotto da `competiotion_scraper.py` (replay Kaggle) |

> **Perché due notebook.** I mazzi degli altri partecipanti su Kaggle non sono pubblici, quindi il
> "meta" della competizione si studia sul **card pool** (quali carte sono oggettivamente più forti).
> Il notebook 2 dà invece il quadro del gioco reale, utile come riferimento per capire quali
> archetipi/carte funzionano nei tornei ufficiali.

## Setup

```bash
# dalla root del repo
python -m pip install -r meta_analysis/requirements.txt
python -m ipykernel install --user --name ptcg-meta   # opzionale: kernel dedicato
```

Poi apri i notebook in VS Code / Jupyter e scegli il kernel Python del progetto.

## Esecuzione da riga di comando

```bash
cd meta_analysis
python -m nbconvert --to notebook --execute --inplace 01_competition_meta_analysis.ipynb
python -m nbconvert --to notebook --execute --inplace 02_limitlesstcg_standard_meta.ipynb
```

## Output

- `output/figures/` — grafici PNG (prefisso `01_` / `02_`)
- `output/data/`    — tabelle CSV riusabili (top carte, meta, mazzi sample, ecc.)
- `.cache/`         — cache delle pagine scaricate da limitlesstcg (non versionata)

## Scraper delle partite (`competiotion_scraper.py`)

Costruisce un **DB SQLite** (`ptcg_data/matches.db`) dai replay degli episodi Kaggle, letto poi
dal notebook 03. Pipeline **interamente pubblica, senza credenziali**:

`GetCompetition → GetLeaderboard(competitionId) → ListEpisodes(submissionId) → replay.json → parsing`

```bash
python competiotion_scraper.py check                 # verifica accesso API (nessun login)
python competiotion_scraper.py inspect <episode_id>  # stampa lo schema di UN replay reale
python competiotion_scraper.py scrape --teams 15 --episodes 8   # popola il DB
python competiotion_scraper.py demo                  # DB sintetico offline (se manca la rete)
```

Tabelle: `matches` (esito, agent, reward, durata), `match_decks` (composizione mazzi),
`cards_played` (sequenza giocate, dai log del replay).

> **Nessun `kaggle.json` necessario.** Gli endpoint usati sono quelli che consuma il frontend del
> sito Kaggle: sono pubblici (serve solo il token anti-CSRF di sessione, che lo script ottiene da
> solo). Verificato sul `competitionId=116727`. Lo **schema del replay è stato confermato su
> episodi reali** — i mazzi stanno in `steps[0][0].visualize[0].action` e le giocate nei
> `observation.logs`. I replay grezzi vengono salvati in `ptcg_data/replays/` (cache).

## Note tecniche

- **`meta_utils.py`** contiene tutta la logica riusabile (caricamento carte, scraping, cache, TLS).
  Ha una self-verification: `python meta_utils.py`.
- **TLS/proxy.** Sulla macchina di sviluppo un proxy rompe la verifica dei certificati con il bundle
  di default di Python; `meta_utils.enable_tls()` usa il trust store dell'OS via `truststore`
  (con fallback a `certifi` e, in ultima istanza, verifica disabilitata).
- **Leaderboard Kaggle** (Sezione 7 del notebook 1): opzionale, richiede il pacchetto `kaggle` e un
  token in `~/.kaggle/kaggle.json`. Senza credenziali il notebook resta eseguibile (stampa un avviso).
  L'estrazione delle *carte giocate nelle partite* dai replay è predisposta come passo successivo:
  lo schema del replay va letto da un episodio reale (con credenziali) prima di parsarlo.
- **Formato Expanded.** Nel notebook 2, imposta `FMT = "expanded"` nella cella di setup.
- **Cortesia verso il sito.** C'è un throttle tra le richieste e una cache su disco: non abbassare
  il throttle e non cancellare la cache senza motivo.
