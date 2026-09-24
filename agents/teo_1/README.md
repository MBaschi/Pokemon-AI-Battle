# teo_1

Agente basato su transformer (`encoding.MyModel`, value + policy). `main.py`
è l'agente che gioca davvero (`agent(obs_dict) -> list[int]`): decisioni
forzate/ovvie eseguite subito, scelte singole tra alternative risolte con una
ricerca action-value a 1-ply (simula ogni opzione, chiede al value network
quale stato risultante è migliore), multi-select rare gestite con
un'euristica generica di fallback. Il resto del modulo allena il value
network in due modi alternativi: self-play MCTS oppure apprendimento
supervisionato su partite reali scaricate.

## File

| file | ruolo |
|---|---|
| `encoding.py` | card encoding, `MyModel`, batching — condiviso da tutto il resto |
| `main.py` | **agente live**: orchestrazione + ricerca action-value col value network |
| `deck.csv` | mazzo da 60 carte usato da `main.py` (e dal training self-play) |
| `train_selfplay.py` | training originale: self-play + MCTS |
| `build_replay_dataset.py` | costruisce un dataset di stati dai replay reali (`meta_analysis/ptcg_data/replays` + `matches.db`) |
| `train_value_function_from_replays.py` | allena solo il ramo value su quel dataset |

## Requisito

Serve il package `cg/` (engine della competizione) alla root del repo — vedi
il README principale. Senza, nessuno di questi script parte.

## Comandi

```bash
cd agents/teo_1

# training originale (self-play MCTS)
python train_selfplay.py

# alternativa: value head da replay reali
python build_replay_dataset.py --max-games 20   # smoke test veloce
python build_replay_dataset.py                  # dataset completo (485 partite)
python train_value_function_from_replays.py

# gioca una partita con l'agente (usa out/value_from_replays.pth se esiste,
# altrimenti pesi non allenati con un warning; TEO1_CHECKPOINT per un altro path)
python -c "from main import agent"  # entry point per l'harness della competizione
```

`data/` e `out/` sono generati (gitignored) — rigenerabili in qualsiasi
momento con i comandi sopra.
