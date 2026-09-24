# Submission — Giorgio | ~900 Elo

## Deck: Mega Lucario ex (Fighting)

### Composition (60 cards)
| Card | Qty |
|---|---|
| Basic Fighting Energy | 14 |
| Lunatone | 2 |
| Solrock | 4 |
| Riolu | 4 |
| Mega Lucario ex | 4 |
| Dusk Ball | 4 |
| Switch | 3 |
| Premium Power Pro | 4 |
| Fighting Gong | 4 |
| Poke Pad | 4 |
| Hero Cape | 1 |
| Boss Orders | 3 |
| Carmine | 4 |
| Lillie — Determination | 4 |
| Gravity Mountain | 2 |

## Strategy Overview

Agent basato su **euristiche ponderate** per il deck Mega Lucario ex. I pesi sono stati ottimizzati tramite tuning locale e possono essere sovrascritti via `params.json` o la variabile d'ambiente `MYBOT_PARAMS`.

### Punti chiave

- **PPP-aware KO planning**: Premium Power Pro viene giocato solo quando trasforma un attacco non-KO in un KO, massimizzando l'efficienza dei boost.
- **Cosmic Beam weakness fix**: il danno di Cosmic Beam non è influenzato da debolezza/resistenza (corretto nel calcolo).
- **Carmine guard**: evita di giocare Carmine quando la mano contiene troppe carte preziose (PPP, Boss Orders, Mega Lucario ex, Hero Cape, Switch), riducendo il discard involontario.
- **Opponent-prize aware Mega exposure**: penalizza l'uso di "Mega Brave" quando l'avversario ha pochi premi rimanenti e un KO del Mega ne darebbe 3 (mega_guard).
- **Setup-bench preferences**: preferenze esplicite per la disposizione iniziale dei Pokémon (Solrock/Riolu active, Lunatone/Solrock bench).
- **Gravity Mountain proactive**: gioca Gravity Mountain quando l'avversario ha uno Stage 2 in campo (es. Dragapult ex 320→290 HP: Mega Brave 270 + un PPP lo KO).
- **Early-game denial bonus**: bonus per KO di Pokémon evolutivi non-ex nei primi turni, per negare il setup avversario.

### Parametri ottimizzati (params.json)

I parametri sono stati tuningati localmente e includono pesi per:
- Scoring degli attacchi (PPP play score, boss score, Carmine score, Lillie score)
- Preferenze di switch/retreat
- Priorità di promozione (Mega Lucario ex > Hariyama > Makuhita > Solrock > Riolu)
- Preferenze di pesca (to-hand priorities per ogni carta)
- Energy attachment scoring
- Setup iniziale e bench preferences

### Performance

- **Elo**: ~900
- Winrate vs Random: alto
- Efficace contro deck con Stage 2 grazie a Gravity Mountain + Mega Brave synergy
