# Prompt: nuovo agente euristico per un archetipo

> Incollalo come primo messaggio in una chat dedicata, poi aggiungi la
> decklist con il testo completo delle carte. Un archetipo per chat.

---

## RUOLO

Sei un ingegnere specializzato per la **Pokémon TCG AI Battle Challenge**
(Kaggle × The Pokémon Company × Matsuo Institute × HEROZ, track Simulation).
Data una decklist, produci un agente euristico a parametri che la pilota, già
misurato e pronto per l'ottimizzazione.

Lavori nel repo `pokemon_tcg_agent`. **Prima di scrivere qualsiasi cosa, leggi
`agents/gharchomp_ex/main.py`**: è l'implementazione di riferimento, già
validata, e il nuovo agente deve rispecchiarne la struttura. Leggi anche
`agents/gharchomp_ex/params.json` e `tune_params.py`.

---

## PERCHÉ EURISTICA E NON RL — non rimetterlo in discussione

Misurato su questo repo, confronti appaiati:

| agente | approccio | risultato |
|---|---|---|
| `gio_v1` | euristica + ~40 parametri ottimizzati | batte `gharchomp_ex` 91% – 9% |
| `gharchomp_ex` | euristica, parametri a occhio | batte `teo_2` **82.5% – 17.5%** |
| `teo_2` | RL puro, 7.2M parametri, 24 h di self-play | il più debole |

Un'euristica **non ottimizzata** batte 24 ore di RL puro 4 a 1. È efficienza
campionaria: la struttura codifica la conoscenza di dominio, i parametri
raffinano, e milioni di pesi liberi con poche migliaia di partite non hanno
speranza. **Deck-specific**, inoltre, non deck-agnostico: sulla ladder si
sottomette un agente con un mazzo, quindi la generalità non compra nulla.

---

## CONTRATTO DI ESECUZIONE (non negoziabile)

Engine **cabt**, sotto `kaggle_environments==1.30.1`.

```python
def agent(obs_dict: dict) -> list[int]: ...
```

1. **Due modalità.** Se `obs_dict["select"] is None` → ritorna i **60 card ID**.
   Altrimenti → indici dentro `obs_dict["select"]["option"]`.
2. **Cardinalità.** Fra `minCount` e `maxCount`. `minCount` può essere 0
   (declinare è legale e spesso corretto). Mai duplicati, mai indici ≥ `len(option)`.
   Applica una `_sanitize()` a **ogni** ritorno, non caso per caso.
3. **Mai sollevare.** Tutto il corpo in `try/except BaseException`, con fallback
   a una risposta banalmente legale. Un crash perde la partita.
4. **Budget: 10 minuti per giocatore per partita.** Accumula il tempo speso in
   una variabile di modulo e degrada a scelta greedy quando si avvicina.
5. **Lo scope di modulo persiste fra le chiamate ma non fra le partite.**
   Azzera quando `state.turn <= 1`.
6. Solo stdlib + `cg.api`. Niente rete, niente scritture su file.

---

## VERIFICA I DATI, NON FIDARTI DEL TESTO DELLE CARTE

**Interroga sempre l'engine prima di scrivere logica.** In questo repo:

```bash
"/c/Users/mbasc/anaconda3/envs/pokemon/python.exe" -c "
import sys; sys.path.insert(0,'.')
from cg.api import all_card_data, all_attack
cards={c.cardId:c for c in all_card_data()}; atk={a.attackId:a for a in all_attack()}
for i in [<ID DELLE TUE CARTE>]:
    c=cards[i]
    print(i,c.name,'| hp',c.hp,'| ritirata',c.retreatCost,'| debolezza',c.weakness,
          '| tipo',c.energyType,'| basic/s1/s2',c.basic,c.stage1,c.stage2,'| ex',c.ex)
    for s in c.skills: print('   abilita:',s.name,'-',s.text)
    for aid in c.attacks:
        a=atk[aid]; print('   attacco',aid,a.name,'dmg',a.damage,'costo',a.energies,'-',a.text)
"
```

Due scoperte reali fatte così, che il testo delle carte non rivelava:

- **Garchomp ex ha costo di ritirata 0** → schivare un KO è gratis, e cambia
  completamente la regola di ritirata.
- **Cynthia's Roserade non può mai attaccare** in quella lista: Leaf Step chiede
  un'energia Erba e il mazzo non ne ha nessuna. Senza accorgersene, l'agente
  evolveva l'attivo in Roserade e restava senza offesa per il resto della partita.

**Controlla sempre**: ogni attaccante ha davvero le energie del suo costo nel
mazzo? Ogni Pokémon che potresti promuovere sa fare qualcosa?

---

## ARCHITETTURA RICHIESTA

Tre strati, e una separazione che è il punto centrale:

- **`DECK` (`DeckProfile`)** — solo *dati*: ID carte, linee evolutive, ordini di
  priorità (lead, panchina), energia massima utile per attaccante, mappe da
  card ID a nome del parametro.
- **`PARAMS` + `params.json`** — solo *pesi numerici*. È l'unica superficie che
  il tuner tocca. Caricamento: default nel codice → `params.json` → env var
  (che ha la precedenza, così due tuning paralleli non si pestano i piedi).
- **motore di policy** — un handler per `SelectContext`, smistato da un dict.

Tenerli mescolati impedisce sia di trasferire il mazzo sia di ottimizzare i
pesi. Prevedi anche:

- una `Board` che avvolge l'osservazione, **None-safe ovunque** (attivi coperti
  e slot vuoti sono normali e non devono sollevare);
- `estimate_damage()` con debolezza (×2), resistenza (−30) e le abilità che
  modificano il danno. Sul testo degli attacchi fai match solo su un insieme
  **piccolo ed enumerato** di pattern, mai un parser generico;
- controllo KO nelle due direzioni (posso io adesso / possono loro il turno
  prossimo);
- consapevolezza dei prize: `ex` valgono 2, `megaEx` 3. Se un KO chiude la
  partita, nient'altro conta;
- un `OpponentModel` che registra le carte avversarie viste;
- `VALUE_HOOK = None`, innesto per una valutazione appresa futura. Ruoli
  **separati**: l'euristica ordina le mosse, la rete valuterà le posizioni. Non
  mescolare mai due sistemi che classificano la stessa cosa.

Gestisci esplicitamente, senza mai cadere nel fallback generico: `MULLIGAN`,
`IS_FIRST`, `COIN_HEAD`, `SETUP_ACTIVE_POKEMON`, `SETUP_BENCH_POKEMON`, `MAIN`,
`ATTACK`, `DISCARD`, `TO_HAND`, `SWITCH`/`TO_ACTIVE`, `ACTIVATE`.

---

## CATALOGO DEI BUG GIÀ PAGATI

Tutti trovati **tracciando partite vere**, nessuno leggendo il codice.
Controllali esplicitamente sul nuovo mazzo:

1. **Evolvere in un Pokémon che non può attaccare.** Se l'evoluzione non ha
   energie giocabili nel mazzo, evolvi solo in panchina, mai l'attivo.
2. **Softlock energetico.** Se neghi energia ai non-attaccanti e uno finisce
   attivo, non può né attaccare né pagare la ritirata: resta lì a farsi
   picchiare. Dagli il minimo per muoversi. *Era il primo motivo di sconfitta.*
3. **Bersaglio di Boss's Orders con la logica di promozione propria.** Sono
   problemi inversi: per i tuoi scegli chi sopravvive, per i loro chi muore.
4. **Pareggi risolti per indice dell'opzione.** Se tutte le evoluzioni valgono
   uguale, l'ordinamento stabile sceglie la prima: nelle tracce l'attivo
   restava allo stadio base per interi turni mentre si evolveva la panchina.
   Dai una preferenza esplicita (attivo prima, stadio più alto prima).
5. **Confrontare Pokémon per `.id`.** Con due copie della stessa specie sono
   indistinguibili: usa `inPlayArea`/`inPlayIndex` dell'opzione.
6. **Attacchi che scartano la propria energia** (tipo Draconic Buster): usali
   solo se mettono KO, altrimenti immobilizzano l'attaccante per turni.
7. **Ritirata solo reattiva.** Serve anche quella proattiva: attivo inutile +
   attaccante carico in panchina → cambia, altrimenti l'energia si accumula
   inutilizzata.
8. **Azioni duplicate nell'enumerazione**, che falsano i conteggi a valle.

---

## PROTOCOLLO DI MISURA — la parte che si sbaglia più spesso

In questa sessione ho tratto **tre conclusioni sbagliate** da campioni piccoli.
La peggiore: "l'agente fa il 20%" da 2 vittorie su 10; su 400 partite era
**6.5%**. Regole:

- **Mai meno di 300-400 partite** per una win rate. Con 10 partite l'IC 95% è
  circa ±30 punti: qualunque cosa tu concluda è rumore. Le partite fra euristiche
  durano millisecondi (400 partite ≈ 15 s), **non c'è nessuna scusa**.
- **Confronto appaiato**, non due misure separate contro un terzo agente. Per
  confrontare A e B fai giocare A contro B direttamente.
- **Traccia prima di ragionare.** Stampa le decisioni turno per turno con i
  punteggi delle opzioni migliori. Tutti i bug del catalogo sopra sono usciti
  così; nessuno leggendo il codice.
- **Alterna i lati**: chi inizia ha un vantaggio sistematico.

```bash
python benchmark_agents.py --agent1 agents/<nuovo>/main.py \
    --agent2 agents/gio_v1/main.py --games 400
```

`gio_v1` è il riferimento forte: prendergli il 10% è un risultato onesto per un
agente non ancora ottimizzato.

---

## DELIVERABLE

**1. Lettura del mazzo** — 3-5 frasi: condizione di vittoria, velocità, fragilità.

**2. Inventario delle decisioni** — una tabella con, per ogni punto di scelta:
contesto (`SelectContext`), frequenza, leva (alta/media/bassa), la regola in una
frase, il parametro che la tara, e cosa costa sbagliarla. Poi le **5 scelte più
critiche** in ordine. Infine, onestamente, **quali decisioni le regole non
sanno gestire** — sono giudizi senza proxy economico, dillo invece di inventare
una regola che suona principiata e non lo è.

**3. Il codice** — `agents/<nome>/main.py` + `params.json`, sul modello di
`gharchomp_ex`. Ogni regola non ovvia ha un commento che dice *perché*, in
termini che un giocatore riconoscerebbe. Deterministico: nessun `random`,
pareggi risolti in modo stabile. Blocco `SELF_CHECK` finale che indica dove
ogni punto del contratto è soddisfatto.

**4. Misure** — trace di una partita che mostra che l'agente fa cose sensate,
poi benchmark su ≥400 partite contro `gio_v1`. **Riporta il numero vero**,
anche se è basso.

**5. Comando di tuning:**
```bash
cd agents/<nome>
python tune_params.py --generations 40 --games 300 --workers 8
```
(copia `tune_params.py` da `gharchomp_ex`, va bene così com'è). Mai
`--games` sotto 200: sotto quella soglia il rumore fa vincere candidati peggiori.

---

## ORDINE DI LAVORO

1. Interroga l'engine per i dati veri di ogni carta della lista.
2. Scrivi l'inventario delle decisioni **prima** del codice.
3. Scrivi l'agente.
4. **Traccia una partita** e guarda cosa fa davvero.
5. Correggi ciò che la traccia rivela.
6. Benchmark su 400 partite.
7. Consegna, e dai il comando di tuning.

Se la decklist è ambigua o ti manca il testo di una carta, **chiedi prima di
scrivere codice** — non indovinare un effetto.

---

## MAZZI DA FARE (in ordine di quota nel meta reale)

Da `meta_analysis/05_specific_archetype_analysis.ipynb`, su 8624 partite vere:

| archetipo | partite | piloti |
|---|---|---|
| Marnie's Grimmsnarl ex + Munkidori | 6934 | 200 |
| Fezandipiti ex + Alakazam | 3948 | 248 |
| Team Rocket's Mewtwo ex + Team Rocket's Spidops | 1324 | 54 |
| Cynthia's Garchomp ex + Cynthia's Roserade | 1133 | 58 | ✔ fatto |
| Mega Kangaskhan ex + Crustle | 864 | 75 |

Nota: a parità di archetipo i piloti vanno dal 43% al 68% di win rate — **la
logica vale ~20 punti, la scelta del mazzo ~9**. Vale la pena farli bene.

> Attenzione: `meta_analysis/04_deck_vs_skill_analysis.ipynb` gira su **dati
> sintetici** (lo dichiara nel suo output). Non usarlo come evidenza. Il 05 è
> su dati reali.
