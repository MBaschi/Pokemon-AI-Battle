"""Ricerca a *completamento di turno* sui soli candidati ambivalenti.

L'idea in una riga: **l'euristica decide sempre; la rete interviene solo quando
l'euristica non sa scegliere.**

    1. l'euristica assegna un punteggio a ogni opzione legale (heuristic.py);
    2. si prendono le opzioni entro una banda dal massimo -- le mosse che
       l'euristica giudica *equivalenti*. Se ne resta una sola, si gioca quella
       e non si spende niente (e' il caso piu' frequente);
    3. per ciascun candidato si applica la mossa nel motore determinizzato e si
       **finisce il turno con l'euristica stessa**, fino a quando il controllo
       passa all'avversario;
    4. si valuta lo stato di confine con la value net e si tiene il migliore.

Tre proprieta' che rendono la cosa sicura, in ordine di importanza:

**Il pavimento e' l'euristica.** La rete non puo' proporre una mossa che
l'euristica non abbia gia' giudicato tra le migliori: non esiste il caso "la
policy debole scavalca l'euristica forte". Nel caso peggiore -- rete non
allenata, cioe' rumore -- l'agente sceglie a caso *dentro un insieme di mosse
equivalenti secondo l'euristica*, e la perdita e' limitata dall'ampiezza della
banda. Con la banda a zero l'agente e' esattamente v1.

**Si confrontano stati confrontabili.** Valutare la posizione subito dopo una
singola opzione di MAIN non dice quasi niente: un turno qui contiene 5-15
decisioni, e "gioco Poffin" e "attacco" portano a stati che non stanno sulla
stessa scala. Completare il turno porta tutti i candidati allo stesso confine
-- il momento in cui passa il tratto -- che e' anche l'unico punto in cui la
domanda "chi sta meglio?" ha una risposta ben definita. E' anche esattamente la
distribuzione su cui la rete viene allenata (train.py), quindi non c'e' scarto
tra come si allena e come viene interrogata.

**Il completamento usa l'euristica, non la rete.** Il rollout e' deterministico
e costa microsecondi per decisione; il costo vero sono le chiamate al motore.

Approssimazioni dichiarate:

  - *determinizzazione singola*. `search_begin` pretende un'ipotesi completa
    sull'informazione nascosta, quindi la ricerca lavora su un solo campione
    del mazzo e dei prize (strategy fusion). Mitigata usando le carte gia'
    osservate dell'avversario, non eliminata.
  - se durante il completamento tocca all'avversario decidere qualcosa (una
    promozione dopo un KO), la scelta la fa la *nostra* euristica sul *suo*
    mazzo. E' un modello d'avversario povero, ma agisce identicamente su tutti
    i candidati, quindi si cancella in gran parte nel confronto.
"""

import time

import numpy as np

import cgpath  # noqa: F401  -- deve precedere qualsiasi import di cg.*

from cg.api import CardType, search_begin, search_end, search_step

import heuristic
from cards import ATTACK_TABLE, CARD_TABLE
from encoding import encode_state
from model import evaluate_batch
from reward import phi as phi_fn

# Riempitivi per le zone avversarie ignote: servono solo a soddisfare i vincoli
# di lunghezza di search_begin (ID validi, almeno un Base nel mazzo avversario).
_BASIC_POKEMON = sorted(
    cid for cid, c in CARD_TABLE.items()
    if c.cardType == CardType.POKEMON and getattr(c, "basic", False)
)
_BASIC_ENERGY = sorted(
    cid for cid, c in CARD_TABLE.items() if c.cardType == CardType.BASIC_ENERGY
)
FILLER_BASIC = _BASIC_POKEMON[0] if _BASIC_POKEMON else 1
FILLER_ENERGY = _BASIC_ENERGY[0] if _BASIC_ENERGY else 1


class SearchConfig:
    """Iperparametri della ricerca. Il default e' volutamente conservativo:
    banda stretta, pochi candidati, rollout corto."""

    def __init__(
        self,
        rel_margin=0.15,        # banda relativa al punteggio migliore
        abs_margin=25.0,        # banda minima assoluta (punteggi piccoli)
        max_candidates=4,
        rollout_steps=40,       # tetto di chiamate al motore per candidato
        phi_weight=0.0,         # quanto pesa Phi nel valore di foglia
        prior_weight=0.05,      # spareggio residuo a favore dell'euristica
        time_budget=None,       # secondi per questa decisione
        min_score=-0.5,         # sotto questo l'opzione e' "mai" per l'euristica
        explore_eps=0.0,        # solo in training: vedi sotto
    ):
        self.rel_margin = rel_margin
        self.abs_margin = abs_margin
        self.max_candidates = max_candidates
        self.rollout_steps = rollout_steps
        self.phi_weight = phi_weight
        self.prior_weight = prior_weight
        self.time_budget = time_budget
        self.min_score = min_score
        # Esplorazione *dentro la banda*: con probabilita' explore_eps si gioca
        # un candidato a caso invece del migliore. Serve solo a diversificare i
        # dati di training -- due agenti deterministici giocherebbero partite
        # quasi identiche e la rete vedrebbe sempre le stesse posizioni. E'
        # sicura perche' i candidati sono, per costruzione, mosse che
        # l'euristica giudica equivalenti: non e' rumore sulla policy, e'
        # rumore su un pareggio. In partita vale 0.
        self.explore_eps = explore_eps


# ---------------------------------------------------------------------------
# Candidati
# ---------------------------------------------------------------------------

def candidate_indices(scores, cfg):
    """Indici delle opzioni che l'euristica giudica equivalenti alla migliore.

    La banda e' *relativa* perche' i punteggi dell'euristica non hanno una
    scala comune: in MAIN vanno da 0 a qualche migliaio, in ATTACK stanno
    intorno a poche centinaia. Una soglia assoluta significherebbe "sempre" in
    un contesto e "mai" nell'altro.

    Le opzioni sotto `min_score` sono escluse sempre: l'euristica usa -1 per
    dire "questa mossa non va fatta", e non e' un giudizio da rimettere in
    discussione (spesso codifica una regola del mazzo, non una preferenza).
    """
    if not scores:
        return []
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    best = scores[order[0]]
    if best <= cfg.min_score:
        return order[:1]
    margin = cfg.abs_margin + cfg.rel_margin * abs(best)
    out = [i for i in order
           if scores[i] > cfg.min_score and best - scores[i] <= margin]
    return out[: max(1, cfg.max_candidates)]


# ---------------------------------------------------------------------------
# Determinizzazione
# ---------------------------------------------------------------------------

def build_determinization(obs, my_deck, rng, opponent_known=None):
    """Ipotesi completa sull'informazione nascosta, per `search_begin`.

    Le lunghezze devono combaciare *esattamente* con lo stato, altrimenti l'API
    solleva ValueError (e la ricerca viene saltata: l'euristica decide).
    """
    state = obs.current
    me_idx = state.yourIndex
    me = state.players[me_idx]
    them = state.players[1 - me_idx]

    # Il nostro mazzo lo conosciamo: campionare da cio' che resta e' molto
    # meglio che riempire a caso.
    visible = [c.id for c in (me.hand or [])] + [c.id for c in me.discard]
    for p in [x for x in me.active if x] + [x for x in me.bench if x]:
        visible.append(p.id)
        visible.extend(c.id for c in p.energyCards)
        visible.extend(c.id for c in p.tools)

    remaining = list(my_deck)
    for cid in visible:
        if cid in remaining:
            remaining.remove(cid)

    need = me.deckCount + len(me.prize)
    while len(remaining) < need:
        remaining.append(FILLER_ENERGY)
    rng.shuffle(remaining)
    my_deck_guess = remaining[:me.deckCount]
    my_prize_guess = remaining[me.deckCount:me.deckCount + len(me.prize)]

    known = list(opponent_known or []) + [c.id for c in them.discard]
    opp_total = them.deckCount + len(them.prize) + them.handCount
    opp_pool = known[:opp_total]
    while len(opp_pool) < opp_total:
        opp_pool.append(FILLER_BASIC if len(opp_pool) % 4 == 0 else FILLER_ENERGY)
    rng.shuffle(opp_pool)

    opp_deck = opp_pool[:them.deckCount]
    opp_prize = opp_pool[them.deckCount:them.deckCount + len(them.prize)]
    opp_hand = opp_pool[them.deckCount + len(them.prize):]
    if them.deckCount > 0 and FILLER_BASIC not in opp_deck:
        opp_deck[0] = FILLER_BASIC

    opp_active = []
    if them.active and them.active[0] is None:
        opp_active = [FILLER_BASIC]

    return dict(
        your_deck=my_deck_guess,
        your_prize=my_prize_guess,
        opponent_deck=opp_deck,
        opponent_prize=opp_prize,
        opponent_hand=opp_hand,
        opponent_active=opp_active,
    )


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def visible_hand(obs, player):
    """ID in mano a `player`, o None se il motore non ce la sta mostrando.

    Il motore rivela la mano solo al giocatore che ha il tratto, quindi questa
    funzione ritorna None piu' spesso di quanto sembri: serve a distinguere
    "mano vuota" da "mano non visibile", che per la rete sono due stati molto
    diversi.
    """
    try:
        ps = obs.current.players[player]
        return [c.id for c in ps.hand] if ps.hand else ([] if ps.handCount == 0 else None)
    except BaseException:
        return None


def complete_turn(state, root_player, max_steps, deadline=None):
    """Finisce il turno corrente con l'euristica.

    Ritorna `(stato_di_confine, mano_di_root_player)`. La mano e' quella
    dell'**ultimo stato in cui era visibile**, cioe' l'ultima decisione che
    root_player ha preso in questo turno: e' esattamente la mano con cui
    inizierebbe il turno successivo, ed e' invisibile allo stato di confine
    (dove il tratto e' gia' passato). Vedi `encoding.encode_state`.

    Si ferma quando: la partita finisce, il contatore di turno cambia (il
    tratto e' passato), non c'e' piu' niente da scegliere, si esaurisce il
    budget di passi o scade il tempo. Il tetto sui passi non e' difensivo per
    modo di dire: un loop di selezioni impreviste dentro un rollout costerebbe
    la partita per timeout, non una mossa mediocre.
    """
    obs = state.observation
    start_turn = obs.current.turn
    hand = visible_hand(obs, root_player)
    steps = 0
    while steps < max_steps:
        if obs.current.result >= 0:
            break
        if obs.select is None:
            break
        if obs.current.turn != start_turn:
            break
        action = heuristic.decide(obs)
        try:
            state = search_step(state.searchId, action)
        except BaseException:
            # Mossa rifiutata su questa determinizzazione: lo stato raggiunto
            # finora e' comunque una valutazione legittima, solo piu' corta.
            break
        obs = state.observation
        seen = visible_hand(obs, root_player)
        if seen is not None:
            hand = seen
        steps += 1
        if deadline is not None and time.perf_counter() > deadline:
            break
    return state, hand


# ---------------------------------------------------------------------------
# Il ricercatore
# ---------------------------------------------------------------------------

class TurnSearch:
    """Tiene modello, vocabolario e configurazione tra una decisione e l'altra.

    `model` puo' essere None: in quel caso il valore di foglia e' Phi puro.
    Serve al primo giro di training (nessun checkpoint) e come degradazione in
    partita se torch manca.
    """

    def __init__(self, model, vocab, device, cfg, rng, my_deck):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.cfg = cfg
        self.rng = rng
        self.my_deck = my_deck
        self.opponent_known = []
        # Contatori diagnostici: dicono quanto la rete stia davvero lavorando.
        self.stats = {"decisions": 0, "searched": 0, "changed": 0, "failed": 0}

    # -- osservazione dell'avversario ---------------------------------------

    def observe(self, obs):
        """Registra le carte avversarie diventate visibili (campo + scarti)."""
        try:
            them = obs.current.players[1 - obs.current.yourIndex]
            bucket = self.opponent_known
            for c in them.discard:
                bucket.append(c.id)
            for p in [x for x in them.active if x] + [x for x in them.bench if x]:
                bucket.append(p.id)
            if len(bucket) > 120:
                del bucket[:-120]
        except BaseException:
            pass

    def reset(self):
        self.opponent_known = []

    # -- valutazione ---------------------------------------------------------

    def _leaf_values(self, leaves, root_player):
        """Valore di ogni foglia `(stato, mano)`, nella prospettiva di `root_player`."""
        n = len(leaves)
        phis = np.array(
            [phi_fn(s.observation.current, root_player, CARD_TABLE, ATTACK_TABLE)
             for s, _ in leaves],
            dtype=np.float32,
        )

        # Uno stato gia' deciso non ha bisogno di stime: vale +-1.
        terminal = np.full(n, np.nan, dtype=np.float32)
        for i, (s, _) in enumerate(leaves):
            r = s.observation.current.result
            if r >= 0:
                terminal[i] = 0.0 if r == 2 else (1.0 if r == root_player else -1.0)

        if self.model is None:
            values = phis
        else:
            encs = [
                encode_state(s.observation, self.my_deck, self.vocab,
                             me_idx=root_player, hand_ids=hand)
                for s, hand in leaves
            ]
            net = evaluate_batch(self.model, encs, self.device)
            w = self.cfg.phi_weight
            values = (1.0 - w) * net + w * phis if w > 0 else net

        return np.where(np.isnan(terminal), values, terminal)

    # -- API -----------------------------------------------------------------

    def choose(self, obs, board, select):
        """Indici scelti, o None se la ricerca non si applica / non serve.

        None significa "decidi tu, euristica": e' il ritorno piu' comune ed e'
        il motivo per cui l'agente resta economico.
        """
        cfg = self.cfg
        scores = heuristic.score_options(obs, board, select)
        if scores is None:
            return None
        self.stats["decisions"] += 1

        cands = candidate_indices(scores, cfg)
        if len(cands) < 2:
            return None
        # Un colpo che chiude la partita non si mette ai voti.
        if any(heuristic.is_winning_attack(board, select.option[i]) for i in cands):
            return None

        deadline = (time.perf_counter() + cfg.time_budget) if cfg.time_budget else None
        root_player = obs.current.yourIndex

        try:
            det = build_determinization(obs, self.my_deck, self.rng, self.opponent_known)
            root = search_begin(obs, **det)
        except BaseException:
            # search_begin puo' fallire dopo aver gia' allocato: chiudere e'
            # sempre sicuro (search_end su una ricerca inesistente non fa nulla)
            # e non chiudere lascerebbe memoria appesa per tutta la partita.
            try:
                search_end()
            except BaseException:
                pass
            self.stats["failed"] += 1
            return None

        try:
            leaves = []
            kept = []
            for i in cands:
                if deadline is not None and time.perf_counter() > deadline:
                    break
                try:
                    nxt = search_step(root.searchId, [i])
                except BaseException:
                    # Il motore rifiuta questa mossa sulla determinizzazione
                    # corrente: si scarta il candidato, non la ricerca.
                    continue
                leaves.append(complete_turn(nxt, root_player, cfg.rollout_steps, deadline))
                kept.append(i)

            if len(kept) < 2:
                return None

            values = self._leaf_values(leaves, root_player)

            # Spareggio residuo a favore dell'euristica: quando due candidati
            # hanno valore quasi identico (e con una rete poco allenata capita
            # spesso) vince quello che l'euristica preferiva. Costa poco e
            # rende l'agente monotono rispetto a v1 anche a rete debole.
            if cfg.prior_weight > 0:
                hs = np.array([scores[i] for i in kept], dtype=np.float32)
                span = float(hs.max() - hs.min())
                if span > 1e-9:
                    values = values + cfg.prior_weight * (hs - hs.min()) / span

            if cfg.explore_eps > 0 and self.rng.random() < cfg.explore_eps:
                pick = self.rng.choice(kept)
            else:
                pick = kept[int(np.argmax(values))]
            self.stats["searched"] += 1
            if pick != cands[0]:
                self.stats["changed"] += 1
            return [pick]
        except BaseException:
            self.stats["failed"] += 1
            return None
        finally:
            try:
                search_end()
            except BaseException:
                pass


# ---------------------------------------------------------------------------
# L'agente ibrido completo
# ---------------------------------------------------------------------------

class HybridAgent:
    """Euristica + ricerca, senza budget temporale.

    Vive qui e non in main.py perche' deve essere **lo stesso oggetto** in
    partita e in training: se la policy che genera i dati non fosse identica
    a quella che li usa, la rete imparerebbe a valutare posizioni che poi non
    incontra mai. E' l'errore piu' facile da introdurre in un ibrido e il piu'
    difficile da vedere dalle metriche di training.

    main.py ci aggiunge sopra il budget temporale e la gestione degli errori;
    train.py lo usa nudo.
    """

    def __init__(self, searcher):
        self.searcher = searcher
        self.opponent_model = heuristic.OpponentModel()

    def reset(self):
        self.opponent_model = heuristic.OpponentModel()
        self.searcher.reset()

    def select(self, obs):
        """Indici scelti per l'`obs` corrente (Observation, non dict)."""
        select = obs.select
        board = heuristic.make_board(obs, self.opponent_model)
        self.opponent_model.observe(board.them)
        self.searcher.observe(obs)
        picked = self.searcher.choose(obs, board, select)
        if picked:
            return heuristic._sanitize(picked, select)
        return heuristic.decide(obs, board, select)
