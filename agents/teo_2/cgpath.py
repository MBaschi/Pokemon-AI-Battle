"""Rende importabile il package `cg` (engine della competizione).

Va importato *per primo* da ogni modulo di teo_2 che tocca `cg.api`, perche'
`cg` non e' pip-installabile: su Kaggle arriva dal dataset allegato alla
competizione, in locale sta alla root del repo.
"""

import glob
import sys
from pathlib import Path

_done = False


def ensure():
    global _done
    if _done:
        return
    kaggle_matches = glob.glob("/kaggle/input/**/cg-lib", recursive=True)
    if kaggle_matches:
        sys.path.append(kaggle_matches[0])
        _done = True
        return
    for base in (Path(__file__).resolve().parents[2], Path.cwd()):
        if (base / "cg").is_dir():
            if str(base) not in sys.path:
                sys.path.append(str(base))
            _done = True
            return
    # Kaggle monta l'agente qui quando gira una submission.
    fallback = Path("/kaggle_simulations/agent")
    if (fallback / "cg").is_dir():
        sys.path.append(str(fallback))
        _done = True
        return
    raise ImportError(
        "cg engine package non trovato. Su Kaggle arriva dal dataset allegato "
        "alla competizione; in locale scarica i materiali starter e metti il "
        "package `cg/` (con il suo .so) alla root del repo. Vedi README.md."
    )


ensure()
