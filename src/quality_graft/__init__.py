"""Quality-Graft: Bridge La-Proteina trunk to Boltz1 confidence head."""

import sys as _sys
from pathlib import Path as _Path

# La-Proteina's internal code uses bare imports (``from proteinfoundation...``,
# ``from openfold...``) that require its own directory on sys.path.
_la_proteina_root = str(_Path(__file__).resolve().parent.parent / "la_proteina")
if _la_proteina_root not in _sys.path:
    _sys.path.insert(0, _la_proteina_root)
