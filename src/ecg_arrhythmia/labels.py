"""Label definitions and AAMI EC57 mapping.

Two label schemes are supported:

- :data:`MITBIH3` — the legacy 3-class scheme (``N`` / ``V`` / ``a``) used by
  v0.1–v0.3 of this library. Kept as the default for back-compat.
- :data:`AAMI5` — the AAMI EC57 5-class scheme (``N`` / ``S`` / ``V`` / ``F`` /
  ``Q``) used by virtually every published MIT-BIH benchmark since
  de Chazal et al. 2004. Letting users opt in to AAMI5 means our metrics
  become directly comparable to those papers.

Mapping references:
- AAMI EC57 (2012, "Testing and reporting performance results of cardiac
  rhythm and ST segment measurement algorithms")
- de Chazal, O'Dwyer & Reilly 2004 ("Automatic classification of heartbeats
  using ECG morphology and heartbeat interval features")
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LabelScheme:
    """A self-contained mapping from MIT-BIH WFDB symbols to integer class ids.

    Attributes
    ----------
    name : str
        Short identifier, e.g. ``"mitbih3"`` or ``"aami5"``.
    label_to_id : dict[str, int]
        WFDB symbol → class id. The same id may appear for several symbols
        (e.g. ``A`` and ``a`` both map to id 2 under MITBIH3, all of
        ``A``/``a``/``J``/``S`` map to id 1 under AAMI5).
    id_to_label : dict[int, str]
        Class id → display label (one short string per class).
    n_classes : int
        Number of classes; ``len(id_to_label)``.
    """

    name: str
    label_to_id: dict[str, int]
    id_to_label: dict[int, str]

    @property
    def n_classes(self) -> int:
        return len(self.id_to_label)

    @property
    def target_symbols(self) -> tuple[str, ...]:
        """WFDB symbols that should be retained when segmenting records."""
        return tuple(self.label_to_id.keys())

    @property
    def labels(self) -> list[str]:
        """Display labels in id order, e.g. ``["N", "V", "a"]``."""
        return [self.id_to_label[i] for i in range(self.n_classes)]


# ---------------------------------------------------------------------------
# Legacy 3-class scheme: N / V / a
# ---------------------------------------------------------------------------
# - N : normal beats
# - V : premature ventricular contraction (PVC)
# - a : premature atrial contraction (PAC); MIT-BIH 'A' and 'a' merged
MITBIH3 = LabelScheme(
    name="mitbih3",
    label_to_id={"N": 0, "V": 1, "A": 2, "a": 2},
    id_to_label={0: "N", 1: "V", 2: "a"},
)


# ---------------------------------------------------------------------------
# AAMI EC57 5-class scheme: N / S / V / F / Q
# ---------------------------------------------------------------------------
# - N (Normal):                 N, L (LBBB), R (RBBB), e (atrial escape), j (nodal escape)
# - S (Supraventricular):       A, a (PAC, both capitalisations), J (nodal premature),
#                               S (supraventricular ectopic)
# - V (Ventricular):            V (PVC), E (ventricular escape)
# - F (Fusion):                 F (fusion of V and N)
# - Q (Unknown / paced):        Q, /, f (fusion of paced and N), ?
#                               '|' is deliberately excluded — some annotation
#                               tools use it as an artifact marker rather than
#                               a beat label. See docs/MODEL_CARD.md.
AAMI5 = LabelScheme(
    name="aami5",
    label_to_id={
        # N
        "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
        # S
        "A": 1, "a": 1, "J": 1, "S": 1,
        # V
        "V": 2, "E": 2,
        # F
        "F": 3,
        # Q
        "Q": 4, "/": 4, "f": 4, "?": 4,
    },
    id_to_label={0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"},
)


SCHEMES: dict[str, LabelScheme] = {
    "mitbih3": MITBIH3,
    "aami5": AAMI5,
}


def get_scheme(name: str) -> LabelScheme:
    """Look up a scheme by name (``"mitbih3"`` or ``"aami5"``)."""
    if name not in SCHEMES:
        raise ValueError(f"Unknown label scheme {name!r}; expected one of {sorted(SCHEMES)}")
    return SCHEMES[name]


# ---------------------------------------------------------------------------
# Module-level back-compat aliases
# ---------------------------------------------------------------------------
# Older code (and notebooks in the wild) imports ``LABEL_TO_ID``, ``ID_TO_LABEL``,
# and ``TARGET_SYMBOLS`` directly from this module. We keep those bindings
# pointing at MITBIH3 so they continue to work; new code should reach for a
# :class:`LabelScheme` explicitly.
LABEL_TO_ID = MITBIH3.label_to_id
ID_TO_LABEL = MITBIH3.id_to_label
TARGET_SYMBOLS = MITBIH3.target_symbols


__all__ = [
    "LabelScheme",
    "MITBIH3",
    "AAMI5",
    "SCHEMES",
    "get_scheme",
    "LABEL_TO_ID",
    "ID_TO_LABEL",
    "TARGET_SYMBOLS",
]
