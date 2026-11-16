"""One place to read every setting the online guard uses (DESIGN.md section 9).

The defaults still live where they are used, as named module constants (AGENTS.md
section 5). This module imports them rather than copying them, so changing a default
in its module changes it here too.

GuardConfig holds the values guard() and the retriever wrapper need, in three ways:
    GuardConfig.from_reference()   the shipped, labelled values; refuses another embedder
    GuardConfig.from_file(path)    a local thresholds.json written by build_thresholds
    GuardConfig(threshold, clean_median, ...)   explicit values; the rest keep the placeholders

guard_kwargs() and retriever_kwargs() return keyword arguments for pipeline.guard()
and retriever.TrustRAGGuard, so a configuration is passed as a whole.

Known limits: a config describes one embedder on one corpus at one retrieval depth,
like the values it loads. Validation catches impossible values (low above high, a
threshold below the clean median), not values that are merely poor. Loading this
module does not load Chroma or LangChain; retriever_kwargs() does, because the
over-retrieval default lives in the retriever.
"""
from dataclasses import asdict, dataclass, field, replace

from .calibration.build_thresholds import DEFAULT_PATH, EMBEDDING_MODEL, REFERENCE_PATH, load, load_reference
from .context_floor import DEFAULT_K_MIN
from .fusion import RISK_HIGH, RISK_LOW
from .layers.clustering import DEFAULT_TAU
from .layers.text_space import DEFAULT_COPY_CUTOFF
from .layers.verifier import DEFAULT_MAX_VERIFIER_CALLS

SEED = 42
FITTED_KEYS = ("threshold", "clean_median", "tau", "low", "high")


def _retriever_defaults():
    """(over_retrieve, target_k) from the retriever, imported only when needed."""
    from .retriever import DEFAULT_OVER_RETRIEVE, DEFAULT_TARGET_K
    return DEFAULT_OVER_RETRIEVE, DEFAULT_TARGET_K


@dataclass(frozen=True)
class GuardConfig:
    threshold: float                 # signal A: clean similarity level, fitted offline
    clean_median: float              # median clean similarity, fitted offline
    tau: float = DEFAULT_TAU         # silhouette cutoff for the k=2 split
    low: float = RISK_LOW            # risk at which a chunk becomes ambiguous
    high: float = RISK_HIGH          # risk at which a chunk is dropped
    copy_cutoff: float = DEFAULT_COPY_CUTOFF
    k_min: int = DEFAULT_K_MIN
    max_calls: int = DEFAULT_MAX_VERIFIER_CALLS
    seed: int = SEED
    over_retrieve: int = field(default_factory=lambda: _retriever_defaults()[0])
    target_k: int = field(default_factory=lambda: _retriever_defaults()[1])

    def __post_init__(self):
        problems = []
        if not -1.0 <= self.clean_median < self.threshold <= 1.0:
            problems.append("need -1 <= clean_median < threshold <= 1 (cosine scale)")
        if not 0.0 <= self.low < self.high:
            problems.append("need 0 <= low < high")
        if not -1.0 <= self.tau <= 1.0:
            problems.append("tau is a silhouette value, so it must lie in [-1, 1]")
        if not 0.0 < self.copy_cutoff <= 1.0:
            problems.append("copy_cutoff is a ROUGE-L value, so it must lie in (0, 1]")
        if self.k_min < 0 or self.max_calls < 0:
            problems.append("k_min and max_calls must not be negative")
        if self.target_k < 1 or self.over_retrieve < self.target_k:
            problems.append("need 1 <= target_k <= over_retrieve")
        if problems:
            raise ValueError("invalid GuardConfig: " + "; ".join(problems))

    @classmethod
    def from_reference(cls, embedding_model=EMBEDDING_MODEL, path=REFERENCE_PATH, **overrides):
        """The shipped values. Raises ValueError for another embedding model."""
        values = load_reference(embedding_model, path)
        return cls(**{k: values[k] for k in FITTED_KEYS}, **overrides)

    @classmethod
    def from_file(cls, path=DEFAULT_PATH, **overrides):
        """Values from a thresholds.json written by build_thresholds.save()."""
        values = load(path)
        return cls(**{k: values[k] for k in FITTED_KEYS}, **overrides)

    def with_changes(self, **changes):
        """A copy with some values replaced, validated again."""
        return replace(self, **changes)

    def guard_kwargs(self):
        """Keyword arguments for pipeline.guard(), after its positional arguments."""
        return {"threshold": self.threshold, "clean_median": self.clean_median, "k_min": self.k_min,
                "tau": self.tau, "seed": self.seed, "cutoff": self.copy_cutoff,
                "max_calls": self.max_calls, "low": self.low, "high": self.high}

    def retriever_kwargs(self):
        """Keyword arguments for retriever.TrustRAGGuard. The wrapper uses the default
        seed and copy cutoff, so a config that changes them is refused here."""
        if self.seed != SEED or self.copy_cutoff != DEFAULT_COPY_CUTOFF:
            raise ValueError("TrustRAGGuard uses the default seed and copy_cutoff; "
                             "call pipeline.guard() with guard_kwargs() to change them")
        return {"threshold": self.threshold, "clean_median": self.clean_median, "tau": self.tau,
                "low": self.low, "high": self.high, "k_min": self.k_min, "max_calls": self.max_calls,
                "over_retrieve": self.over_retrieve, "target_k": self.target_k}

    def to_dict(self):
        return asdict(self)
