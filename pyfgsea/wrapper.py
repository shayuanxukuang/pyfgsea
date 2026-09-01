import hashlib
import logging
import math
import warnings
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union, cast

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from . import _core as _ext  # type: ignore
except ImportError:
    # Fallback for development/editable mode where _core might be in the same directory
    import _core as _ext  # type: ignore

# Expose core functions
fgsea_multilevel = _ext.fgsea_multilevel
fgsea_multilevel_batched = _ext.fgsea_multilevel_batched
fgsea_multilevel_batched_scores = _ext.fgsea_multilevel_batched_scores
get_random_es_means = _ext.get_random_es_means
_build_tail_curve_ext = _ext.build_tail_curve
_query_tail_curve_ext = _ext.query_tail_curve
calculate_es = _ext.calculate_es
GseaPrerankedRunner = _ext.GseaPrerankedRunner


_TAIL_CURVE_DEPRECATION = (
    "build_tail_curve/query_tail_curve are deprecated legacy empirical helpers; "
    "they do not run the aligned multilevel estimator. Use run_gsea or "
    "fgsea_multilevel for aligned inference."
)


def build_tail_curve(
    scores,
    size,
    sample_size,
    seed,
    gsea_param,
    eps,
    score_type=None,
    sign=1,
):
    """Build the deprecated, approximate empirical tail helper."""
    warnings.warn(_TAIL_CURVE_DEPRECATION, DeprecationWarning, stacklevel=2)
    canonical_score_type = _normalize_score_type(score_type)
    return _build_tail_curve_ext(
        scores,
        size,
        sample_size,
        seed,
        gsea_param,
        eps,
        canonical_score_type,
        sign,
    )


def query_tail_curve(curve, obs_es, score_type=None, sign=None):
    """Query the deprecated, approximate empirical tail helper."""
    warnings.warn(_TAIL_CURVE_DEPRECATION, DeprecationWarning, stacklevel=2)
    canonical_score_type = (
        None if score_type is None else _normalize_score_type(score_type)
    )
    return _query_tail_curve_ext(curve, obs_es, canonical_score_type, sign)


def _missing_multilevel_error(*_args, **_kwargs):
    raise RuntimeError(
        "The loaded pyfgsea Rust extension predates multilevel_error; rebuild "
        "the extension from the current source tree."
    )


multilevel_error = getattr(_ext, "multilevel_error", _missing_multilevel_error)

__all__ = [
    "run_gsea",
    "load_gmt",
    "prepare_pathways",
    "GseaRunner",
    "multilevel_error",
    "build_tail_curve",
    "query_tail_curve",
]

GeneSetDict = Dict[str, List[str]]
Mode = Literal["aligned", "fast"]
ScoreType = Literal["std", "pos", "neg"]
TiePolicy = Literal["gene_id", "input_order", "error"]

_VALID_MODES = {"aligned", "fast"}
_VALID_SCORE_TYPES = {"std", "pos", "neg"}
_LEGACY_SCORE_TYPES = {"two_sided_abs", "one_sided_signed"}
_VALID_TIE_POLICIES = {"gene_id", "input_order", "error"}
_VALID_DEDUP_POLICIES = {"max_abs", "first", "error"}
_FALLBACK_ALGORITHM_REVISION = "unknown-unverified-extension"
_SUPPORTED_ALGORITHM_REVISIONS = {"fgsea-1.38-pr178-v1"}
_WARNED_LEGACY_SCORE_TYPES: set[str] = set()


def _algorithm_revision() -> str:
    for name in ("algorithm_revision", "ALGORITHM_REVISION"):
        value = getattr(_ext, name, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                value = None
        if value:
            return str(value)
    return _FALLBACK_ALGORITHM_REVISION


def _require_int(name: str, value, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _require_float(name: str, value) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be numeric")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _normalize_mode(mode: str) -> str:
    mode = str(mode).lower().replace("-", "_")
    if mode not in _VALID_MODES:
        raise ValueError("mode must be one of 'aligned' or 'fast'")
    return mode


def _normalize_score_type(score_type: Optional[str]) -> str:
    if score_type is None:
        return "std"
    score_type = str(score_type).lower().replace("-", "_")
    if score_type in _VALID_SCORE_TYPES:
        return score_type
    if score_type == "two_sided_abs":
        if score_type not in _WARNED_LEGACY_SCORE_TYPES:
            warnings.warn(
                "score_type='two_sided_abs' is deprecated. It is a PyFgsea-specific "
                "absolute-tail mode and is not equivalent to "
                "fgseaMultilevel(scoreType='std').",
                FutureWarning,
                stacklevel=3,
            )
            _WARNED_LEGACY_SCORE_TYPES.add(score_type)
        return score_type
    if score_type == "one_sided_signed":
        if score_type not in _WARNED_LEGACY_SCORE_TYPES:
            warnings.warn(
                "score_type='one_sided_signed' is deprecated; use 'std', 'pos', or 'neg'.",
                FutureWarning,
                stacklevel=3,
            )
            _WARNED_LEGACY_SCORE_TYPES.add(score_type)
        return score_type
    allowed = sorted(_VALID_SCORE_TYPES | _LEGACY_SCORE_TYPES)
    raise ValueError(f"score_type must be one of {allowed}")


def _normalize_tie_policy(tie_policy: str) -> TiePolicy:
    tie_policy = str(tie_policy).lower().replace("-", "_")
    if tie_policy not in _VALID_TIE_POLICIES:
        raise ValueError("tie_policy must be 'gene_id', 'input_order', or 'error'")
    return cast(TiePolicy, tie_policy)


def _normalize_sample_size(sample_size: int) -> int:
    sample_size = _require_int("sample_size", sample_size, 3)
    if sample_size % 2 == 0:
        adjusted = sample_size + 1
        warnings.warn(
            f"sample_size={sample_size} is even; using {adjusted} so the "
            "multilevel median split is unambiguous.",
            UserWarning,
            stacklevel=3,
        )
        return adjusted
    return sample_size


def _normalize_run_options(
    *,
    sample_size: int,
    seed: int,
    nperm_nes: int,
    gsea_param: float,
    eps: float,
    score_type: Optional[str],
    mode: str,
    tie_policy: str,
    bin_width: Optional[int],
    calculate_nes: bool,
    nperm_simple: Optional[int],
    max_levels: Optional[int],
    precheck_n: Optional[int] = None,
    precheck_eps: Optional[float] = None,
) -> dict:
    mode = _normalize_mode(mode)
    score_type = _normalize_score_type(score_type)
    revision = _algorithm_revision()
    if mode == "aligned" and revision not in _SUPPORTED_ALGORITHM_REVISIONS:
        raise RuntimeError(
            "The loaded Rust extension does not expose a supported algorithm "
            f"revision ({revision!r}); mode='aligned' requires the verified "
            "fgsea-1.38-pr178-v1 core."
        )
    tie_policy = _normalize_tie_policy(tie_policy)
    sample_size = _normalize_sample_size(sample_size)
    seed = _require_int("seed", seed, 0)
    if seed > np.iinfo(np.uint64).max:
        raise ValueError("seed must fit in an unsigned 64-bit integer")
    nperm_nes = _require_int("nperm_nes", nperm_nes, 1)
    if nperm_simple is not None:
        nperm_simple = _require_int("nperm_simple", nperm_simple, 1)
    if max_levels is not None:
        max_levels = _require_int("max_levels", max_levels, 0)
    gsea_param = _require_float("gsea_param", gsea_param)
    if gsea_param < 0:
        raise ValueError("gsea_param must be >= 0")
    eps = _require_float("eps", eps)
    if not 0 <= eps <= 1:
        raise ValueError("eps must be between 0 and 1 inclusive")
    if not isinstance(calculate_nes, (bool, np.bool_)):
        raise TypeError("calculate_nes must be boolean")
    bin_width = 0 if bin_width is None else _require_int("bin_width", bin_width, 0)
    if precheck_n is not None:
        precheck_n = _require_int("precheck_n", precheck_n, 1)
    if precheck_eps is not None:
        precheck_eps = _require_float("precheck_eps", precheck_eps)
        if not 0 <= precheck_eps <= 1:
            raise ValueError("precheck_eps must be between 0 and 1 inclusive")
    if mode == "aligned" and bin_width > 0:
        raise ValueError(
            "mode='aligned' requires exact pathway sizes (bin_width must be 0 or None)"
        )
    if mode == "aligned" and (precheck_n is not None or precheck_eps is not None):
        raise ValueError(
            "mode='aligned' does not permit fast precheck parameters; use mode='fast'"
        )
    if mode == "fast":
        if precheck_n is None:
            precheck_n = nperm_simple if nperm_simple is not None else 64
        if precheck_eps is None:
            precheck_eps = 0.005
    elif nperm_simple is None:
        nperm_simple = 1000
    return {
        "mode": mode,
        "score_type": score_type,
        "tie_policy": tie_policy,
        "sample_size": sample_size,
        "seed": seed,
        "nperm_nes": nperm_nes,
        "nperm_simple": nperm_simple,
        "max_levels": max_levels,
        "gsea_param": gsea_param,
        "eps": eps,
        "bin_width": bin_width,
        "calculate_nes": bool(calculate_nes),
        "precheck_n": precheck_n,
        "precheck_eps": precheck_eps,
    }


def _validate_score_vector(scores, name: str = "scores") -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if scores.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(scores).all():
        bad = int((~np.isfinite(scores)).sum())
        raise ValueError(f"{name} contains {bad} NaN or infinite values")
    return np.ascontiguousarray(scores)


def _warn_or_error_ties(
    scores: np.ndarray,
    tie_policy: str,
    *,
    gene_ids_available: bool,
    warn: bool = True,
) -> bool:
    values = pd.Series(scores)
    tied = values.ne(0.0) & values.duplicated(keep=False)
    if not bool(tied.any()):
        return False
    percentage = 100.0 * float(tied.mean())
    if tie_policy == "error":
        raise ValueError(
            "There are non-zero ties in the preranked statistics "
            f"({percentage:.1f}% of entries); tie_policy='error' forbids them."
        )
    if tie_policy == "gene_id" and not gene_ids_available:
        raise ValueError(
            "tie_policy='gene_id' requires gene_ids when GseaRunner receives tied scores"
        )
    if warn:
        warnings.warn(
            "There are non-zero ties in the preranked statistics "
            f"({percentage:.1f}% of entries). Their within-tie order is determined "
            f"by tie_policy='{tie_policy}' and may affect ES.",
            UserWarning,
            stacklevel=3,
        )
    return True


def _cache_ranking_hash(scores: np.ndarray) -> str:
    ranked = np.sort(np.asarray(scores, dtype=np.float64), kind="mergesort")[::-1]
    canonical = np.ascontiguousarray(ranked.astype("<f8", copy=False))
    digest = hashlib.sha256()
    digest.update(str(canonical.size).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _public_ranking_hash(results: pd.DataFrame, fallback: str) -> str:
    if results.empty or "ranking_hash" not in results:
        return fallback
    hashes = results["ranking_hash"].dropna().astype(str).unique()
    if len(hashes) != 1:
        raise RuntimeError("Rust core returned inconsistent ranking hashes")
    return str(hashes[0])


def _validate_binning(bin_width: int, pathway_sizes: Sequence[int], n_genes: int) -> None:
    if bin_width <= 0:
        return
    if bin_width > n_genes:
        raise ValueError("bin_width cannot exceed the ranked gene count")
    invalid = []
    for size in sorted(set(map(int, pathway_sizes))):
        null_size = ((size + bin_width // 2) // bin_width) * bin_width
        null_size = bin_width if null_size == 0 else null_size
        if null_size <= 0 or null_size >= n_genes:
            invalid.append((size, null_size))
    if invalid:
        details = ", ".join(f"{size}->{null_size}" for size, null_size in invalid[:5])
        raise ValueError(
            "bin_width produces an invalid null pathway size for this gene universe: "
            f"{details}"
        )


def load_gmt(gmt_path: str) -> GeneSetDict:
    """Parse a GMT file into a pathway-to-members mapping."""
    pathways: GeneSetDict = {}
    with open(gmt_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            if name in pathways:
                warnings.warn(
                    f"Duplicate pathway name {name!r} at GMT line {line_number}; "
                    "the later definition replaces the earlier one.",
                    UserWarning,
                    stacklevel=2,
                )
            pathways[name] = parts[2:]
    return pathways


def prepare_pathways(
    genes: Union[Sequence[str], np.ndarray],
    gmt: Union[str, GeneSetDict],
    min_size: int = 15,
    max_size: int = 500,
) -> Tuple[List[str], List[List[int]]]:
    """Map unique pathway members to ranked-gene indices, then filter by size."""
    gene_list = [str(gene) for gene in genes]
    if not gene_list:
        raise ValueError("genes must be non-empty")
    duplicate_genes = pd.Index(gene_list).duplicated(keep=False)
    if duplicate_genes.any():
        examples = ", ".join(map(str, pd.Index(gene_list)[duplicate_genes][:5]))
        raise ValueError(f"genes contains duplicate identifiers. Examples: {examples}")
    min_size = _require_int("min_size", min_size, 1)
    max_size = _require_int("max_size", max_size, 1)
    if max_size < min_size:
        raise ValueError("max_size must be >= min_size")
    if len(gene_list) < 2:
        raise ValueError("at least two ranked genes are required")
    if min_size >= len(gene_list):
        raise ValueError("min_size must be smaller than the ranked gene count")
    effective_max_size = min(max_size, len(gene_list) - 1)
    if effective_max_size != max_size:
        warnings.warn(
            f"max_size={max_size} is not smaller than the ranked gene count; "
            f"using max_size={effective_max_size}.",
            UserWarning,
            stacklevel=2,
        )

    raw_pathways = load_gmt(gmt) if isinstance(gmt, str) else gmt
    if not isinstance(raw_pathways, dict):
        raise TypeError("gmt must be a path or a pathway-to-members mapping")
    gene_to_idx = {gene: index for index, gene in enumerate(gene_list)}
    filtered_pathways = []
    duplicate_member_sets = 0
    full_universe_sets = 0
    for raw_name, gene_set in raw_pathways.items():
        members = [str(gene) for gene in gene_set]
        mapped = [gene_to_idx[gene] for gene in members if gene in gene_to_idx]
        indices = sorted(set(mapped))
        duplicate_member_sets += int(len(indices) != len(mapped))
        if len(indices) == len(gene_list):
            full_universe_sets += 1
            continue
        if min_size <= len(indices) <= effective_max_size:
            filtered_pathways.append((str(raw_name), indices))

    if duplicate_member_sets:
        warnings.warn(
            f"Removed duplicate pathway members before size filtering in "
            f"{duplicate_member_sets} pathway(s).",
            UserWarning,
            stacklevel=2,
        )
    if full_universe_sets:
        warnings.warn(
            f"Excluded {full_universe_sets} pathway(s) whose mapped size equals "
            "the ranked gene universe.",
            UserWarning,
            stacklevel=2,
        )

    if not filtered_pathways:
        logger.warning(
            f"No valid pathways found after filtering (min_size={min_size}, max_size={effective_max_size}). "
            f"Raw pathways: {len(raw_pathways)}. "
            f"Ensure gene symbols match your input data."
        )
        return [], []

    pathway_names, pathway_indices = zip(*filtered_pathways)
    return list(pathway_names), list(pathway_indices)


def _result_value(result, name: str, default):
    value = getattr(result, name, default)
    try:
        return value() if callable(value) else value
    except Exception:
        return default


def _format_results(
    pathway_names: Sequence[str],
    pathway_sizes: Sequence[int],
    multi_results: Sequence,
    mean_lookup: Dict[int, Sequence[float]],
    *,
    calculate_nes: bool,
    eps: float,
    mode: str,
    score_type: str,
    bin_width: int,
    ranking_hash: str,
) -> pd.DataFrame:
    records = []
    revision = _algorithm_revision()
    for name, size, result in zip(pathway_names, pathway_sizes, multi_results):
        es = float(_result_value(result, "es", np.nan))
        pval = float(_result_value(result, "pval", np.nan))
        log2err = float(_result_value(result, "log2err", np.nan))
        nes = np.nan
        if calculate_nes and size in mean_lookup:
            pos_mean, neg_mean = map(float, mean_lookup[size])
            if es > 0 and pos_mean > 1e-12:
                nes = es / pos_mean
            elif es < 0 and abs(neg_mean) > 1e-12:
                nes = es / abs(neg_mean)
            elif es == 0:
                nes = 0.0

        debug = _result_value(result, "debug_info", None)
        debug_levels = _result_value(debug, "current_level", 0) if debug is not None else 0
        debug_rates = (
            list(_result_value(debug, "accept_rates", [])) if debug is not None else []
        )
        n_levels = int(_result_value(result, "n_levels", debug_levels) or 0)
        acceptance_min = float(
            _result_value(
                result,
                "acceptance_rate_min",
                min(debug_rates) if debug_rates else np.nan,
            )
        )
        acceptance_mean = float(
            _result_value(
                result,
                "acceptance_rate_mean",
                float(np.mean(debug_rates)) if debug_rates else np.nan,
            )
        )
        if math.isfinite(pval) and pval > 0:
            default_log_pval = math.log(pval)
        elif pval == 0:
            default_log_pval = -math.inf
        else:
            default_log_pval = np.nan
        log_pval = float(_result_value(result, "log_pval", default_log_pval))
        null_curve_size = int(_result_value(result, "null_curve_size", size) or size)
        if not math.isfinite(pval):
            default_status = "unresolved"
        elif eps > 0 and pval <= eps:
            default_status = "eps_floor"
        else:
            default_status = "resolved"
        approximate_default = (
            mode == "fast" or bin_width > 0 or score_type in _LEGACY_SCORE_TYPES
        )
        approximate = bool(
            _result_value(result, "approximate", approximate_default)
        )
        records.append(
            {
                "Pathway": str(name),
                "ES": es,
                "NES": nes,
                "P-value": pval,
                "log_pval": log_pval,
                "log2err": log2err,
                "Size": int(size),
                "observed_pathway_size": int(size),
                "null_curve_size": null_curve_size,
                "size_binned": bool(null_curve_size != int(size)),
                "approximate": approximate,
                "approximate_mode": approximate,
                "status": str(_result_value(result, "status", default_status)),
                "termination_reason": str(
                    _result_value(result, "termination_reason", "") or ""
                ),
                "n_levels": n_levels,
                "acceptance_rate_min": acceptance_min,
                "acceptance_rate_mean": acceptance_mean,
                "ranking_hash": str(
                    _result_value(result, "ranking_hash", ranking_hash) or ranking_hash
                ),
                "algorithm_revision": str(
                    _result_value(result, "algorithm_revision", revision) or revision
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _add_bh_adjustment(results: pd.DataFrame, eps: float) -> pd.DataFrame:
    """BH-adjust finite p-values without allowing unresolved NaNs to propagate."""
    if results.empty:
        return results
    results = results.copy()
    pvalues = pd.to_numeric(results["P-value"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(pvalues) & (pvalues >= 0.0) & (pvalues <= 1.0)
    adjusted = np.full(len(results), np.nan, dtype=float)
    if valid.any():
        valid_indices = np.flatnonzero(valid)
        order = np.argsort(pvalues[valid_indices], kind="mergesort")
        ordered_indices = valid_indices[order]
        ranks = np.arange(1, len(ordered_indices) + 1, dtype=float)
        raw = pvalues[ordered_indices] * len(results) / ranks
        adjusted[ordered_indices] = np.minimum(
            np.minimum.accumulate(raw[::-1])[::-1], 1.0
        )
    results["padj"] = adjusted
    results["pval_capped"] = valid & (pvalues <= eps) if eps > 0 else False
    return results.sort_values(
        ["P-value", "Pathway"],
        ascending=[True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


class GseaRunner:
    """Stateful repeated-ranking runner with a ranking-aware optional NES cache."""

    def __init__(
        self,
        pathway_names: List[str],
        pathway_indices: List[List[int]],
        min_size: int = 15,
        max_size: int = 500,
        *,
        gene_ids: Optional[Sequence[str]] = None,
        tie_policy: TiePolicy = "gene_id",
    ):
        if len(pathway_names) != len(pathway_indices):
            raise ValueError("pathway_names and pathway_indices must have equal length")
        names = [str(name) for name in pathway_names]
        if not names:
            raise ValueError("at least one pathway is required")
        if len(set(names)) != len(names):
            raise ValueError("pathway_names must be unique")
        min_size = _require_int("min_size", min_size, 1)
        max_size = _require_int("max_size", max_size, 1)
        if max_size < min_size:
            raise ValueError("max_size must be >= min_size")
        tie_policy = _normalize_tie_policy(tie_policy)

        prepared = []
        duplicate_sets = 0
        for pathway in pathway_indices:
            raw = [int(index) for index in pathway]
            if any(index < 0 for index in raw):
                raise ValueError("pathway indices must be non-negative")
            unique = sorted(set(raw))
            duplicate_sets += int(len(unique) != len(raw))
            if not min_size <= len(unique) <= max_size:
                raise ValueError(
                    f"pathway size {len(unique)} is outside [{min_size}, {max_size}]"
                )
            prepared.append(unique)
        if duplicate_sets:
            warnings.warn(
                f"Removed duplicate members from {duplicate_sets} pathway(s) before "
                "constructing GseaRunner.",
                UserWarning,
                stacklevel=2,
            )

        self.pathway_names = names
        self.min_size = min_size
        self.max_size = max_size
        self.tie_policy = tie_policy
        self._gene_ids: Optional[np.ndarray] = None
        self._score_order: Optional[np.ndarray] = None
        if gene_ids is not None:
            ids = np.asarray([str(gene) for gene in gene_ids], dtype=object)
            if ids.size == 0:
                raise ValueError("gene_ids must be non-empty")
            duplicates = pd.Index(ids).duplicated(keep=False)
            if duplicates.any():
                examples = ", ".join(map(str, pd.Index(ids)[duplicates][:5]))
                raise ValueError(f"gene_ids contains duplicates. Examples: {examples}")
            if any(pathway and pathway[-1] >= len(ids) for pathway in prepared):
                raise ValueError("pathway index is outside gene_ids")
            if any(len(pathway) == len(ids) for pathway in prepared):
                raise ValueError("pathway size cannot equal the ranked gene count")
            self._gene_ids = ids

        if tie_policy == "gene_id" and self._gene_ids is not None:
            self._score_order = np.argsort(self._gene_ids.astype(str), kind="mergesort")
            rank_position = np.empty(len(self._score_order), dtype=int)
            rank_position[self._score_order] = np.arange(len(self._score_order))
            prepared = [
                sorted(rank_position[np.asarray(pathway, dtype=int)].tolist())
                for pathway in prepared
            ]

        self.pathway_indices = prepared
        self.sizes = [len(pathway) for pathway in prepared]
        self.unique_sizes = sorted(set(self.sizes))
        self.rust_runner = GseaPrerankedRunner(prepared, min_size, max_size)
        self._warned_ties = False
        self._nes_cache_key: Optional[tuple] = None
        self._nes_cache: Optional[Dict[int, Sequence[float]]] = None

    @property
    def prepared_gene_ids(self) -> Optional[np.ndarray]:
        if self._gene_ids is None:
            return None
        if self._score_order is None:
            return self._gene_ids.copy()
        return self._gene_ids[self._score_order]

    def prepare_scores(self, scores: np.ndarray) -> np.ndarray:
        scores = _validate_score_vector(scores)
        if self._gene_ids is not None and scores.size != self._gene_ids.size:
            raise ValueError("scores and gene_ids must have equal length")
        if any(pathway and pathway[-1] >= scores.size for pathway in self.pathway_indices):
            raise ValueError("pathway index is outside scores")
        if any(size == scores.size for size in self.sizes):
            raise ValueError("pathway size cannot equal the ranked gene count")
        has_ties = _warn_or_error_ties(
            scores,
            self.tie_policy,
            gene_ids_available=self._gene_ids is not None,
            warn=not self._warned_ties,
        )
        self._warned_ties = self._warned_ties or has_ties
        if self._score_order is not None:
            scores = scores[self._score_order]
        return np.ascontiguousarray(scores, dtype=np.float64)

    def run(
        self,
        scores: np.ndarray,
        sample_size: int = 101,
        seed: int = 42,
        nperm_nes: int = 500,
        gsea_param: float = 1.0,
        eps: float = 1e-50,
        score_type: ScoreType = "std",
        calculate_nes: bool = True,
        bin_width: Optional[int] = 0,
        precheck_n: Optional[int] = None,
        precheck_eps: Optional[float] = None,
        use_nes_cache: bool = False,
        mode: Mode = "aligned",
        nperm_simple: Optional[int] = None,
        max_levels: Optional[int] = None,
        tie_policy: Optional[TiePolicy] = None,
    ) -> pd.DataFrame:
        if tie_policy is not None and _normalize_tie_policy(tie_policy) != self.tie_policy:
            raise ValueError(
                "tie_policy is fixed when GseaRunner is constructed; create a new "
                "runner to use a different policy"
            )
        if not isinstance(use_nes_cache, (bool, np.bool_)):
            raise TypeError("use_nes_cache must be boolean")
        options = _normalize_run_options(
            sample_size=sample_size,
            seed=seed,
            nperm_nes=nperm_nes,
            gsea_param=gsea_param,
            eps=eps,
            score_type=score_type,
            mode=mode,
            tie_policy=self.tie_policy,
            bin_width=bin_width,
            calculate_nes=calculate_nes,
            nperm_simple=nperm_simple,
            max_levels=max_levels,
            precheck_n=precheck_n,
            precheck_eps=precheck_eps,
        )
        scores = self.prepare_scores(scores)
        _validate_binning(options["bin_width"], self.sizes, len(scores))
        cache_ranking_hash = _cache_ranking_hash(scores)

        mean_lookup: Dict[int, Sequence[float]] = {}
        if options["calculate_nes"]:
            cache_key = (
                cache_ranking_hash,
                tuple(self.unique_sizes),
                options["gsea_param"],
                options["nperm_nes"],
                options["score_type"],
                options["mode"],
                options["seed"],
                _algorithm_revision(),
            )
            if (
                use_nes_cache
                and self._nes_cache_key == cache_key
                and self._nes_cache is not None
            ):
                mean_lookup = self._nes_cache
            else:
                means_vec = get_random_es_means(
                    scores,
                    self.unique_sizes,
                    options["nperm_nes"],
                    options["seed"],
                    options["gsea_param"],
                    options["score_type"],
                )
                if not (
                    isinstance(means_vec, list)
                    and len(means_vec) == len(self.unique_sizes)
                    and all(len(means) == 2 for means in means_vec)
                ):
                    raise ValueError(
                        "get_random_es_means returned invalid format; expected one "
                        "(positive_mean, negative_mean) pair per pathway size"
                    )
                mean_lookup = dict(zip(self.unique_sizes, means_vec))
                if use_nes_cache:
                    self._nes_cache_key = cache_key
                    self._nes_cache = mean_lookup

        multi_results = self.rust_runner.run(
            scores,
            options["sample_size"],
            options["seed"],
            options["gsea_param"],
            options["eps"],
            options["score_type"],
            options["bin_width"],
            options["precheck_n"],
            options["precheck_eps"],
            options["mode"],
            options["nperm_simple"],
            options["max_levels"],
        )
        if len(multi_results) != len(self.pathway_names):
            raise RuntimeError("Rust core returned a result count that does not match pathways")
        result = _format_results(
            self.pathway_names,
            self.sizes,
            multi_results,
            mean_lookup,
            calculate_nes=options["calculate_nes"],
            eps=options["eps"],
            mode=options["mode"],
            score_type=options["score_type"],
            bin_width=options["bin_width"],
            ranking_hash=cache_ranking_hash,
        )
        result = _add_bh_adjustment(result, options["eps"])
        public_ranking_hash = _public_ranking_hash(result, cache_ranking_hash)
        result.attrs["params"] = {
            **options,
            "use_nes_cache": bool(use_nes_cache),
            "algorithm_revision": _algorithm_revision(),
            "ranking_hash": public_ranking_hash,
            "cache_ranking_hash": cache_ranking_hash,
        }
        return result


def run_gsea(
    data: Union[pd.DataFrame, pd.Series],
    gmt: Union[str, GeneSetDict],
    gene_col: Union[str, int] = 0,
    score_col: Union[str, int] = 1,
    min_size: int = 15,
    max_size: int = 500,
    sample_size: int = 101,
    seed: int = 42,
    nperm_nes: int = 500,
    gsea_param: float = 1.0,
    eps: float = 1e-50,
    dedup_genes: str = "max_abs",
    score_type: ScoreType = "std",
    use_batched: bool = True,
    bin_width: Optional[int] = 0,
    calculate_nes: bool = True,
    mode: Mode = "aligned",
    tie_policy: TiePolicy = "gene_id",
    nperm_simple: Optional[int] = None,
    max_levels: Optional[int] = None,
    precheck_n: Optional[int] = None,
    precheck_eps: Optional[float] = None,
) -> pd.DataFrame:
    """Run preranked GSEA with current fgsea-aligned semantics by default."""
    options = _normalize_run_options(
        sample_size=sample_size,
        seed=seed,
        nperm_nes=nperm_nes,
        gsea_param=gsea_param,
        eps=eps,
        score_type=score_type,
        mode=mode,
        tie_policy=tie_policy,
        bin_width=bin_width,
        calculate_nes=calculate_nes,
        nperm_simple=nperm_simple,
        max_levels=max_levels,
        precheck_n=precheck_n,
        precheck_eps=precheck_eps,
    )
    if not isinstance(use_batched, (bool, np.bool_)):
        raise TypeError("use_batched must be boolean")
    if not use_batched and options["bin_width"] > 0:
        raise ValueError("bin_width requires use_batched=True")

    if isinstance(data, pd.Series):
        df = data.reset_index()
        df.columns = ["Gene", "Score"]
        gene_col = "Gene"
        score_col = "Score"
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("data must be a pandas DataFrame or Series")
    if df.empty:
        raise ValueError("scores must be non-empty")

    try:
        score_c = df.columns[score_col] if isinstance(score_col, int) else score_col
        gene_c = df.columns[gene_col] if isinstance(gene_col, int) else gene_col
        numeric_scores = pd.to_numeric(df[score_c], errors="raise").astype(float)
        gene_values = df[gene_c]
    except (IndexError, KeyError) as exc:
        raise ValueError("gene_col or score_col does not identify an input column") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError("score column must contain numeric values") from exc
    if gene_values.isna().any():
        raise ValueError("gene identifiers must not be missing")
    df[score_c] = numeric_scores
    df[gene_c] = gene_values.astype(str)
    if df[gene_c].str.strip().eq("").any():
        raise ValueError("gene identifiers must not be empty")
    _validate_score_vector(df[score_c].to_numpy(dtype=float))

    dedup_genes = str(dedup_genes).lower().replace("-", "_")
    if dedup_genes not in _VALID_DEDUP_POLICIES:
        raise ValueError("dedup_genes must be 'max_abs', 'first', or 'error'")
    duplicated = df[gene_c].duplicated(keep=False)
    if duplicated.any() and dedup_genes == "error":
        examples = ", ".join(df.loc[duplicated, gene_c].astype(str).head(5))
        raise ValueError(f"duplicate gene identifiers are not allowed. Examples: {examples}")
    if duplicated.any() and dedup_genes == "max_abs":
        df = df.assign(
            __pyfgsea_abs_score=df[score_c].abs(),
            __pyfgsea_input_order=np.arange(len(df)),
        )
        df = df.sort_values(
            ["__pyfgsea_abs_score", score_c, "__pyfgsea_input_order"],
            ascending=[False, False, True],
            kind="mergesort",
        ).drop_duplicates(subset=gene_c, keep="first")
        df = df.drop(columns=["__pyfgsea_abs_score", "__pyfgsea_input_order"])
    elif duplicated.any() and dedup_genes == "first":
        df = df.drop_duplicates(subset=gene_c, keep="first")

    _warn_or_error_ties(
        df[score_c].to_numpy(dtype=np.float64),
        options["tie_policy"],
        gene_ids_available=True,
    )
    if options["tie_policy"] == "gene_id":
        df = df.sort_values(
            [score_c, gene_c], ascending=[False, True], kind="mergesort"
        )
    else:
        df = df.sort_values(score_c, ascending=False, kind="mergesort")
    df = df.reset_index(drop=True)
    genes = df[gene_c].to_numpy(dtype=str)
    scores = _validate_score_vector(df[score_c].to_numpy(dtype=np.float64))
    pathway_names, pathway_indices = prepare_pathways(
        genes, gmt, min_size=min_size, max_size=max_size
    )
    if not pathway_indices:
        return pd.DataFrame()
    sizes = [len(pathway) for pathway in pathway_indices]
    _validate_binning(options["bin_width"], sizes, len(scores))
    cache_ranking_hash = _cache_ranking_hash(scores)

    mean_lookup: Dict[int, Sequence[float]] = {}
    if options["calculate_nes"]:
        unique_sizes = sorted(set(sizes))
        means_vec = get_random_es_means(
            scores,
            unique_sizes,
            options["nperm_nes"],
            options["seed"],
            options["gsea_param"],
            options["score_type"],
        )
        if not (
            isinstance(means_vec, list)
            and len(means_vec) == len(unique_sizes)
            and all(len(means) == 2 for means in means_vec)
        ):
            raise ValueError(
                "get_random_es_means returned invalid format; expected one "
                "(positive_mean, negative_mean) pair per pathway size"
            )
        mean_lookup = dict(zip(unique_sizes, means_vec))

    if use_batched:
        multi_results = fgsea_multilevel_batched(
            scores,
            pathway_indices,
            options["sample_size"],
            options["seed"],
            options["gsea_param"],
            options["eps"],
            options["score_type"],
            options["bin_width"],
            options["mode"],
            options["nperm_simple"],
            options["max_levels"],
            options["precheck_n"],
            options["precheck_eps"],
        )
    else:
        multi_results = fgsea_multilevel(
            scores,
            pathway_indices,
            options["sample_size"],
            options["seed"],
            options["gsea_param"],
            options["eps"],
            options["score_type"],
            options["mode"],
            options["nperm_simple"],
            options["max_levels"],
            options["precheck_n"],
            options["precheck_eps"],
        )
    if len(multi_results) != len(pathway_names):
        raise RuntimeError("Rust core returned a result count that does not match pathways")
    result = _format_results(
        pathway_names,
        sizes,
        multi_results,
        mean_lookup,
        calculate_nes=options["calculate_nes"],
        eps=options["eps"],
        mode=options["mode"],
        score_type=options["score_type"],
        bin_width=options["bin_width"],
        ranking_hash=cache_ranking_hash,
    )
    result = _add_bh_adjustment(result, options["eps"])
    public_ranking_hash = _public_ranking_hash(result, cache_ranking_hash)
    result.attrs["params"] = {
        **options,
        "dedup_genes": dedup_genes,
        "min_size": int(min_size),
        "max_size": int(max_size),
        "use_batched": bool(use_batched),
        "algorithm_revision": _algorithm_revision(),
        "ranking_hash": public_ranking_hash,
        "cache_ranking_hash": cache_ranking_hash,
    }
    return result
