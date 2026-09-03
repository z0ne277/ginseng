"""Pure, JSON-serializable metrics for instance-retrieval evaluation.

Callers must pass the complete ranked gallery to :func:`metrics_for_ranking`
when reporting AP and MRR.  This module deliberately has no model or tensor
dependencies so the metric contract is shared by every baseline environment.
"""

import math
import random
from numbers import Real
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union


MetricValue = Union[float, int, None]


METRIC_DEFINITIONS = {
    "recall_fraction": (
        "recall@k = relevant items in the top-k divided by all "
        "relevant items for that query."
    ),
    "hit_cmc": (
        "hit@k and cmc@k = 1 when the top-k contains at least one "
        "relevant item, otherwise 0."
    ),
    "full_ranking_ap_mrr": (
        "AP and MRR are computed from the complete ranked list; "
        "missing relevant items contribute zero to AP."
    ),
}


def metric_definition_metadata() -> Dict[str, str]:
    """Return an independent copy suitable for embedding in result JSON."""

    return dict(METRIC_DEFINITIONS)


def _normalized_ks(ks: Iterable[int]) -> Sequence[int]:
    normalized = []
    seen = set()
    for k in ks:
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("each k must be a positive integer")
        if k not in seen:
            normalized.append(k)
            seen.add(k)
    return normalized


def _validated_ids(value: object, parameter_name: str) -> List[str]:
    """Materialize an ID iterable while enforcing the public input contract."""

    message = f"{parameter_name} must be an iterable of non-empty strings"
    if isinstance(value, (str, bytes)):
        raise ValueError(message)
    try:
        iterator = iter(value)
    except TypeError as exc:
        raise ValueError(message) from exc

    result = []
    try:
        for item in iterator:
            if not isinstance(item, str) or not item:
                raise ValueError(message)
            result.append(item)
    except TypeError as exc:
        raise ValueError(message) from exc
    return result


def metrics_for_ranking(
    ranked_ids: Iterable[str],
    relevant_ids: Iterable[str],
    ks: Iterable[int] = (1, 5, 10, 20),
) -> Dict[str, MetricValue]:
    """Compute per-query retrieval metrics from a complete ranking.

    Recall uses the project's retained fractional definition: relevant items
    retrieved in the top-k divided by the total number of relevant items.
    ``hit@k`` and ``cmc@k`` are the corresponding at-least-one-hit indicators.
    Duplicate relevant IDs are collapsed; duplicate ranked IDs are rejected.
    Relevant items absent from the ranking contribute zero to AP and recall.
    """

    ranking = _validated_ids(ranked_ids, "ranked_ids")
    if len(set(ranking)) != len(ranking):
        raise ValueError("ranked_ids contains duplicate IDs")

    relevant = set(_validated_ids(relevant_ids, "relevant_ids"))
    if not relevant:
        raise ValueError("relevant_ids must contain at least one ID")

    normalized_ks = _normalized_ks(ks)
    relevant_count = len(relevant)
    hits_seen = 0
    precision_sum = 0.0
    first_relevant_rank: Optional[int] = None

    for rank, item_id in enumerate(ranking, start=1):
        if item_id in relevant:
            hits_seen += 1
            precision_sum += hits_seen / rank
            if first_relevant_rank is None:
                first_relevant_rank = rank

    result: Dict[str, MetricValue] = {
        "mrr": 0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank,
        "map": precision_sum / relevant_count,
        "first_relevant_rank": first_relevant_rank,
    }
    for k in normalized_ks:
        top_k_hits = sum(item_id in relevant for item_id in ranking[:k])
        hit = 1.0 if top_k_hits else 0.0
        result[f"recall@{k}"] = top_k_hits / relevant_count
        result[f"hit@{k}"] = hit
        result[f"cmc@{k}"] = hit

    return result


def _finite_number(value: object, metric_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"metric '{metric_name}' must be a finite number")
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            f"metric '{metric_name}' must be a finite number"
        ) from exc
    if not math.isfinite(numeric):
        raise ValueError(f"metric '{metric_name}' must be a finite number")
    return numeric


def _stable_mean(values: Sequence[float], result_name: str) -> float:
    """Compute a finite mean without overflowing the intermediate sum."""

    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0
    result = math.fsum(value / scale for value in values) / len(values) * scale
    if not math.isfinite(result):
        raise ValueError(f"{result_name} produced a non-finite result")
    return result


def aggregate_query_metrics(
    per_query: Sequence[Mapping[str, MetricValue]],
) -> Dict[str, object]:
    """Macro-average a non-empty set of per-query metric mappings.

    Every query contributes equal weight; no micro-averaged counts are mixed
    into ``macro``.  If ``first_relevant_rank`` is present, its mean and median
    are computed over successful queries only, while ``no_hit_count`` makes the
    excluded failures explicit.  All mappings must expose the same keys.
    """

    queries = list(per_query)
    if not queries:
        raise ValueError("per_query must be non-empty")

    first_keys = list(queries[0].keys())
    expected_keys = set(first_keys)
    for query in queries[1:]:
        if set(query.keys()) != expected_keys:
            raise ValueError("all queries must have the same metric keys")

    rank_key = "first_relevant_rank"
    macro_values: Dict[str, list] = {
        key: [] for key in first_keys if key != rank_key
    }
    successful_ranks = []

    for query in queries:
        for key in macro_values:
            macro_values[key].append(_finite_number(query[key], key))

        if rank_key in expected_keys:
            rank = query[rank_key]
            if rank is None:
                continue
            if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
                raise ValueError(
                    "first_relevant_rank must be a positive integer or None"
                )
            try:
                successful_ranks.append(_finite_number(rank, rank_key))
            except ValueError as exc:
                raise ValueError(
                    "first_relevant_rank must be a finite positive integer or None"
                ) from exc

    result: Dict[str, object] = {
        "query_count": len(queries),
        "macro": {
            key: _stable_mean(values, f"macro metric '{key}'")
            for key, values in macro_values.items()
        },
    }
    if rank_key in expected_keys:
        result[rank_key] = {
            "mean": (
                _stable_mean(successful_ranks, "first_relevant_rank mean")
                if successful_ranks
                else None
            ),
            "median": (
                _percentile(sorted(successful_ranks), 0.5)
                if successful_ranks
                else None
            ),
            "hit_count": len(successful_ranks),
            "no_hit_count": len(queries) - len(successful_ranks),
        }
    return result


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated percentile (the R-7 convention)."""

    position = (len(sorted_values) - 1) * probability
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return sorted_values[lower_index]
    weight = position - lower_index
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    scale = max(abs(lower_value), abs(upper_value))
    if scale == 0.0:
        return 0.0
    result = math.fsum(
        (
            lower_value / scale * (1.0 - weight),
            upper_value / scale * weight,
        )
    ) * scale
    if not math.isfinite(result):
        raise ValueError("percentile interpolation produced a non-finite result")
    return result


def bootstrap_confidence_intervals(
    per_query: Sequence[Mapping[str, MetricValue]],
    metric_keys: Sequence[str],
    iterations: int = 2000,
    seed: int = 42,
    cluster_ids: Optional[Sequence[object]] = None,
    confidence: float = 0.95,
) -> Dict[str, object]:
    """Estimate percentile confidence intervals by query or identity cluster.

    Without ``cluster_ids``, each replicate draws ``n`` queries with
    replacement for 2,000 replicates by default.  With cluster IDs, it draws
    the original number of clusters with replacement and includes *every
    query* belonging to each selected cluster.  This preserves within-ginseng
    multi-view dependence.  Point estimates remain the macro means over the
    original queries.
    """

    queries = list(per_query)
    if not queries:
        raise ValueError("per_query must be non-empty")

    keys = list(metric_keys)
    if (
        not keys
        or any(not isinstance(key, str) for key in keys)
        or len(set(keys)) != len(keys)
    ):
        raise ValueError("metric_keys must be non-empty unique strings")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 2:
        raise ValueError("iterations must be an integer of at least 2")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(confidence, bool) or not isinstance(confidence, Real):
        raise ValueError("confidence must be a finite number between 0 and 1")
    try:
        confidence_value = float(confidence)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            "confidence must be a finite number between 0 and 1"
        ) from exc
    if not math.isfinite(confidence_value) or not 0.0 < confidence_value < 1.0:
        raise ValueError("confidence must be a finite number between 0 and 1")

    values_by_metric: Dict[str, List[float]] = {key: [] for key in keys}
    for query in queries:
        for key in keys:
            if key not in query:
                raise ValueError(f"query is missing metric '{key}'")
            values_by_metric[key].append(_finite_number(query[key], key))

    groups: Optional[List[List[int]]] = None
    if cluster_ids is not None:
        clusters = list(cluster_ids)
        if len(clusters) != len(queries):
            raise ValueError("cluster_ids length must equal per_query length")
        grouped_indices: Dict[object, List[int]] = {}
        for query_index, cluster_id in enumerate(clusters):
            try:
                hash(cluster_id)
            except TypeError as exc:
                raise ValueError("cluster_ids must be hashable") from exc
            grouped_indices.setdefault(cluster_id, []).append(query_index)
        groups = list(grouped_indices.values())

    rng = random.Random(seed)
    replicates: Dict[str, List[float]] = {key: [] for key in keys}
    query_count = len(queries)
    for _ in range(iterations):
        if groups is None:
            sampled_indices = [
                rng.randrange(query_count) for _ in range(query_count)
            ]
        else:
            sampled_indices = []
            for _ in range(len(groups)):
                sampled_indices.extend(groups[rng.randrange(len(groups))])

        sampled_count = len(sampled_indices)
        for key in keys:
            sampled_mean = _stable_mean(
                [values_by_metric[key][index] for index in sampled_indices],
                f"bootstrap replicate for metric '{key}'",
            )
            replicates[key].append(sampled_mean)

    tail_probability = (1.0 - confidence_value) / 2.0
    metric_results: Dict[str, Dict[str, float]] = {}
    for key in keys:
        sorted_replicates = sorted(replicates[key])
        point_estimate = _stable_mean(
            values_by_metric[key], f"point estimate for metric '{key}'"
        )
        lower = _percentile(sorted_replicates, tail_probability)
        upper = _percentile(sorted_replicates, 1.0 - tail_probability)
        if not all(
            math.isfinite(value) for value in (point_estimate, lower, upper)
        ):
            raise ValueError(f"confidence interval for metric '{key}' is non-finite")
        metric_results[key] = {
            "point_estimate": point_estimate,
            "lower": lower,
            "upper": upper,
        }

    return {
        "sampling_unit": "cluster" if groups is not None else "query",
        "iterations": iterations,
        "seed": seed,
        "confidence": confidence_value,
        "metrics": metric_results,
    }
