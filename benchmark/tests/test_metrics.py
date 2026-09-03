"""Unit tests for retrieval metric definitions and per-query metrics."""

import json
import unittest

from ginseng_benchmark.metrics import (
    METRIC_DEFINITIONS,
    aggregate_query_metrics,
    bootstrap_confidence_intervals,
    metric_definition_metadata,
    metrics_for_ranking,
)


class MetricsForRankingTests(unittest.TestCase):
    def test_computes_fractional_recall_and_full_ranking_ap_mrr(self):
        result = metrics_for_ranking(
            ["p1", "n1", "p2"],
            ["p1", "p2"],
            ks=(1, 2, 3),
        )

        self.assertEqual(result["first_relevant_rank"], 1)
        self.assertAlmostEqual(result["mrr"], 1.0)
        self.assertAlmostEqual(result["map"], (1.0 + 2.0 / 3.0) / 2.0)
        self.assertAlmostEqual(result["recall@1"], 0.5)
        self.assertAlmostEqual(result["recall@2"], 0.5)
        self.assertAlmostEqual(result["recall@3"], 1.0)
        self.assertEqual(result["hit@1"], 1.0)
        self.assertEqual(result["cmc@1"], 1.0)

    def test_no_hit_returns_zero_metrics_and_no_first_rank(self):
        result = metrics_for_ranking(["n1", "n2"], {"p1"}, ks=(1, 5))

        self.assertEqual(
            result,
            {
                "mrr": 0.0,
                "map": 0.0,
                "first_relevant_rank": None,
                "recall@1": 0.0,
                "hit@1": 0.0,
                "cmc@1": 0.0,
                "recall@5": 0.0,
                "hit@5": 0.0,
                "cmc@5": 0.0,
            },
        )

    def test_missing_relevant_items_contribute_zero_to_ap_and_recall(self):
        result = metrics_for_ranking(["n1", "p1"], ["p1", "p2"], ks=(2,))

        self.assertAlmostEqual(result["mrr"], 0.5)
        self.assertAlmostEqual(result["map"], 0.25)
        self.assertAlmostEqual(result["recall@2"], 0.5)
        self.assertEqual(result["hit@2"], 1.0)

    def test_duplicate_relevant_ids_do_not_change_denominator(self):
        unique = metrics_for_ranking(["p1", "p2"], ["p1", "p2"], ks=(2,))
        duplicate = metrics_for_ranking(
            ["p1", "p2"], ["p1", "p1", "p2", "p2"], ks=(2,)
        )

        self.assertEqual(duplicate, unique)

    def test_duplicate_ks_are_deduplicated_in_first_seen_order(self):
        result = metrics_for_ranking(["p1"], ["p1"], ks=(5, 1, 5, 1))

        self.assertEqual(
            list(result),
            [
                "mrr",
                "map",
                "first_relevant_rank",
                "recall@5",
                "hit@5",
                "cmc@5",
                "recall@1",
                "hit@1",
                "cmc@1",
            ],
        )
        self.assertEqual(result["recall@5"], 1.0)

    def test_k_larger_than_ranking_uses_available_items(self):
        result = metrics_for_ranking(["n1", "p1"], ["p1", "p2"], ks=(20,))

        self.assertEqual(result["recall@20"], 0.5)
        self.assertEqual(result["hit@20"], 1.0)
        self.assertEqual(result["cmc@20"], 1.0)

    def test_rejects_duplicate_ranked_ids(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            metrics_for_ranking(["p1", "p1"], ["p1"])

    def test_rejects_empty_relevant_ids(self):
        with self.assertRaisesRegex(ValueError, "relevant_ids"):
            metrics_for_ranking(["n1"], [])

    def test_rejects_non_positive_non_integer_and_boolean_k(self):
        invalid_values = (0, -1, 1.5, "1", True)

        for invalid in invalid_values:
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    metrics_for_ranking(["p1"], ["p1"], ks=(invalid,))

    def test_accepts_generators_for_both_id_collections(self):
        ranked = (item for item in ["p1", "n1", "p2"])
        relevant = (item for item in ["p1", "p2"])

        result = metrics_for_ranking(ranked, relevant, ks=(3,))

        self.assertEqual(result["recall@3"], 1.0)

    def test_rejects_invalid_ranked_id_collection_and_items_consistently(self):
        invalid_inputs = (
            "p1",
            "",
            b"p1",
            123,
            [""],
            [1],
            [["unhashable"]],
        )

        for invalid in invalid_inputs:
            with self.subTest(invalid=repr(invalid)):
                with self.assertRaisesRegex(ValueError, "ranked_ids"):
                    metrics_for_ranking(invalid, ["p1"])

    def test_rejects_invalid_relevant_id_collection_and_items_consistently(self):
        invalid_inputs = (
            "p1",
            "",
            b"p1",
            123,
            [""],
            [1],
            [["unhashable"]],
        )

        for invalid in invalid_inputs:
            with self.subTest(invalid=repr(invalid)):
                with self.assertRaisesRegex(ValueError, "relevant_ids"):
                    metrics_for_ranking(["p1"], invalid)


class MetricDefinitionTests(unittest.TestCase):
    def test_definition_strings_are_stable_and_explicit(self):
        self.assertEqual(
            METRIC_DEFINITIONS,
            {
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
            },
        )

    def test_metadata_helper_returns_json_safe_independent_copy(self):
        metadata = metric_definition_metadata()
        metadata["recall_fraction"] = "changed"

        self.assertEqual(
            metric_definition_metadata()["recall_fraction"],
            METRIC_DEFINITIONS["recall_fraction"],
        )
        self.assertTrue(all(isinstance(value, str) for value in metadata.values()))
        json.dumps(metadata, allow_nan=False)


class AggregateQueryMetricsTests(unittest.TestCase):
    def test_macro_averages_queries_and_summarizes_successful_first_ranks(self):
        result = aggregate_query_metrics(
            [
                {
                    "mrr": 1.0,
                    "map": 0.75,
                    "first_relevant_rank": 1,
                    "recall@1": 0.5,
                    "hit@1": 1.0,
                    "cmc@1": 1.0,
                },
                {
                    "mrr": 0.25,
                    "map": 0.25,
                    "first_relevant_rank": 4,
                    "recall@1": 0.0,
                    "hit@1": 0.0,
                    "cmc@1": 0.0,
                },
                {
                    "mrr": 0.0,
                    "map": 0.0,
                    "first_relevant_rank": None,
                    "recall@1": 0.0,
                    "hit@1": 0.0,
                    "cmc@1": 0.0,
                },
            ]
        )

        self.assertEqual(result["query_count"], 3)
        self.assertAlmostEqual(result["macro"]["mrr"], 1.25 / 3.0)
        self.assertAlmostEqual(result["macro"]["map"], 1.0 / 3.0)
        self.assertAlmostEqual(result["macro"]["recall@1"], 1.0 / 6.0)
        self.assertEqual(result["macro"]["hit@1"], 1.0 / 3.0)
        self.assertEqual(result["macro"]["cmc@1"], 1.0 / 3.0)
        self.assertEqual(
            result["first_relevant_rank"],
            {
                "mean": 2.5,
                "median": 2.5,
                "hit_count": 2,
                "no_hit_count": 1,
            },
        )
        json.dumps(result, allow_nan=False)

    def test_large_finite_values_have_stable_finite_macro_mean(self):
        same_sign = aggregate_query_metrics(
            [{"score": 1e308}, {"score": 1e308}]
        )
        opposite_sign = aggregate_query_metrics(
            [{"score": 1e308}, {"score": -1e308}]
        )

        self.assertEqual(same_sign["macro"]["score"], 1e308)
        self.assertEqual(opposite_sign["macro"]["score"], 0.0)
        json.dumps(same_sign, allow_nan=False)
        json.dumps(opposite_sign, allow_nan=False)

    def test_integer_too_large_for_float_is_reported_as_value_error(self):
        with self.assertRaisesRegex(ValueError, "finite number"):
            aggregate_query_metrics([{"score": 10**10000}])

    def test_all_no_hit_queries_report_null_rank_statistics(self):
        result = aggregate_query_metrics(
            [
                {"mrr": 0.0, "first_relevant_rank": None},
                {"mrr": 0.0, "first_relevant_rank": None},
            ]
        )

        self.assertEqual(
            result["first_relevant_rank"],
            {
                "mean": None,
                "median": None,
                "hit_count": 0,
                "no_hit_count": 2,
            },
        )

    def test_rejects_empty_query_collection(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            aggregate_query_metrics([])

    def test_rejects_different_metric_keys(self):
        with self.assertRaisesRegex(ValueError, "same metric keys"):
            aggregate_query_metrics([{"mrr": 1.0}, {"map": 1.0}])

    def test_rejects_non_finite_or_non_numeric_metric_values(self):
        invalid_values = (float("nan"), float("inf"), -float("inf"), "1", True)

        for invalid in invalid_values:
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "finite number"):
                    aggregate_query_metrics([{"mrr": invalid}])

    def test_rejects_invalid_first_relevant_rank(self):
        invalid_values = (0, -1, 1.5, float("inf"), "1", True)

        for invalid in invalid_values:
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "first_relevant_rank"):
                    aggregate_query_metrics(
                        [{"mrr": 1.0, "first_relevant_rank": invalid}]
                    )


class BootstrapConfidenceIntervalTests(unittest.TestCase):
    def test_default_iteration_count_is_two_thousand(self):
        result = bootstrap_confidence_intervals([{"map": 1.0}], ["map"])

        self.assertEqual(result["iterations"], 2000)

    def test_query_bootstrap_has_hand_checked_percentile_interval(self):
        result = bootstrap_confidence_intervals(
            [{"map": 0.0}, {"map": 1.0}],
            ["map"],
            iterations=4,
            seed=1,
            confidence=0.5,
        )

        self.assertEqual(result["sampling_unit"], "query")
        self.assertEqual(result["iterations"], 4)
        self.assertEqual(result["seed"], 1)
        self.assertEqual(result["confidence"], 0.5)
        self.assertEqual(
            result["metrics"]["map"],
            {"point_estimate": 0.5, "lower": 0.375, "upper": 1.0},
        )
        json.dumps(result, allow_nan=False)

    def test_large_finite_values_keep_all_bootstrap_outputs_finite(self):
        same_sign = bootstrap_confidence_intervals(
            [{"map": 1e308}, {"map": 1e308}],
            ["map"],
            iterations=8,
            seed=1,
            confidence=0.5,
        )
        opposite_sign = bootstrap_confidence_intervals(
            [{"map": 1e308}, {"map": -1e308}],
            ["map"],
            iterations=8,
            seed=1,
            confidence=0.5,
        )

        self.assertEqual(
            same_sign["metrics"]["map"],
            {"point_estimate": 1e308, "lower": 1e308, "upper": 1e308},
        )
        self.assertEqual(opposite_sign["metrics"]["map"]["point_estimate"], 0.0)
        json.dumps(same_sign, allow_nan=False)
        json.dumps(opposite_sign, allow_nan=False)

    def test_overflowing_float_conversion_is_reported_as_value_error(self):
        with self.assertRaisesRegex(ValueError, "finite number"):
            bootstrap_confidence_intervals(
                [{"map": 10**10000}], ["map"], iterations=2
            )

    def test_overflowing_confidence_conversion_is_reported_as_value_error(self):
        with self.assertRaisesRegex(ValueError, "confidence"):
            bootstrap_confidence_intervals(
                [{"map": 1.0}], ["map"], confidence=10**10000
            )

    def test_same_seed_is_deterministic_for_multiple_metrics(self):
        queries = [
            {"map": 0.1, "mrr": 0.2},
            {"map": 0.5, "mrr": 0.8},
            {"map": 0.9, "mrr": 1.0},
        ]

        first = bootstrap_confidence_intervals(
            queries, ["map", "mrr"], iterations=37, seed=42
        )
        second = bootstrap_confidence_intervals(
            queries, ["map", "mrr"], iterations=37, seed=42
        )

        self.assertEqual(first, second)

    def test_cluster_bootstrap_resamples_whole_identity_groups(self):
        result = bootstrap_confidence_intervals(
            [{"map": 0.0}, {"map": 0.0}, {"map": 1.0}],
            ["map"],
            iterations=8,
            seed=1,
            cluster_ids=["same-root", "same-root", "other-root"],
            confidence=0.5,
        )

        self.assertEqual(result["sampling_unit"], "cluster")
        self.assertEqual(
            result["metrics"]["map"],
            {
                "point_estimate": 1.0 / 3.0,
                "lower": 0.25,
                "upper": 1.0,
            },
        )

    def test_rejects_non_finite_selected_metric_values(self):
        for invalid in (float("nan"), float("inf"), -float("inf"), None, "1", True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "finite number"):
                    bootstrap_confidence_intervals(
                        [{"map": invalid}], ["map"], iterations=2
                    )

    def test_rejects_missing_selected_metric(self):
        with self.assertRaisesRegex(ValueError, "missing metric 'map'"):
            bootstrap_confidence_intervals([{"mrr": 1.0}], ["map"])

    def test_rejects_empty_queries_or_metric_keys(self):
        with self.assertRaisesRegex(ValueError, "per_query"):
            bootstrap_confidence_intervals([], ["map"])
        with self.assertRaisesRegex(ValueError, "metric_keys"):
            bootstrap_confidence_intervals([{"map": 1.0}], [])

    def test_rejects_duplicate_or_non_string_metric_keys(self):
        with self.assertRaisesRegex(ValueError, "unique strings"):
            bootstrap_confidence_intervals([{"map": 1.0}], ["map", "map"])
        with self.assertRaisesRegex(ValueError, "unique strings"):
            bootstrap_confidence_intervals([{"map": 1.0}], [1])

    def test_rejects_iterations_below_two_or_non_integer(self):
        for invalid in (0, 1, 1.5, True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "iterations"):
                    bootstrap_confidence_intervals(
                        [{"map": 1.0}], ["map"], iterations=invalid
                    )

    def test_rejects_invalid_confidence(self):
        for invalid in (0.0, 1.0, -0.1, 1.1, float("nan"), True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "confidence"):
                    bootstrap_confidence_intervals(
                        [{"map": 1.0}], ["map"], confidence=invalid
                    )

    def test_rejects_cluster_length_mismatch_and_unhashable_ids(self):
        with self.assertRaisesRegex(ValueError, "cluster_ids length"):
            bootstrap_confidence_intervals(
                [{"map": 0.0}, {"map": 1.0}],
                ["map"],
                cluster_ids=["only-one"],
            )
        with self.assertRaisesRegex(ValueError, "hashable"):
            bootstrap_confidence_intervals(
                [{"map": 1.0}], ["map"], cluster_ids=[["not-hashable"]]
            )


if __name__ == "__main__":
    unittest.main()
