import unittest

from scripts.paired_bootstrap import paired_cluster_bootstrap


class PairedBootstrapTest(unittest.TestCase):
    def test_cluster_bootstrap_reports_paired_improvement(self):
        baseline = [
            {"group_id": "1", "query_image": "a.jpg", "map": 0.2, "mrr": 0.4},
            {"group_id": "1", "query_image": "b.jpg", "map": 0.4, "mrr": 0.6},
            {"group_id": "2", "query_image": "c.jpg", "map": 0.3, "mrr": 0.5},
            {"group_id": "2", "query_image": "d.jpg", "map": 0.5, "mrr": 0.7},
        ]
        challenger = [
            {"group_id": "1", "query_image": "a.jpg", "map": 0.4, "mrr": 0.5},
            {"group_id": "1", "query_image": "b.jpg", "map": 0.6, "mrr": 0.7},
            {"group_id": "2", "query_image": "c.jpg", "map": 0.5, "mrr": 0.6},
            {"group_id": "2", "query_image": "d.jpg", "map": 0.7, "mrr": 0.8},
        ]

        result = paired_cluster_bootstrap(
            baseline,
            challenger,
            metrics=("map", "mrr"),
            iterations=200,
            seed=7,
        )

        self.assertEqual(result["group_count"], 2)
        self.assertEqual(result["query_count"], 4)
        self.assertAlmostEqual(result["metrics"]["map"]["difference"], 0.2)
        self.assertAlmostEqual(result["metrics"]["mrr"]["difference"], 0.1)
        self.assertGreater(result["metrics"]["map"]["ci_lower"], 0)
        self.assertGreater(result["metrics"]["mrr"]["ci_lower"], 0)

    def test_rejects_misaligned_queries(self):
        baseline = [{"group_id": "1", "query_image": "a.jpg", "map": 0.2}]
        challenger = [{"group_id": "1", "query_image": "b.jpg", "map": 0.3}]
        with self.assertRaisesRegex(ValueError, "query sets"):
            paired_cluster_bootstrap(
                baseline,
                challenger,
                metrics=("map",),
                iterations=20,
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
