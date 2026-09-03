import unittest

import numpy as np

from ginseng_benchmark.cache import FeatureCache
from ginseng_benchmark.query_groups import _query_protocol_sha256
from ginseng_benchmark.ssi import compute_group_ssi


def _protocol():
    groups = [
        {
            "group_id": "a",
            "name": "a",
            "query_image": "a1.jpg",
            "same_ginsengs": ["a2.jpg"],
        },
        {
            "group_id": "a",
            "name": "a",
            "query_image": "a2.jpg",
            "same_ginsengs": ["a1.jpg"],
        },
        {
            "group_id": "b",
            "name": "b",
            "query_image": "b1.jpg",
            "same_ginsengs": ["b2.jpg"],
        },
        {
            "group_id": "b",
            "name": "b",
            "query_image": "b2.jpg",
            "same_ginsengs": ["b1.jpg"],
        },
    ]
    return {
        "metadata": {
            "dataset_manifest_sha256": "a" * 64,
            "group_count": 2,
            "query_count": 4,
            "gallery_count": 4,
            "positive_count_distribution": {"1": 4},
            "query_protocol_sha256": _query_protocol_sha256(groups),
        },
        "query_groups": groups,
    }


class SsiAnalysisTest(unittest.TestCase):
    def test_group_ssi_uses_all_views_and_maps_cosine_to_unit_interval(self):
        cache = FeatureCache(
            features=np.asarray(
                [
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            paths=("a1.jpg", "a2.jpg", "b1.jpg", "b2.jpg"),
            metadata={
                "dataset_manifest_sha256": "a" * 64,
                "num_images": 4,
            },
        )
        rows = compute_group_ssi(cache, _protocol())
        by_group = {row["group_id"]: row for row in rows}
        self.assertEqual(by_group["a"]["num_images"], 2)
        self.assertAlmostEqual(by_group["a"]["ssi"], 1.0)
        self.assertAlmostEqual(by_group["b"]["ssi"], 0.5)


if __name__ == "__main__":
    unittest.main()
