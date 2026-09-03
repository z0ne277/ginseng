import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from ginseng_benchmark.cache import FeatureCache
from ginseng_benchmark.query_groups import _query_protocol_sha256
from ginseng_benchmark.robustness import (
    apply_query_perturbation,
    evaluate_shifted_queries,
)
from scripts.prepare_robustness_queries import prepare_queries


def _toy_protocol():
    groups = [
        {
            "group_id": "1",
            "name": "1",
            "query_image": "q.jpg",
            "same_ginsengs": ["positive.jpg"],
        }
    ]
    return {
        "metadata": {
            "dataset_manifest_sha256": "a" * 64,
            "group_count": 1,
            "query_count": 1,
            "gallery_count": 3,
            "positive_count_distribution": {"1": 1},
            "query_protocol_sha256": _query_protocol_sha256(groups),
        },
        "query_groups": groups,
    }


class PerturbationTest(unittest.TestCase):
    def test_mask_erosion_and_dilation_change_foreground_in_expected_direction(self):
        image = np.zeros((21, 21, 3), dtype=np.uint8)
        image[5:16, 5:16] = 255

        eroded = apply_query_perturbation(
            image,
            kind="mask_erode",
            severity=2,
            seed=42,
        )
        dilated = apply_query_perturbation(
            image,
            kind="mask_dilate",
            severity=2,
            seed=42,
        )

        original_foreground = np.count_nonzero(image[..., 0])
        self.assertLess(np.count_nonzero(eroded[..., 0]), original_foreground)
        self.assertGreater(np.count_nonzero(dilated[..., 0]), original_foreground)

    def test_branch_occlusion_is_deterministic_for_a_fixed_seed(self):
        image = np.full((32, 32, 3), 255, dtype=np.uint8)

        first = apply_query_perturbation(
            image,
            kind="branch_occlusion",
            severity=3,
            seed=123,
        )
        second = apply_query_perturbation(
            image,
            kind="branch_occlusion",
            severity=3,
            seed=123,
        )

        np.testing.assert_array_equal(first, second)
        self.assertLess(np.count_nonzero(first), np.count_nonzero(image))


class ShiftedQueryEvaluationTest(unittest.TestCase):
    def test_uses_shifted_query_features_against_clean_gallery(self):
        features = np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        paths = ("q.jpg", "positive.jpg", "distractor.jpg")
        metadata = {
            "dataset_manifest_sha256": "a" * 64,
            "model_id": "toy",
            "model_source": "unit-test",
            "num_images": 3,
        }
        clean_cache = FeatureCache(features=features, paths=paths, metadata=metadata)
        shifted_features = np.asarray([[1.0, 0.0]], dtype=np.float32)

        result = evaluate_shifted_queries(
            clean_cache,
            shifted_features=shifted_features,
            shifted_paths=("q.jpg",),
            query_protocol=_toy_protocol(),
            condition="mask_erode_s1",
            ks=(1,),
            bootstrap_iterations=20,
            bootstrap_seed=1,
        )

        self.assertEqual(result["metadata"]["query_feature_source"], "shifted")
        self.assertEqual(result["metadata"]["gallery_feature_source"], "clean")
        self.assertEqual(result["metadata"]["condition"], "mask_erode_s1")
        self.assertAlmostEqual(result["aggregate"]["macro"]["mrr"], 1.0)
        self.assertAlmostEqual(result["aggregate"]["macro"]["map"], 1.0)


class RobustnessPipelineTest(unittest.TestCase):
    def test_prepare_queries_preserves_protocol_basenames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source" / "identity_1"
            source.mkdir(parents=True)
            image_path = source / "query.png"
            image = np.zeros((24, 24, 3), dtype=np.uint8)
            image[6:18, 6:18] = 255
            Image.fromarray(image).save(image_path)
            protocol = {
                "query_groups": [
                    {
                        "group_id": "identity_1",
                        "name": "identity_1",
                        "query_image": "query.png",
                        "same_ginsengs": ["positive.png"],
                    }
                ]
            }
            output = root / "output"

            count = prepare_queries(
                source_root=root / "source",
                output_root=output,
                protocol=protocol,
                kind="mask_erode",
                severity=1,
                seed=42,
            )

            self.assertEqual(count, 1)
            self.assertTrue((output / "query.png").is_file())

    def test_powershell_dry_run_uses_shifted_query_and_clean_gallery(self):
        repo_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(repo_root / "scripts" / "run_robustness.ps1"),
                "-DryRun",
                "-Models",
                "single_topo_plain",
                "-Conditions",
                "mask_erode_s1",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("prepare:mask_erode_s1", completed.stdout)
        self.assertIn("single_topo_plain:mask_erode_s1:extract", completed.stdout)
        self.assertIn("single_topo_plain:mask_erode_s1:evaluate", completed.stdout)
        self.assertIn("single_topo_plain_271_1075.npz", completed.stdout)
        self.assertIn("single_topo_plain__mask_erode_s1.pt", completed.stdout)
        self.assertNotIn(":stamp", completed.stdout)


if __name__ == "__main__":
    unittest.main()
