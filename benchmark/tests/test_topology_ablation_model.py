import base64
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import unittest

import torch
import torch.nn.functional as F


MAIN_CODE_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = MAIN_CODE_ROOT / "single_topo" / "model.py"


def _load_model_module():
    spec = importlib.util.spec_from_file_location("single_topo_ablation_model", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TopologyOperatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_model_module()

    def test_min_max_and_average_operators_match_pooling_definitions(self):
        value = torch.tensor(
            [[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]]
        )
        for operator, expected in (
            ("min", -F.max_pool2d(-value, 3, stride=1, padding=1)),
            ("max", F.max_pool2d(value, 3, stride=1, padding=1)),
            ("avg", F.avg_pool2d(value, 3, stride=1, padding=1)),
            ("identity", value),
        ):
            with self.subTest(operator=operator):
                layer = self.module.MorphologicalErosion(
                    kernel_size=3,
                    num_erosions=1,
                    operator=operator,
                )
                torch.testing.assert_close(layer(value), expected)

    def test_invalid_even_kernel_and_unknown_operator_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "odd"):
            self.module.MorphologicalErosion(kernel_size=4)
        with self.assertRaisesRegex(ValueError, "operator"):
            self.module.MorphologicalErosion(operator="median")

    def test_default_config_exposes_controlled_ablation_factors(self):
        config_path = MAIN_CODE_ROOT / "single_topo" / "configs" / "default.json"
        common = json.loads(config_path.read_text(encoding="utf-8"))["common"]
        self.assertEqual(common["erosion_kernel_size"], 3)
        self.assertEqual(common["topology_operator"], "min")
        self.assertEqual(common["topology_negative_source"], "queue")
        self.assertTrue(common["use_cbam"])
        self.assertEqual(common["backbone_name"], "resnet50")
        self.assertEqual(common["seed"], 42)

    def test_training_ablation_matrix_is_one_factor_at_a_time(self):
        repo_root = Path(__file__).resolve().parents[1]
        matrix = json.loads(
            (repo_root / "configs" / "topology_training_ablations.json").read_text(
                encoding="utf-8"
            )
        )
        variants = {item["id"]: item for item in matrix["variants"]}
        self.assertIn("reference", variants)
        reference = variants["reference"]["overrides"]
        self.assertEqual(reference["topology_operator"], "min")
        required = {
            "levels_2",
            "levels_4",
            "operator_identity",
            "cbam_off",
            "backbone_convnext_tiny",
            "backbone_swin_v2_t",
            "backbone_vit_b_16",
        }
        self.assertEqual(set(variants), required | {"reference"})
        self.assertEqual(variants["levels_2"]["overrides"]["num_erosion_levels"], 3)
        self.assertEqual(variants["levels_4"]["overrides"]["num_erosion_levels"], 5)
        for variant_id, variant in variants.items():
            self.assertFalse(variant["requires_identity_labels"])
            changed = {
                key
                for key, value in variant["overrides"].items()
                if reference.get(key) != value
            }
            if variant_id != "reference":
                self.assertEqual(
                    len(changed),
                    1,
                    f"{variant_id} is not a one-factor-at-a-time variant",
                )

    def test_training_ablation_runner_has_visible_staged_dry_run(self):
        repo_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(repo_root / "scripts" / "run_topology_ablations.ps1"),
                "-DryRun",
                "-Variants",
                "levels_2",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("levels_2:train", completed.stdout)
        self.assertIn("levels_2:extract", completed.stdout)
        self.assertIn("levels_2:stamp", completed.stdout)
        self.assertIn("levels_2:evaluate", completed.stdout)
        self.assertIn("num_erosion_levels=3", completed.stdout)
        self.assertIn("topology_operator=min", completed.stdout)
        self.assertIn("tta_enabled=false", completed.stdout)
        self.assertIn("--preprocessing-json-base64", completed.stdout)
        self.assertNotIn("--preprocessing-json ", completed.stdout)
        match = re.search(
            r"--preprocessing-json-base64 ([A-Za-z0-9_-]+)",
            completed.stdout,
        )
        self.assertIsNotNone(match)
        padding = "=" * (-len(match.group(1)) % 4)
        metadata = json.loads(
            base64.urlsafe_b64decode(match.group(1) + padding).decode("utf-8")
        )
        self.assertEqual(
            metadata["training_information"],
            "image-only self-supervision; no identity labels",
        )


if __name__ == "__main__":
    unittest.main()
