import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_results import METRICS, summarize, write_tables


REPO_ROOT = Path(__file__).resolve().parents[1]


class ExistingModelConfigTest(unittest.TestCase):
    def setUp(self):
        self.config_path = REPO_ROOT / "configs" / "existing_models.json"
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))

    def test_config_contains_exact_five_rebuilt_models(self):
        models = self.config["models"]
        self.assertEqual(
            [model["id"] for model in models],
            ["simclr", "moco_v3", "moco_v3_cbam", "single_topo_plain", "single_topo_tta"],
        )
        self.assertEqual(self.config["protocol_tag"], "271_1075")
        text = self.config_path.read_text(encoding="utf-8")
        self.assertNotIn("1182", text)
        for model in models:
            self.assertGreater(model["feature_dim"], 0)
            self.assertIn("preprocessing", model)
            self.assertIn("tta", model)

    def test_plain_and_tta_are_explicitly_distinct(self):
        models = {model["id"]: model for model in self.config["models"]}
        plain = models["single_topo_plain"]
        tta = models["single_topo_tta"]
        self.assertFalse(plain["tta"]["enabled"])
        self.assertEqual(plain["tta"]["weights"], [1.0])
        self.assertIn("tta_enabled=false", plain["extractor"]["arguments"])
        self.assertIn("tta_weights=[1.0]", plain["extractor"]["arguments"])
        self.assertTrue(tta["tta"]["enabled"])
        self.assertEqual(tta["tta"]["weights"], [0.6, 0.25, 0.15])
        self.assertIn(
            "tta_modes=stretch224,contain224,contain256",
            tta["extractor"]["arguments"],
        )
        self.assertFalse(
            any(
                argument.startswith("tta_modes=[")
                for argument in tta["extractor"]["arguments"]
            )
        )

    def test_runners_capture_complete_native_stderr_before_failing(self):
        for runner_name in (
            "run_existing_models.ps1",
            "run_strong_baselines.ps1",
            "run_transreid.ps1",
        ):
            with self.subTest(runner=runner_name):
                script = (REPO_ROOT / "scripts" / runner_name).read_text(
                    encoding="utf-8"
                )
                self.assertIn('$ErrorActionPreference = "Continue"', script)
                self.assertIn(
                    "$ErrorActionPreference = $previousErrorActionPreference",
                    script,
                )

    def test_runners_escape_json_arguments_for_windows_conda(self):
        for runner_name in (
            "run_existing_models.ps1",
            "run_transreid.ps1",
        ):
            with self.subTest(runner=runner_name):
                script = (REPO_ROOT / "scripts" / runner_name).read_text(
                    encoding="utf-8"
                )
                self.assertIn("function ConvertTo-NativeJsonArgument", script)
                self.assertGreaterEqual(
                    script.count("ConvertTo-NativeJsonArgument"),
                    4,
                )
        strong_runner = (
            REPO_ROOT / "scripts" / "run_strong_baselines.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn("function ConvertTo-UrlSafeBase64", strong_runner)
        self.assertGreaterEqual(strong_runner.count("ConvertTo-UrlSafeBase64"), 4)

    def test_powershell_dry_run_contains_all_three_phases_and_no_legacy_cache(self):
        completed = subprocess.run(
            [
                "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                str(REPO_ROOT / "scripts" / "run_existing_models.ps1"),
                "-DryRun", "-Models", "single_topo_plain",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        output = completed.stdout
        self.assertIn("single_topo_plain:extract", output)
        self.assertIn("single_topo_plain:stamp", output)
        self.assertIn("single_topo_plain:evaluate", output)
        self.assertIn("tta_enabled=false", output)
        self.assertIn("tta_weights=[1.0]", output)
        self.assertIn("single_topo_plain_271_1075.pt", output)
        self.assertEqual(
            output.count("conda run --no-capture-output -n gsam python -u"),
            3,
        )
        self.assertIn("[single_topo_plain:extract] log=", output)
        self.assertNotIn("1182", output)

    def test_powershell_accepts_comma_separated_model_ids(self):
        completed = subprocess.run(
            [
                "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                str(REPO_ROOT / "scripts" / "run_existing_models.ps1"),
                "-DryRun", "-Models", "simclr,moco_v3",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("simclr:extract", completed.stdout)
        self.assertIn("moco_v3:extract", completed.stdout)

    def test_controlled_ablation_config_uses_one_checkpoint_without_tta(self):
        config_path = REPO_ROOT / "configs" / "controlled_ablations.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        models = {model["id"]: model for model in config["models"]}
        self.assertEqual(
            list(models),
            ["single_topo_visual_plain", "single_topo_topology_plain"],
        )
        self.assertEqual(
            {model["checkpoint"] for model in models.values()},
            {"single_topo/checkpoints/moco_v3_topo/best_model.pth"},
        )
        self.assertEqual(models["single_topo_visual_plain"]["feature_dim"], 256)
        self.assertEqual(models["single_topo_topology_plain"]["feature_dim"], 128)
        for model_id, expected_type in (
            ("single_topo_visual_plain", "visual"),
            ("single_topo_topology_plain", "topo"),
        ):
            model = models[model_id]
            self.assertFalse(model["tta"]["enabled"])
            self.assertIn("tta_enabled=false", model["extractor"]["arguments"])
            self.assertIn(
                f"feature_type={expected_type}",
                model["extractor"]["arguments"],
            )

        completed = subprocess.run(
            [
                "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                str(REPO_ROOT / "scripts" / "run_existing_models.ps1"),
                "-DryRun",
                "-Config", str(config_path),
                "-Models", "single_topo_visual_plain,single_topo_topology_plain",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("feature_type=visual", completed.stdout)
        self.assertIn("feature_type=topo", completed.stdout)
        self.assertIn("single_topo_visual_plain_271_1075.pt", completed.stdout)
        self.assertIn("single_topo_topology_plain_271_1075.pt", completed.stdout)

    def test_command_guide_covers_safe_staged_execution(self):
        guide = (REPO_ROOT / "docs" / "commands" / "existing-models.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("run_existing_models.ps1 -DryRun", guide)
        self.assertIn('-Phase extract', guide)
        self.assertIn('-Phase stamp', guide)
        self.assertIn('-Phase evaluate', guide)
        self.assertIn("summarize_results.py", guide)
        self.assertNotIn("1182", guide)


def fake_result(model_id, manifest="a" * 64, protocol="b" * 64):
    macro = {metric: 0.5 for metric in METRICS}
    intervals = {
        metric: {"point_estimate": 0.5, "lower": 0.4, "upper": 0.6}
        for metric in METRICS
    }
    return {
        "metadata": {
            "model_id": model_id,
            "ranking_scope": "full",
            "query_count": 1075,
            "gallery_count": 12787,
            "dataset_manifest_sha256": manifest,
            "query_protocol_sha256": protocol,
        },
        "aggregate": {"macro": macro},
        "bootstrap": {"metrics": intervals},
    }


class SummaryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.results = self.root / "results"
        self.results.mkdir()
        self.config = {
            "protocol_tag": "271_1075",
            "models": [
                {"id": "one", "display_name": "One"},
                {"id": "two", "display_name": "Two"},
            ],
        }

    def tearDown(self):
        self.temporary.cleanup()

    def test_summary_rejects_missing_and_cross_protocol_results(self):
        (self.results / "one_271_1075.json").write_text(
            json.dumps(fake_result("one")), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "missing result"):
            summarize(self.config, self.results, allow_missing=False)

        (self.results / "two_271_1075.json").write_text(
            json.dumps(fake_result("two", manifest="c" * 64)), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "share one"):
            summarize(self.config, self.results, allow_missing=False)

    def test_allow_missing_and_table_writers(self):
        (self.results / "one_271_1075.json").write_text(
            json.dumps(fake_result("one")), encoding="utf-8"
        )
        rows = summarize(self.config, self.results, allow_missing=True)
        self.assertEqual([row["status"] for row in rows], ["OK", "PENDING"])
        markdown = self.root / "table.md"
        csv_path = self.root / "table.csv"
        write_tables(rows, markdown, csv_path)
        self.assertIn("0.5000 [0.4000, 0.6000]", markdown.read_text(encoding="utf-8"))
        self.assertIn("PENDING", markdown.read_text(encoding="utf-8"))
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        self.assertEqual(len(csv_rows), 2)
        self.assertEqual(csv_rows[0]["map"], "0.5")


if __name__ == "__main__":
    unittest.main()
