import base64
import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from ginseng_benchmark.vision_features import (
    discover_flat_image_paths,
    forward_hf_features,
    l2_normalize_rows,
    select_pooled_features,
)
from scripts import extract_hf_vision
from scripts import stamp_feature_cache
from scripts.extract_hf_vision import _load_options


REPO_ROOT = Path(__file__).resolve().parents[1]


class VisionFeatureHelperTest(unittest.TestCase):
    def test_hf_model_prefers_safetensors_but_allows_pinned_bin_fallback(self):
        processor, model = _load_options("abc123", "secret")
        self.assertEqual(
            processor,
            {"revision": "abc123", "token": "secret", "use_fast": False},
        )
        self.assertEqual(
            model,
            {"revision": "abc123", "token": "secret", "use_safetensors": None},
        )

    def test_configures_slow_network_hub_defaults_before_import(self):
        configure = getattr(extract_hf_vision, "_configure_hf_runtime", None)
        self.assertIsNotNone(configure)
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            with mock.patch.dict(os.environ, {"HF_HOME": "C:/old-cache"}, clear=True):
                settings = configure({}, repo_root=repo_root)
                expected_home = str(
                    (repo_root / "artifacts" / "models" / "huggingface").resolve()
                )
                self.assertEqual(os.environ["HF_HOME"], expected_home)
                self.assertEqual(os.environ["HF_HUB_CACHE"], str(Path(expected_home) / "hub"))
                self.assertEqual(os.environ["HF_HUB_DISABLE_PROGRESS_BARS"], "0")
                self.assertEqual(os.environ["HF_HUB_DOWNLOAD_TIMEOUT"], "120")
                self.assertEqual(os.environ["HF_HUB_ETAG_TIMEOUT"], "60")
                self.assertEqual(settings["download_timeout"], 120)
                self.assertEqual(settings["etag_timeout"], 60)
                self.assertEqual(settings["hf_home"], expected_home)
                self.assertEqual(settings["hub_cache"], str(Path(expected_home) / "hub"))

    def test_env_file_overrides_hub_runtime_without_exposing_token(self):
        configure = getattr(extract_hf_vision, "_configure_hf_runtime", None)
        self.assertIsNotNone(configure)
        values = {
            "HF_HUB_DOWNLOAD_TIMEOUT": "240",
            "HF_HUB_ETAG_TIMEOUT": "90",
            "HF_ENDPOINT": "https://huggingface.co",
            "HTTPS_PROXY": "http://127.0.0.1:9888",
            "HF_TOKEN": "must-not-be-returned",
        }
        with mock.patch.dict(os.environ, {}, clear=True):
            settings = configure(values)
            self.assertEqual(os.environ["HF_HUB_DOWNLOAD_TIMEOUT"], "240")
            self.assertEqual(os.environ["HTTPS_PROXY"], "http://127.0.0.1:9888")
            self.assertNotIn("must-not-be-returned", repr(settings))

    def test_rejects_invalid_hub_timeout(self):
        configure = getattr(extract_hf_vision, "_configure_hf_runtime", None)
        self.assertIsNotNone(configure)
        with self.assertRaisesRegex(ValueError, "HF_HUB_DOWNLOAD_TIMEOUT"):
            configure({"HF_HUB_DOWNLOAD_TIMEOUT": "zero"})

    def test_formats_timeout_and_gated_repository_errors(self):
        formatter = getattr(extract_hf_vision, "_format_hf_error", None)
        self.assertIsNotNone(formatter)
        timeout = formatter(RuntimeError("Read timed out. (read timeout=10)"))
        gated = formatter(RuntimeError("401 Client Error: GatedRepoError"))
        self.assertIn("HF_HUB_DOWNLOAD_TIMEOUT", timeout)
        self.assertIn("HF_TOKEN", gated)
        self.assertIn("模型页面", gated)

    def test_formats_line_based_progress_with_eta(self):
        formatter = getattr(extract_hf_vision, "_format_progress", None)
        self.assertIsNotNone(formatter)

        line = formatter(processed=320, total=1280, elapsed_seconds=20.0)

        self.assertIn("[####------------]", line)
        self.assertIn("320/1280", line)
        self.assertIn("25.0%", line)
        self.assertIn("elapsed=00:20", line)
        self.assertIn("ETA=01:00", line)

    def test_progress_reporter_prints_first_periodic_and_final_updates(self):
        reporter_class = getattr(extract_hf_vision, "_ProgressReporter", None)
        self.assertIsNotNone(reporter_class)
        output = []
        reporter = reporter_class(
            total=100,
            update_every=25,
            clock=iter((0.0, 1.0, 2.0, 3.0, 4.0, 5.0)).__next__,
            emit=output.append,
        )

        for processed in (10, 20, 30, 50, 75, 100):
            reporter.update(processed)

        self.assertEqual(len(output), 5)
        self.assertIn("10/100", output[0])
        self.assertIn("30/100", output[1])
        self.assertIn("100/100", output[-1])

    def test_discovers_supported_flat_images_in_stable_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("b.PNG", "A.jpg", "c.jpeg", "ignore.txt"):
                (root / name).write_bytes(b"x")
            paths = discover_flat_image_paths(root)
            self.assertEqual([path.name for path in paths], ["A.jpg", "b.PNG", "c.jpeg"])

    def test_rejects_nested_images_in_gallery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "nested"
            nested.mkdir()
            (nested / "x.jpg").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "flat"):
                discover_flat_image_paths(root)

    def test_selects_pooler_then_cls_and_l2_normalizes(self):
        pooler = torch.tensor([[3.0, 4.0]])
        hidden = torch.tensor([[[9.0, 9.0], [1.0, 1.0]]])
        selected = select_pooled_features(
            SimpleNamespace(pooler_output=pooler, last_hidden_state=hidden)
        )
        self.assertTrue(torch.equal(selected, pooler))
        fallback = select_pooled_features(
            SimpleNamespace(pooler_output=None, last_hidden_state=hidden)
        )
        self.assertTrue(torch.equal(fallback, hidden[:, 0]))
        normalized = l2_normalize_rows(selected)
        self.assertTrue(torch.allclose(normalized, torch.tensor([[0.6, 0.8]])))

    def test_dispatches_official_image_features_and_generic_pooling(self):
        class ImageFeatureModel:
            def get_image_features(self, **inputs):
                self.inputs = inputs
                return torch.tensor([[3.0, 4.0]])

        class PooledModel:
            def __call__(self, **inputs):
                self.inputs = inputs
                return SimpleNamespace(pooler_output=torch.tensor([[5.0, 12.0]]))

        pixels = torch.ones(1, 3, 2, 2)
        image_model = ImageFeatureModel()
        pooled_model = PooledModel()
        self.assertTrue(
            torch.equal(
                forward_hf_features(
                    image_model, {"pixel_values": pixels}, "model_image_features"
                ),
                torch.tensor([[3.0, 4.0]]),
            )
        )
        self.assertTrue(
            torch.equal(
                forward_hf_features(
                    pooled_model, {"pixel_values": pixels}, "pooler_or_cls"
                ),
                torch.tensor([[5.0, 12.0]]),
            )
        )
        self.assertIs(image_model.inputs["pixel_values"], pixels)
        self.assertIs(pooled_model.inputs["pixel_values"], pixels)

    def test_rejects_unknown_or_unsupported_feature_strategy(self):
        with self.assertRaisesRegex(ValueError, "unsupported extractor kind"):
            forward_hf_features(object(), {}, "unknown")
        with self.assertRaisesRegex(ValueError, "get_image_features"):
            forward_hf_features(object(), {}, "model_image_features")


class StampMetadataArgumentTest(unittest.TestCase):
    def test_decodes_urlsafe_base64_json_containing_spaces(self):
        decoder = getattr(stamp_feature_cache, "_json_object_argument", None)
        self.assertIsNotNone(decoder)
        raw = '{"source":"official AutoImageProcessor","resize":256}'
        encoded = base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii").rstrip("=")
        self.assertEqual(
            decoder(None, encoded, "preprocessing-json"),
            {"source": "official AutoImageProcessor", "resize": 256},
        )

    def test_rejects_ambiguous_plain_and_base64_json(self):
        decoder = getattr(stamp_feature_cache, "_json_object_argument", None)
        self.assertIsNotNone(decoder)
        with self.assertRaisesRegex(ValueError, "only one"):
            decoder("{}", "e30", "preprocessing-json")


class StrongBaselineConfigTest(unittest.TestCase):
    def setUp(self):
        self.config_path = REPO_ROOT / "configs" / "strong_models.json"
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))

    def test_config_pins_public_official_huggingface_models(self):
        models = self.config["models"]
        self.assertEqual(
            [model["id"] for model in models],
            [
                "dinov2_base",
                "siglip2_base",
                "clip_vit_b16",
                "swinv2_base",
                "convnextv2_base",
            ],
        )
        self.assertEqual(
            [model["revision"] for model in models],
            [
                "f9e44c814b77203eaa57a6bdbbd535f21ede1415",
                "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
                "57c216476eefef5ab752ec549e440a49ae4ae5f3",
                "485f4d7059ce28604233e20188df5ee19ab960c6",
                "758ff0922dc09136abb55774e7f8b1e1bd0dc344",
            ],
        )
        self.assertEqual(
            [model["feature_dim"] for model in models], [768, 768, 512, 1024, 1024]
        )
        self.assertEqual(
            [model["extractor_kind"] for model in models],
            [
                "pooler_or_cls",
                "model_image_features",
                "model_image_features",
                "pooler_or_cls",
                "pooler_or_cls",
            ],
        )
        self.assertEqual(models[0]["conda_env"], "gsam")
        self.assertTrue(all(model["conda_env"] == "ginseng-baselines" for model in models[1:]))
        self.assertTrue(all(model["gated"] is False for model in models))
        self.assertEqual(models[0]["local_model_env_key"], "DINOV2_WEIGHTS")
        self.assertEqual(
            models[0]["default_local_model_dir"],
            "artifacts/models/dinov2_base",
        )
        self.assertNotIn("dinov3_base", {model["id"] for model in models})
        self.assertNotIn("HF_TOKEN", self.config_path.read_text(encoding="utf-8"))

    def test_dinov2_curl_prefetch_is_resumable_and_integrity_checked(self):
        script = (REPO_ROOT / "scripts" / "download_dinov2.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn("f9e44c814b77203eaa57a6bdbbd535f21ede1415", script)
        self.assertIn(
            "d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841",
            script,
        )
        self.assertIn("346345912", script)
        self.assertIn('"-C", "-"', script)
        self.assertIn('"--progress-bar"', script)
        self.assertIn("Get-FileHash", script)
        self.assertIn("model.safetensors", script)

    def test_modern_environment_assets_are_complete_and_pinned(self):
        requirements = (REPO_ROOT / "requirements-modern.txt").read_text(
            encoding="utf-8"
        )
        setup = (REPO_ROOT / "scripts" / "setup_modern_env.ps1").read_text(
            encoding="utf-8"
        )
        check = (REPO_ROOT / "scripts" / "check_modern_env.py").read_text(
            encoding="utf-8"
        )
        env_example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn("transformers==4.57.0", requirements)
        self.assertIn("huggingface-hub==0.35.3", requirements)
        self.assertIn("torch==2.7.1", setup)
        self.assertIn("[switch]$DryRun", setup)
        self.assertIn("ginseng-baselines", check)
        for model_type in ("siglip", "clip", "swinv2", "convnextv2"):
            self.assertIn(model_type, check)
        self.assertIn("HF_HUB_DOWNLOAD_TIMEOUT=120", env_example)
        self.assertIn("HF_HUB_ETAG_TIMEOUT=60", env_example)

    def test_modern_setup_uses_direct_pip_with_resilient_download_options(self):
        completed = subprocess.run(
            [
                "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                str(REPO_ROOT / "scripts" / "setup_modern_env.ps1"), "-DryRun",
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
        self.assertIn("python.exe -m pip install", output)
        self.assertNotIn("conda run", output)
        self.assertIn("--timeout 600", output)
        self.assertIn("--retries 20", output)
        self.assertIn("--resume-retries 20", output)
        setup = (REPO_ROOT / "scripts" / "setup_modern_env.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn('Join-Path $EnvironmentPrefix "python.exe"', setup)
        self.assertIn("pip-modern-$PID", setup)

    def test_gsam_dinov2_upgrade_preserves_legacy_torch_stack(self):
        setup = (REPO_ROOT / "scripts" / "setup_gsam_dinov2.ps1").read_text(
            encoding="utf-8"
        )
        check = (REPO_ROOT / "scripts" / "check_gsam_dinov2_env.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("transformers==4.35.2", setup)
        self.assertIn("tokenizers==0.15.2", setup)
        self.assertIn("--no-deps", setup)
        self.assertNotIn("pip uninstall", setup)
        self.assertIn('"torch": "1.13.1"', check)
        self.assertIn('"torchvision": "0.14.1"', check)
        self.assertIn('"transformers": "4.35.2"', check)
        self.assertIn('"tokenizers": "0.15.2"', check)

    def test_runner_dry_run_covers_extract_stamp_and_full_evaluation(self):
        completed = subprocess.run(
            [
                "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                str(REPO_ROOT / "scripts" / "run_strong_baselines.ps1"),
                "-DryRun",
                "-Models",
                "dinov2_base,siglip2_base,clip_vit_b16,swinv2_base,convnextv2_base",
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
        for model_id in (
            "dinov2_base",
            "siglip2_base",
            "clip_vit_b16",
            "swinv2_base",
            "convnextv2_base",
        ):
            self.assertIn(f"{model_id}:extract", output)
            self.assertIn(f"{model_id}:stamp", output)
            self.assertIn(f"{model_id}:evaluate", output)
            self.assertIn(f"{model_id}_271_1075", output)
        self.assertIn("evaluate_features.py", output)
        self.assertIn("--bootstrap-iterations 2000", output)
        self.assertIn("--preprocessing-json-base64", output)
        self.assertNotIn("--preprocessing-json ", output)
        match = re.search(r"--preprocessing-json-base64 ([A-Za-z0-9_-]+)", output)
        self.assertIsNotNone(match)
        padding = "=" * (-len(match.group(1)) % 4)
        decoded = base64.urlsafe_b64decode(match.group(1) + padding).decode("utf-8")
        self.assertEqual(
            json.loads(decoded)["source"],
            "official AutoImageProcessor",
        )
        self.assertIn("--extractor-kind model_image_features", output)
        self.assertIn("--extractor-kind pooler_or_cls", output)
        self.assertEqual(output.count("--no-capture-output"), 15)
        self.assertEqual(output.count("python -u"), 15)
        self.assertNotIn("1182", output)

    def test_command_guide_separates_frozen_models_from_transreid(self):
        guide = (REPO_ROOT / "docs" / "commands" / "strong-baselines.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("run_strong_baselines.ps1 -DryRun", guide)
        self.assertIn("setup_transreid.ps1 -DryRun", guide)
        for model_id in ("siglip2_base", "clip_vit_b16", "swinv2_base", "convnextv2_base"):
            self.assertIn(model_id, guide)
        self.assertIn("DINOv3", guide)
        self.assertIn("不再运行", guide)
        self.assertIn("不能把普通 ViT", guide)
        self.assertIn("身份标签", guide)


if __name__ == "__main__":
    unittest.main()
