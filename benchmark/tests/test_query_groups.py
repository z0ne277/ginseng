import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from ginseng_benchmark.query_groups import (
    build_query_groups,
    build_query_protocol,
    write_query_protocol_atomic,
)


class QueryGroupsTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.test_root = self.root / "test"
        self.gallery_root = self.root / "gallery"
        self.test_root.mkdir()
        self.gallery_root.mkdir()

    def _add_group(self, group_id: int, names):
        group = self.test_root / str(group_id)
        group.mkdir()
        for index, name in enumerate(names):
            content = f"group-{group_id}-image-{index}".encode("ascii")
            (group / name).write_bytes(content)
            (self.gallery_root / name).write_bytes(content)

    def test_all_images_become_queries_and_self_is_excluded(self):
        self._add_group(1, ["c.png", "A.png", "b.PNG"])

        groups = build_query_groups(
            self.test_root,
            self.gallery_root,
            expected_groups=1,
        )

        self.assertEqual(
            [Path(group["query_image"]).name for group in groups],
            ["A.png", "b.PNG", "c.png"],
        )
        self.assertEqual({group["group_id"] for group in groups}, {"1"})
        for group in groups:
            self.assertEqual(group["name"], group["group_id"])
            query = Path(group["query_image"])
            positives = [Path(path) for path in group["same_ginsengs"]]
            self.assertEqual(query.parent, self.gallery_root.resolve())
            self.assertTrue(query.is_file())
            self.assertEqual(len(positives), 2)
            self.assertTrue(all(path.parent == self.gallery_root.resolve() for path in positives))
            self.assertTrue(all(path.is_file() for path in positives))
            self.assertNotIn(
                query.name.casefold(),
                {positive.name.casefold() for positive in positives},
            )

    def test_groups_are_sorted_numerically(self):
        self._add_group(2, ["two-a.png", "two-b.png"])
        self._add_group(1, ["one-a.png", "one-b.png"])

        groups = build_query_groups(
            self.test_root,
            self.gallery_root,
            expected_groups=2,
        )

        self.assertEqual([group["group_id"] for group in groups], ["1", "1", "2", "2"])

    def test_rejects_non_numeric_non_contiguous_and_wrong_group_counts(self):
        cases = (
            ("non-numeric", ["1", "group-2"], 2, "numeric"),
            ("non-contiguous", ["1", "3"], 2, "1..2"),
            ("wrong-count", ["1", "2"], 3, "expected 3"),
        )
        for label, group_names, expected_groups, message in cases:
            with self.subTest(label=label):
                case_root = self.root / label
                test_root = case_root / "test"
                gallery_root = case_root / "gallery"
                test_root.mkdir(parents=True)
                gallery_root.mkdir()
                for group_name in group_names:
                    group = test_root / group_name
                    group.mkdir()
                    for suffix in ("a", "b"):
                        name = f"{group_name}-{suffix}.png"
                        content = name.encode("utf-8")
                        (group / name).write_bytes(content)
                        (gallery_root / name).write_bytes(content)

                with self.assertRaisesRegex(ValueError, message):
                    build_query_groups(test_root, gallery_root, expected_groups)

    def test_rejects_group_with_fewer_than_two_images(self):
        self._add_group(1, ["only.png"])

        with self.assertRaisesRegex(ValueError, "at least 2"):
            build_query_groups(self.test_root, self.gallery_root, expected_groups=1)

    def test_rejects_root_level_and_nested_test_images(self):
        for label, invalid_relative_path in (
            ("root", Path("root.png")),
            ("nested", Path("1") / "nested" / "deep.png"),
        ):
            with self.subTest(label=label):
                case_root = self.root / label
                test_root = case_root / "test"
                gallery_root = case_root / "gallery"
                (test_root / "1").mkdir(parents=True)
                gallery_root.mkdir(parents=True)
                for name in ("a.png", "b.png"):
                    content = name.encode("ascii")
                    (test_root / "1" / name).write_bytes(content)
                    (gallery_root / name).write_bytes(content)
                invalid = test_root / invalid_relative_path
                invalid.parent.mkdir(parents=True, exist_ok=True)
                invalid.write_bytes(b"invalid")

                with self.assertRaisesRegex(ValueError, "directly inside"):
                    build_query_groups(test_root, gallery_root, expected_groups=1)

    def test_rejects_casefold_basename_collision_across_groups(self):
        (self.test_root / "1").mkdir()
        (self.test_root / "2").mkdir()
        for group_id, names in (("1", ("same.PNG", "one.png")), ("2", ("SAME.png", "two.png"))):
            for name in names:
                (self.test_root / group_id / name).write_bytes(
                    f"{group_id}-{name}".encode("ascii")
                )

        with self.assertRaisesRegex(ValueError, "duplicate basename"):
            build_query_groups(self.test_root, self.gallery_root, expected_groups=2)

    def test_rejects_missing_gallery_image(self):
        self._add_group(1, ["a.png", "b.png"])
        (self.gallery_root / "b.png").unlink()

        with self.assertRaisesRegex(ValueError, "missing from merged"):
            build_query_groups(self.test_root, self.gallery_root, expected_groups=1)

    def test_rejects_test_and_gallery_content_mismatch(self):
        self._add_group(1, ["a.png", "b.png"])
        (self.gallery_root / "a.png").write_bytes(b"wrong-content")

        with self.assertRaisesRegex(ValueError, "content mismatch"):
            build_query_groups(self.test_root, self.gallery_root, expected_groups=1)

    def test_protocol_hash_is_deterministic_and_independent_of_roots(self):
        self._add_group(1, ["z.png", "a.png"])
        first = build_query_protocol(
            self.test_root,
            self.gallery_root,
            dataset_manifest_sha256="a" * 64,
            gallery_count=2,
            expected_groups=1,
        )

        other_test = self.root / "other" / "test"
        other_gallery = self.root / "other" / "gallery"
        other_test.mkdir(parents=True)
        other_gallery.mkdir()
        (other_test / "1").mkdir()
        for name in ("z.png", "a.png"):
            (other_test / "1" / name).write_bytes((self.test_root / "1" / name).read_bytes())
            (other_gallery / name).write_bytes((self.gallery_root / name).read_bytes())
        second = build_query_protocol(
            other_test,
            other_gallery,
            dataset_manifest_sha256="a" * 64,
            gallery_count=2,
            expected_groups=1,
        )
        repeated = build_query_protocol(
            self.test_root,
            self.gallery_root,
            dataset_manifest_sha256="a" * 64,
            gallery_count=2,
            expected_groups=1,
        )

        self.assertEqual(
            first["metadata"]["query_protocol_sha256"],
            second["metadata"]["query_protocol_sha256"],
        )
        self.assertEqual(first, repeated)
        self.assertEqual(
            first["metadata"]["positive_count_distribution"],
            {"1": 2},
        )

        expected_canonical = [
            {
                "group_id": group["group_id"],
                "name": group["name"],
                "query_name": Path(group["query_image"]).name,
                "same_ginseng_names": [
                    Path(path).name for path in group["same_ginsengs"]
                ],
            }
            for group in first["query_groups"]
        ]
        expected_hash = hashlib.sha256(
            json.dumps(
                expected_canonical,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(
            first["metadata"]["query_protocol_sha256"],
            expected_hash,
        )

    def test_rejects_gallery_count_that_does_not_match_actual_files(self):
        self._add_group(1, ["a.png", "b.png"])

        with self.assertRaisesRegex(ValueError, "gallery_count.*actual"):
            build_query_protocol(
                self.test_root,
                self.gallery_root,
                dataset_manifest_sha256="c" * 64,
                gallery_count=999,
                expected_groups=1,
            )

    def test_rejects_resolved_gallery_file_outside_gallery_root(self):
        self._add_group(1, ["a.png", "b.png"])
        escaped_path = self.root / "outside" / "a.png"
        real_resolve = Path.resolve
        gallery_a = self.gallery_root / "a.png"

        def resolve_with_escape(path, *args, **kwargs):
            if path == gallery_a:
                return escaped_path
            return real_resolve(path, *args, **kwargs)

        with mock.patch(
            "ginseng_benchmark.query_groups.Path.resolve",
            autospec=True,
            side_effect=resolve_with_escape,
        ):
            with self.assertRaisesRegex(ValueError, "outside merged gallery"):
                build_query_groups(
                    self.test_root,
                    self.gallery_root,
                    expected_groups=1,
                )

    def test_payload_has_loader_compatible_structure(self):
        self._add_group(1, ["a.png", "b.png"])

        payload = build_query_protocol(
            self.test_root,
            self.gallery_root,
            dataset_manifest_sha256="b" * 64,
            gallery_count=2,
            expected_groups=1,
        )

        self.assertIsInstance(payload, dict)
        self.assertIsInstance(payload["query_groups"], list)
        self.assertEqual(payload["metadata"]["schema_version"], "1.0")
        self.assertEqual(payload["metadata"]["dataset_manifest_sha256"], "b" * 64)
        for index, group in enumerate(payload["query_groups"]):
            self.assertIsInstance(group["group_id"], str)
            self.assertEqual(group["name"], group["group_id"])
            legacy_name = group.get("name", f"group_{index}")
            self.assertEqual(legacy_name, group["group_id"])
            self.assertNotEqual(legacy_name, f"group_{index}")
            self.assertTrue(Path(group["query_image"]).is_absolute())
            self.assertIsInstance(group["same_ginsengs"], list)

    def test_atomic_writer_preserves_existing_file_and_cleans_temp_on_failure(self):
        output = self.root / "manifests" / "query_groups.json"
        output.parent.mkdir()
        output.write_text("old-content", encoding="utf-8")

        with mock.patch(
            "ginseng_benchmark.query_groups.os.replace",
            side_effect=OSError("simulated replace failure"),
        ):
            with self.assertRaisesRegex(OSError, "replace failure"):
                write_query_protocol_atomic({"query_groups": []}, output)

        self.assertEqual(output.read_text(encoding="utf-8"), "old-content")
        self.assertEqual(list(output.parent.glob(".query-groups-*.tmp")), [])

    def test_atomic_writer_cleans_partial_json_after_serialization_failure(self):
        output = self.root / "manifests" / "query_groups.json"
        output.parent.mkdir()
        output.write_text("old-content", encoding="utf-8")

        def fail_after_partial_write(payload, file_handle, **kwargs):
            file_handle.write('{"partial":')
            raise TypeError("simulated serialization failure")

        with mock.patch(
            "ginseng_benchmark.query_groups.json.dump",
            side_effect=fail_after_partial_write,
        ):
            with self.assertRaisesRegex(TypeError, "serialization failure"):
                write_query_protocol_atomic({"query_groups": []}, output)

        self.assertEqual(output.read_text(encoding="utf-8"), "old-content")
        self.assertEqual(list(output.parent.glob(".query-groups-*.tmp")), [])


class BuildQueryGroupsScriptTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.library = self.root / "library"
        self.test_root = self.root / "test"
        self.gallery = self.root / "gallery"
        self.library.mkdir()
        (self.test_root / "1").mkdir(parents=True)
        self.gallery.mkdir()
        (self.library / "library.png").write_bytes(b"library")
        (self.gallery / "library.png").write_bytes(b"library")
        for name, content in (("q1.png", b"query-1"), ("q2.png", b"query-2")):
            (self.test_root / "1" / name).write_bytes(content)
            (self.gallery / name).write_bytes(content)
        self.env_path = self.root / ".env"
        self.env_path.write_text(
            f"LIBRARY_BINARY={self.library}\n"
            f"TEST_BINARY_ROOT={self.test_root}\n"
            f"MERGED_GALLERY={self.gallery}\n"
            "HF_TOKEN=must-not-appear\n",
            encoding="utf-8",
        )
        self.output = self.root / "query_groups.json"
        self.script = Path(__file__).parents[1] / "scripts" / "build_query_groups.py"

    def _run(self, *extra_arguments):
        return subprocess.run(
            [
                sys.executable,
                str(self.script),
                "--env",
                str(self.env_path),
                "--output",
                str(self.output),
                "--expected-groups",
                "1",
                "--expected-library-count",
                "1",
                "--expected-query-count",
                "2",
                "--expected-gallery-count",
                "3",
                "--expected-positive-distribution",
                "1:2",
                *extra_arguments,
            ],
            cwd=self.script.parents[1],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_cli_audits_then_writes_stable_loader_compatible_json(self):
        result = self._run()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("groups=1 queries=2 gallery=3 missing=0 self_positive=0", result.stdout)
        payload = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["query_groups"]), 2)
        self.assertEqual(payload["metadata"]["group_count"], 1)
        self.assertEqual(payload["metadata"]["query_count"], 2)
        self.assertEqual(payload["metadata"]["gallery_count"], 3)
        self.assertNotIn("must-not-appear", self.output.read_text(encoding="utf-8"))

    def test_cli_mismatch_gate_does_not_create_or_replace_output(self):
        self.output.write_text("sentinel", encoding="utf-8")
        (self.gallery / "q1.png").write_bytes(b"wrong")

        result = self._run()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mismatch", result.stderr.casefold())
        self.assertEqual(self.output.read_text(encoding="utf-8"), "sentinel")

    def test_cli_size_gate_does_not_create_or_replace_output(self):
        self.output.write_text("sentinel", encoding="utf-8")

        result = self._run("--expected-library-count", "2")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("library count mismatch", result.stderr.casefold())
        self.assertEqual(self.output.read_text(encoding="utf-8"), "sentinel")

    def test_cli_positive_distribution_gate_does_not_write_output(self):
        result = self._run("--expected-positive-distribution", "2:2")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("positive count distribution mismatch", result.stderr.casefold())
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
