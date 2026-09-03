import hashlib
import json
from dataclasses import replace
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
import shutil

from ginseng_benchmark.protocol import (
    audit_sources,
    digest,
    image_files,
    repair_mismatches,
)


class AuditSourcesTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.library = self.root / "library"
        self.test = self.root / "test"
        self.merged = self.root / "merged"
        self.library.mkdir()
        self.test.mkdir()
        self.merged.mkdir()

    @staticmethod
    def _write(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def test_reports_counts_records_and_only_content_mismatch(self):
        self._write(self.library / "a.png", b"library-a")
        self._write(self.test / "1" / "b.png", b"test-b")
        self._write(self.test / "1" / "c.png", b"test-c")
        self._write(self.merged / "a.png", b"wrong-a")
        self._write(self.merged / "b.png", b"test-b")
        self._write(self.merged / "c.png", b"test-c")

        report = audit_sources(
            self.library,
            self.test,
            self.merged,
            expected_groups=1,
        )

        self.assertEqual(report.library_count, 1)
        self.assertEqual(report.test_count, 2)
        self.assertEqual(report.merged_count, 3)
        self.assertEqual(report.group_count, 1)
        self.assertEqual([mismatch.name for mismatch in report.mismatches], ["a.png"])
        self.assertEqual(report.mismatches[0].source.sha256, hashlib.sha256(b"library-a").hexdigest())
        self.assertEqual(report.mismatches[0].merged.sha256, hashlib.sha256(b"wrong-a").hexdigest())
        self.assertEqual(
            [(record.source, record.name, record.group_id) for record in report.records],
            [("library", "a.png", None), ("test", "b.png", "1"), ("test", "c.png", "1")],
        )

    def test_rejects_wrong_group_count(self):
        (self.test / "group-a").mkdir()

        with self.assertRaisesRegex(ValueError, "group"):
            audit_sources(self.library, self.test, self.merged, expected_groups=2)

    def test_rejects_cross_source_duplicate_basename(self):
        self._write(self.library / "same.png", b"library")
        self._write(self.test / "1" / "same.png", b"test")
        self._write(self.merged / "same.png", b"library")

        with self.assertRaisesRegex(ValueError, "same.png"):
            audit_sources(self.library, self.test, self.merged, expected_groups=1)

    def test_rejects_duplicate_basename_inside_library(self):
        self._write(self.library / "left" / "same.png", b"left")
        self._write(self.library / "right" / "same.png", b"right")
        (self.test / "1").mkdir()

        with self.assertRaisesRegex(ValueError, "same.png"):
            audit_sources(self.library, self.test, self.merged, expected_groups=1)

    def test_rejects_case_only_duplicate_basename_inside_one_source(self):
        self._write(self.library / "left" / "A.PNG", b"left")
        self._write(self.library / "right" / "a.png", b"right")
        (self.test / "1").mkdir()
        self._write(self.merged / "A.PNG", b"left")

        with self.assertRaises(ValueError) as error:
            audit_sources(self.library, self.test, self.merged, expected_groups=1)

        message = str(error.exception)
        self.assertIn("duplicate basename", message)
        self.assertIn("A.PNG", message)
        self.assertIn("a.png", message)

    def test_rejects_duplicate_basename_inside_test(self):
        self._write(self.test / "1" / "same.png", b"one")
        self._write(self.test / "2" / "same.png", b"two")
        self._write(self.merged / "same.png", b"one")

        with self.assertRaisesRegex(ValueError, "same.png"):
            audit_sources(self.library, self.test, self.merged, expected_groups=2)

    def test_rejects_duplicate_basename_inside_merged(self):
        self._write(self.library / "same.png", b"source")
        (self.test / "1").mkdir()
        self._write(self.merged / "left" / "same.png", b"source")
        self._write(self.merged / "right" / "same.png", b"source")

        with self.assertRaisesRegex(ValueError, "same.png"):
            audit_sources(self.library, self.test, self.merged, expected_groups=1)

    def test_rejects_cross_source_basename_conflict_that_differs_only_by_case(self):
        self._write(self.library / "A.PNG", b"library")
        self._write(self.test / "1" / "a.png", b"test")
        self._write(self.merged / "A.PNG", b"library")

        with self.assertRaises(ValueError) as error:
            audit_sources(self.library, self.test, self.merged, expected_groups=1)

        message = str(error.exception)
        self.assertIn("both library and test", message)
        self.assertIn("A.PNG", message)
        self.assertIn("a.png", message)

    def test_matches_merged_basename_case_insensitively_and_preserves_names(self):
        self._write(self.library / "A.PNG", b"source")
        (self.test / "1").mkdir()
        self._write(self.merged / "a.png", b"different")

        report = audit_sources(
            self.library,
            self.test,
            self.merged,
            expected_groups=1,
        )

        self.assertEqual(report.records[0].name, "A.PNG")
        self.assertEqual(len(report.mismatches), 1)
        self.assertEqual(report.mismatches[0].name, "A.PNG")
        self.assertEqual(report.mismatches[0].source.name, "A.PNG")
        self.assertEqual(report.mismatches[0].merged.name, "a.png")

    def test_rejects_missing_merged_file(self):
        self._write(self.library / "a.png", b"a")
        (self.test / "1").mkdir()

        with self.assertRaisesRegex(ValueError, "missing"):
            audit_sources(self.library, self.test, self.merged, expected_groups=1)

    def test_rejects_extra_merged_file(self):
        (self.test / "1").mkdir()
        self._write(self.merged / "extra.png", b"extra")

        with self.assertRaisesRegex(ValueError, "extra"):
            audit_sources(self.library, self.test, self.merged, expected_groups=1)

    def test_rejects_test_images_outside_a_direct_group_directory(self):
        for scenario, relative_path in (
            ("root", Path("root.png")),
            ("nested", Path("group-a") / "nested" / "deep.png"),
        ):
            with self.subTest(scenario=scenario):
                fixture_root = self.root / scenario
                library = fixture_root / "library"
                test = fixture_root / "test"
                merged = fixture_root / "merged"
                library.mkdir(parents=True)
                (test / "group-a").mkdir(parents=True)
                merged.mkdir(parents=True)
                self._write(test / relative_path, b"test")
                self._write(merged / relative_path.name, b"test")

                with self.assertRaisesRegex(ValueError, "directly inside"):
                    audit_sources(library, test, merged, expected_groups=1)

    def test_limits_missing_and_extra_name_lists_and_reports_totals(self):
        (self.test / "1").mkdir()
        for index in range(12):
            self._write(self.library / f"missing-{index:02}.png", b"source")
            self._write(self.merged / f"extra-{index:02}.png", b"merged")

        with self.assertRaises(ValueError) as error:
            audit_sources(self.library, self.test, self.merged, expected_groups=1)

        message = str(error.exception)
        self.assertEqual(message.count("total=12"), 2)
        self.assertIn("missing-09.png", message)
        self.assertNotIn("missing-10.png", message)
        self.assertNotIn("missing-11.png", message)
        self.assertIn("extra-09.png", message)
        self.assertNotIn("extra-10.png", message)
        self.assertNotIn("extra-11.png", message)

    def test_manifest_hash_does_not_depend_on_absolute_root(self):
        manifests = []
        for fixture_name in ("first", "second"):
            fixture_root = self.root / fixture_name
            library = fixture_root / "library"
            test = fixture_root / "test"
            merged = fixture_root / "merged"
            self._write(library / "a.png", b"a")
            self._write(test / "group-a" / "b.png", b"b")
            self._write(merged / "a.png", b"a")
            self._write(merged / "b.png", b"b")

            report = audit_sources(library, test, merged, expected_groups=1)
            manifests.append(report.manifest_sha256)

        self.assertEqual(manifests[0], manifests[1])

    def test_rejects_missing_or_non_directory_roots_explicitly(self):
        file_root = self.root / "not-a-directory"
        file_root.write_bytes(b"file")
        roots = {
            "library": self.library,
            "test": self.test,
            "merged": self.merged,
        }
        for label in roots:
            for invalid_root, expected_message in (
                (self.root / f"missing-{label}", "does not exist"),
                (file_root, "not a directory"),
            ):
                with self.subTest(label=label, invalid_root=invalid_root):
                    invalid_roots = dict(roots)
                    invalid_roots[label] = invalid_root
                    with self.assertRaisesRegex(ValueError, expected_message):
                        audit_sources(
                            invalid_roots["library"],
                            invalid_roots["test"],
                            invalid_roots["merged"],
                            expected_groups=0,
                        )


class DigestTest(unittest.TestCase):
    def test_matches_hashlib_sha256_across_one_mebibyte_chunks(self):
        payload = (b"0123456789abcdef" * 65_536) + b"chunk-boundary-tail"
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "payload.png"
            path.write_bytes(payload)

            actual = digest(path)

        self.assertEqual(actual, hashlib.sha256(payload).hexdigest())


class RepairMismatchesTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.library = self.root / "data" / "library"
        self.test = self.root / "data" / "test"
        self.merged = self.root / "data" / "merged"
        self.backup = self.root / "backups" / "repair"
        self.library.mkdir(parents=True)
        (self.test / "group-a").mkdir(parents=True)
        self.merged.mkdir(parents=True)

    def _mismatched_report(self):
        (self.library / "a.png").write_bytes(b"source-a")
        (self.merged / "a.png").write_bytes(b"old-merged-a")
        return audit_sources(self.library, self.test, self.merged, expected_groups=1)

    def test_backs_up_old_merged_content_before_restoring_source_content(self):
        report = self._mismatched_report()

        repaired = repair_mismatches(report, self.backup)

        self.assertEqual(repaired, 1)
        self.assertEqual((self.backup / "a.png").read_bytes(), b"old-merged-a")
        self.assertEqual((self.merged / "a.png").read_bytes(), b"source-a")

    def test_no_mismatches_returns_zero_without_creating_backup_directory(self):
        (self.library / "a.png").write_bytes(b"same-a")
        (self.merged / "a.png").write_bytes(b"same-a")
        report = audit_sources(self.library, self.test, self.merged, expected_groups=1)

        repaired = repair_mismatches(report, self.backup)

        self.assertEqual(repaired, 0)
        self.assertFalse(self.backup.exists())

    def test_rejects_backup_directory_inside_any_data_root(self):
        for root_name in ("library", "test", "merged"):
            with self.subTest(root_name=root_name):
                fixture = self.root / root_name
                library = fixture / "library"
                test = fixture / "test"
                merged = fixture / "merged"
                library.mkdir(parents=True)
                (test / "group-a").mkdir(parents=True)
                merged.mkdir(parents=True)
                (library / "a.png").write_bytes(b"source-a")
                (merged / "a.png").write_bytes(b"old-merged-a")
                report = audit_sources(library, test, merged, expected_groups=1)
                roots = {"library": library, "test": test, "merged": merged}
                backup = roots[root_name] / "nested-backup"

                with self.assertRaisesRegex(ValueError, "backup"):
                    repair_mismatches(report, backup)

                self.assertEqual((merged / "a.png").read_bytes(), b"old-merged-a")
                self.assertFalse(backup.exists())

    def test_rejects_existing_backup_directory_without_changing_merged(self):
        report = self._mismatched_report()
        self.backup.mkdir(parents=True)
        marker = self.backup / "keep.txt"
        marker.write_bytes(b"keep")

        with self.assertRaises(FileExistsError):
            repair_mismatches(report, self.backup)

        self.assertEqual((self.merged / "a.png").read_bytes(), b"old-merged-a")
        self.assertEqual(marker.read_bytes(), b"keep")

    def test_rejects_source_or_merged_changes_since_audit_before_creating_backup(self):
        for changed_file in ("source", "merged"):
            with self.subTest(changed_file=changed_file):
                fixture = self.root / f"changed-{changed_file}"
                library = fixture / "data" / "library"
                test = fixture / "data" / "test"
                merged = fixture / "data" / "merged"
                backup = fixture / "backups" / "repair"
                library.mkdir(parents=True)
                (test / "group-a").mkdir(parents=True)
                merged.mkdir(parents=True)
                source_path = library / "a.png"
                merged_path = merged / "a.png"
                source_path.write_bytes(b"source-a")
                merged_path.write_bytes(b"old-merged-a")
                report = audit_sources(library, test, merged, expected_groups=1)
                changed_path = source_path if changed_file == "source" else merged_path
                changed_path.write_bytes(f"changed-{changed_file}".encode("ascii"))

                with self.assertRaisesRegex(ValueError, "changed since audit"):
                    repair_mismatches(report, backup)

                self.assertFalse(backup.exists())
                self.assertEqual(
                    changed_path.read_bytes(),
                    f"changed-{changed_file}".encode("ascii"),
                )
                if changed_file == "source":
                    self.assertEqual(merged_path.read_bytes(), b"old-merged-a")

    def test_second_backup_failure_occurs_before_any_merged_file_is_modified(self):
        (self.library / "a.png").write_bytes(b"source-a")
        (self.library / "b.png").write_bytes(b"source-b")
        (self.merged / "a.png").write_bytes(b"old-a")
        (self.merged / "b.png").write_bytes(b"old-b")
        report = audit_sources(self.library, self.test, self.merged, expected_groups=1)
        real_copy2 = shutil.copy2

        def fail_second_backup(source, destination, *args, **kwargs):
            destination_path = Path(destination)
            if (
                destination_path.parent == self.backup
                and destination_path.name == "b.png"
            ):
                raise OSError("simulated second backup failure")
            return real_copy2(source, destination, *args, **kwargs)

        with mock.patch(
            "ginseng_benchmark.protocol.shutil.copy2",
            side_effect=fail_second_backup,
        ):
            with self.assertRaisesRegex(OSError, "second backup"):
                repair_mismatches(report, self.backup)

        self.assertEqual((self.merged / "a.png").read_bytes(), b"old-a")
        self.assertEqual((self.merged / "b.png").read_bytes(), b"old-b")
        self.assertEqual((self.backup / "a.png").read_bytes(), b"old-a")

    def test_second_replace_failure_rolls_back_entire_batch(self):
        (self.library / "a.png").write_bytes(b"source-a")
        (self.library / "b.png").write_bytes(b"source-b")
        (self.merged / "a.png").write_bytes(b"old-a")
        (self.merged / "b.png").write_bytes(b"old-b")
        report = audit_sources(self.library, self.test, self.merged, expected_groups=1)
        real_replace = os.replace

        def fail_second_repair_replace(source, destination):
            source_path = Path(source)
            destination_path = Path(destination)
            if (
                source_path.name.startswith(".ginseng-repair-")
                and destination_path.name == "b.png"
            ):
                raise OSError("simulated second replace failure")
            return real_replace(source, destination)

        with mock.patch(
            "ginseng_benchmark.protocol.os.replace",
            side_effect=fail_second_repair_replace,
        ):
            with self.assertRaisesRegex(OSError, "second replace"):
                repair_mismatches(report, self.backup)

        self.assertEqual((self.merged / "a.png").read_bytes(), b"old-a")
        self.assertEqual((self.merged / "b.png").read_bytes(), b"old-b")
        self.assertEqual((self.backup / "a.png").read_bytes(), b"old-a")
        self.assertEqual((self.backup / "b.png").read_bytes(), b"old-b")
        self.assertEqual(list(self.merged.glob(".ginseng-*-*.tmp")), [])

    def test_digest_exception_after_replace_rolls_back_entire_batch(self):
        (self.library / "a.png").write_bytes(b"source-a")
        (self.library / "b.png").write_bytes(b"source-b")
        (self.merged / "a.png").write_bytes(b"old-a")
        (self.merged / "b.png").write_bytes(b"old-b")
        report = audit_sources(self.library, self.test, self.merged, expected_groups=1)
        real_digest = digest
        second_merged = self.merged / "b.png"

        def raise_after_second_replace(path):
            path = Path(path)
            if path == second_merged and path.read_bytes() == b"source-b":
                raise PermissionError("simulated post-replace digest denial")
            return real_digest(path)

        with mock.patch(
            "ginseng_benchmark.protocol.digest",
            side_effect=raise_after_second_replace,
        ):
            with self.assertRaisesRegex(PermissionError, "digest denial"):
                repair_mismatches(report, self.backup)

        self.assertEqual((self.merged / "a.png").read_bytes(), b"old-a")
        self.assertEqual((self.merged / "b.png").read_bytes(), b"old-b")
        self.assertEqual((self.backup / "a.png").read_bytes(), b"old-a")
        self.assertEqual((self.backup / "b.png").read_bytes(), b"old-b")
        self.assertEqual(list(self.merged.glob(".ginseng-*-*.tmp")), [])

    def test_replace_failure_preserves_backup_and_cleans_temporary_file(self):
        report = self._mismatched_report()

        with mock.patch("os.replace", side_effect=OSError("simulated replace failure")):
            with self.assertRaisesRegex(
                RuntimeError,
                "repair failed.*replace failure.*rollback failed",
            ):
                repair_mismatches(report, self.backup)

        self.assertEqual((self.backup / "a.png").read_bytes(), b"old-merged-a")
        self.assertEqual((self.merged / "a.png").read_bytes(), b"old-merged-a")
        self.assertEqual(list(self.merged.glob(".ginseng-repair-*.tmp")), [])

    def test_staging_verification_failure_preserves_backup_and_cleans_temp(self):
        report = self._mismatched_report()
        real_copy2 = shutil.copy2

        def corrupt_staging_copy(source, destination, *args, **kwargs):
            destination_path = Path(destination)
            if destination_path.name.startswith(".ginseng-repair-"):
                destination_path.write_bytes(b"corrupted-staging-copy")
                return str(destination_path)
            return real_copy2(source, destination, *args, **kwargs)

        with mock.patch(
            "ginseng_benchmark.protocol.shutil.copy2",
            side_effect=corrupt_staging_copy,
        ):
            with self.assertRaisesRegex(OSError, "staging verification"):
                repair_mismatches(report, self.backup)

        self.assertEqual((self.backup / "a.png").read_bytes(), b"old-merged-a")
        self.assertEqual((self.merged / "a.png").read_bytes(), b"old-merged-a")
        self.assertEqual(list(self.merged.glob(".ginseng-repair-*.tmp")), [])

    def test_post_replace_verification_failure_rolls_back_from_backup(self):
        report = self._mismatched_report()
        merged_path = self.merged / "a.png"
        real_digest = digest

        def fail_only_post_replace_verification(path):
            path = Path(path)
            if path == merged_path and path.read_bytes() == b"source-a":
                return "0" * 64
            return real_digest(path)

        with mock.patch(
            "ginseng_benchmark.protocol.digest",
            side_effect=fail_only_post_replace_verification,
        ):
            with self.assertRaisesRegex(OSError, "repair verification"):
                repair_mismatches(report, self.backup)

        self.assertEqual((self.backup / "a.png").read_bytes(), b"old-merged-a")
        self.assertEqual(merged_path.read_bytes(), b"old-merged-a")
        self.assertEqual(list(self.merged.glob(".ginseng-*-*.tmp")), [])

    def test_rejects_source_and_merged_resolving_to_same_file(self):
        report = self._mismatched_report()
        mismatch = report.mismatches[0]
        unsafe_merged = replace(
            mismatch.merged,
            path=mismatch.source.path,
            size=mismatch.source.size,
            sha256=mismatch.source.sha256,
        )
        unsafe_report = replace(
            report,
            mismatches=(replace(mismatch, merged=unsafe_merged),),
        )

        with self.assertRaisesRegex(ValueError, "same file"):
            repair_mismatches(unsafe_report, self.backup)

        self.assertFalse(self.backup.exists())
        self.assertEqual((self.library / "a.png").read_bytes(), b"source-a")


class ImageFilesTest(unittest.TestCase):
    def test_filters_suffixes_case_insensitively_and_sorts_by_relative_path(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "z.JPG").write_bytes(b"z")
            (root / "nested").mkdir()
            (root / "nested" / "a.png").write_bytes(b"a")
            (root / "ignored.txt").write_bytes(b"ignored")

            paths = image_files(root)

        self.assertEqual(
            [path.relative_to(root).as_posix() for path in paths],
            ["nested/a.png", "z.JPG"],
        )


class AuditDatasetScriptTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.library = self.root / "library"
        self.test = self.root / "test"
        self.merged = self.root / "merged"
        self.library.mkdir()
        (self.test / "group-a").mkdir(parents=True)
        self.merged.mkdir()
        (self.library / "a.png").write_bytes(b"source-a")
        (self.test / "group-a" / "b.png").write_bytes(b"source-b")
        (self.merged / "a.png").write_bytes(b"different-a")
        (self.merged / "b.png").write_bytes(b"source-b")
        self.env_path = self.root / ".env"
        self.env_path.write_text(
            f"LIBRARY_BINARY={self.library}\n"
            f"TEST_BINARY_ROOT={self.test}\n"
            f"MERGED_GALLERY={self.merged}\n"
            "HF_TOKEN=test-only-placeholder\n",
            encoding="utf-8",
        )
        self.output_path = self.root / "audit.json"
        self.script_path = Path(__file__).parents[1] / "scripts" / "audit_dataset.py"

    def _run_script(self, *extra_arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                str(self.script_path),
                "--env",
                str(self.env_path),
                "--output",
                str(self.output_path),
                "--expected-groups",
                "1",
                "--expected-library-count",
                "1",
                "--expected-test-count",
                "1",
                "--expected-merged-count",
                "2",
                *extra_arguments,
            ],
            cwd=self.script_path.parents[1],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_writes_json_and_allows_unspecified_mismatch_count(self):
        result = self._run_script()

        self.assertEqual(result.returncode, 0, result.stderr)
        raw_output = self.output_path.read_text(encoding="utf-8")
        payload = json.loads(raw_output)
        self.assertEqual(
            payload["report"],
            {
                "group_count": 1,
                "library_count": 1,
                "merged_count": 2,
                "mismatch_count": 1,
                "record_count": 2,
                "test_count": 1,
            },
        )
        self.assertEqual([item["name"] for item in payload["mismatches"]], ["a.png"])
        self.assertEqual(len(payload["records"]), 2)
        self.assertEqual(len(payload["manifest_sha256"]), 64)
        self.assertEqual(
            {
                (record["source"], record["name"]): record["path"]
                for record in payload["records"]
            },
            {
                ("library", "a.png"): "a.png",
                ("test", "b.png"): "group-a/b.png",
            },
        )
        self.assertEqual(payload["mismatches"][0]["source"]["path"], "a.png")
        self.assertEqual(payload["mismatches"][0]["merged"]["path"], "a.png")
        for data_root in (self.library, self.test, self.merged):
            escaped_root = json.dumps(str(data_root), ensure_ascii=False)[1:-1]
            self.assertNotIn(escaped_root, raw_output)
        self.assertNotIn("HF_TOKEN", raw_output)
        self.assertNotIn("test-only-placeholder", raw_output)

    def test_returns_nonzero_when_expected_mismatch_count_is_wrong(self):
        result = self._run_script("--expected-mismatches", "0")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mismatch", result.stderr.casefold())
        self.assertTrue(self.output_path.is_file())


class RepairMergedGalleryScriptTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.library = self.root / "data" / "library"
        self.test = self.root / "data" / "test"
        self.merged = self.root / "data" / "merged"
        self.backup = self.root / "backups" / "repair"
        self.library.mkdir(parents=True)
        (self.test / "group-a").mkdir(parents=True)
        self.merged.mkdir(parents=True)
        (self.library / "a.png").write_bytes(b"source-a")
        (self.merged / "a.png").write_bytes(b"old-merged-a")
        self.env_path = self.root / ".env"
        self.env_path.write_text(
            f"LIBRARY_BINARY={self.library}\n"
            f"TEST_BINARY_ROOT={self.test}\n"
            f"MERGED_GALLERY={self.merged}\n"
            "HF_TOKEN=test-only-placeholder\n",
            encoding="utf-8",
        )
        self.script_path = (
            Path(__file__).parents[1] / "scripts" / "repair_merged_gallery.py"
        )

    def _run_script(self, *extra_arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                str(self.script_path),
                "--env",
                str(self.env_path),
                "--expected-groups",
                "1",
                "--expected-library-count",
                "1",
                "--expected-test-count",
                "0",
                "--expected-merged-count",
                "1",
                "--expected-mismatches",
                "1",
                *extra_arguments,
            ],
            cwd=self.script_path.parents[1],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_default_dry_run_reports_count_without_writing_data_or_backup(self):
        result = self._run_script("--backup-dir", str(self.backup))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("would_repair=1", result.stdout)
        self.assertEqual((self.merged / "a.png").read_bytes(), b"old-merged-a")
        self.assertFalse(self.backup.exists())
        self.assertNotIn("test-only-placeholder", result.stdout + result.stderr)

    def test_apply_requires_explicit_backup_directory(self):
        result = self._run_script("--apply")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--backup-dir is required with --apply", result.stderr)
        self.assertEqual((self.merged / "a.png").read_bytes(), b"old-merged-a")

    def test_apply_repairs_temp_fixture_and_reports_absolute_backup_directory(self):
        result = self._run_script(
            "--apply",
            "--backup-dir",
            str(self.backup),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("repaired=1", result.stdout)
        self.assertIn(f"backup_dir={self.backup.resolve()}", result.stdout)
        self.assertEqual((self.backup / "a.png").read_bytes(), b"old-merged-a")
        self.assertEqual((self.merged / "a.png").read_bytes(), b"source-a")

    def test_mismatch_expectation_error_refuses_to_write(self):
        result = self._run_script(
            "--expected-mismatches",
            "0",
            "--apply",
            "--backup-dir",
            str(self.backup),
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mismatch count mismatch", result.stderr)
        self.assertEqual((self.merged / "a.png").read_bytes(), b"old-merged-a")
        self.assertFalse(self.backup.exists())
        self.assertNotIn("test-only-placeholder", result.stdout + result.stderr)

    def test_dataset_count_error_refuses_to_write(self):
        result = self._run_script(
            "--expected-library-count",
            "2",
            "--apply",
            "--backup-dir",
            str(self.backup),
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("library count mismatch", result.stderr)
        self.assertEqual((self.merged / "a.png").read_bytes(), b"old-merged-a")
        self.assertFalse(self.backup.exists())

    def test_dry_run_and_apply_are_mutually_exclusive(self):
        result = self._run_script(
            "--dry-run",
            "--apply",
            "--backup-dir",
            str(self.backup),
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.merged / "a.png").read_bytes(), b"old-merged-a")
        self.assertFalse(self.backup.exists())


if __name__ == "__main__":
    unittest.main()
