import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

from ginseng_benchmark.env import load_env_file


class LoadEnvFileTest(unittest.TestCase):
    def test_preserves_windows_path_ignores_comments_and_keeps_empty_token(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            env_path = Path(temporary_directory) / ".env"
            env_path.write_text(
                "# local\nLIBRARY_BINARY=D:/data/library\nHF_TOKEN=\n",
                encoding="utf-8",
            )

            values = load_env_file(env_path)

        self.assertEqual(
            values,
            {
                "LIBRARY_BINARY": "D:/data/library",
                "HF_TOKEN": "",
            },
        )

    def test_handles_utf8_bom_blank_lines_and_indented_comments(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            env_path = Path(temporary_directory) / ".env"
            env_path.write_text(
                "\ufeff\n   # indented comment\nBOM_KEY=loaded\n",
                encoding="utf-8",
            )

            values = load_env_file(env_path)

        self.assertEqual(values, {"BOM_KEY": "loaded"})

    def test_strips_keys_and_values_and_removes_paired_quotes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            env_path = Path(temporary_directory) / ".env"
            env_path.write_text(
                "  DOUBLE_KEY  =  \"double value\"  \n"
                "  SINGLE_KEY  =  'single value'  \n",
                encoding="utf-8",
            )

            values = load_env_file(env_path)

        self.assertEqual(
            values,
            {
                "DOUBLE_KEY": "double value",
                "SINGLE_KEY": "single value",
            },
        )

    def test_splits_values_only_on_the_first_equals_sign(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            env_path = Path(temporary_directory) / ".env"
            env_path.write_text(
                "SIGNED_URL=https://example.test/download?token=a=b=c\n",
                encoding="utf-8",
            )

            values = load_env_file(env_path)

        self.assertEqual(
            values,
            {"SIGNED_URL": "https://example.test/download?token=a=b=c"},
        )

    def test_rejects_missing_equals_with_line_number_without_echoing_value(self):
        sensitive_line = "placeholder-sensitive-value-without-separator"
        with tempfile.TemporaryDirectory() as temporary_directory:
            env_path = Path(temporary_directory) / ".env"
            env_path.write_text(
                f"VALID=value\n\n   # comment\n{sensitive_line}\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError) as error:
                load_env_file(env_path)

        message = str(error.exception)
        self.assertIn("line 4", message)
        self.assertNotIn(sensitive_line, message)

    def test_does_not_modify_process_environment(self):
        unique_key = f"GINSENG_BENCHMARK_TEST_{uuid.uuid4().hex.upper()}"

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(unique_key, None)
            environment_before = dict(os.environ)

            with tempfile.TemporaryDirectory() as temporary_directory:
                env_path = Path(temporary_directory) / ".env"
                env_path.write_text(
                    f"{unique_key}=temporary-value\n",
                    encoding="utf-8",
                )

                values = load_env_file(env_path)

            self.assertEqual(values, {unique_key: "temporary-value"})
            self.assertEqual(dict(os.environ), environment_before)


if __name__ == "__main__":
    unittest.main()
