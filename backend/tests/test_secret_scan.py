import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.scan_staged_secrets import (
    contains_secret,
    decode_text,
    main,
    scan_staged,
)


class SecretScanTests(unittest.TestCase):
    def test_detects_tokens_and_long_assigned_credentials(self):
        self.assertTrue(contains_secret("token=" + "sk-" + "a" * 24))
        self.assertTrue(contains_secret("API_KEY=" + "a" * 24))
        self.assertTrue(contains_secret("Authorization: Bearer " + "a" * 40))
        self.assertTrue(contains_secret("AWS_ACCESS_KEY_ID=AKIA" + "A" * 16))
        self.assertTrue(contains_secret("API_KEY=example-live-" + "a" * 24))
        self.assertTrue(contains_secret("PASSWORD=$plaintext-" + "a" * 24))

    def test_allows_environment_references_and_explicit_placeholders(self):
        self.assertFalse(contains_secret("API_KEY=${DEEPSEEK_API_KEY}"))
        self.assertFalse(
            contains_secret("API_KEY=replace-with-your-deepseek-key")
        )
        self.assertFalse(
            contains_secret("PASSWORD=replace-with-a-long-random-password")
        )
        self.assertFalse(contains_secret("API_KEY=$DEEPSEEK_API_KEY"))

    def test_decodes_utf16_and_explicitly_skips_binary(self):
        secret = "API_KEY=" + "a" * 24
        self.assertEqual(decode_text(secret.encode("utf-16")), secret)
        self.assertIsNone(decode_text(b"\x00\x01binary"))

    def test_scans_staged_untracked_index_snapshot_with_space_in_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            path = root / "new secret.txt"
            staged_secret = "API_KEY=" + "a" * 24
            path.write_text(staged_secret, encoding="utf-8")
            subprocess.run(["git", "add", "--", path.name], cwd=root, check=True)
            path.write_text(
                "API_KEY=replace-with-your-deepseek-key",
                encoding="utf-8",
            )

            self.assertEqual(scan_staged(root), [path.name])

            subprocess.run(["git", "add", "--", path.name], cwd=root, check=True)
            self.assertEqual(scan_staged(root), [])

    def test_main_exits_one_and_prints_paths_without_secret_content(self):
        with patch(
            "scripts.scan_staged_secrets.scan_staged",
            return_value=["path with space.txt"],
        ), patch("builtins.print") as output:
            with self.assertRaises(SystemExit) as raised:
                main(Path("."))

        self.assertEqual(raised.exception.code, 1)
        rendered = output.call_args.args[0]
        self.assertIn("path with space.txt", rendered)
        self.assertNotIn("API_KEY", rendered)


if __name__ == "__main__":
    unittest.main()
