"""Re-run path rewriting: hardcoded output directories point at the re-run folder.

Code generated on one machine is re-run on another (a Windows desktop session
re-run on the Linux server), so every output root the app has ever used must
be recognised. A root that slips through is not an error on Linux -- the
backslashes are legal filename characters -- so the files land in one oddly
named directory and the UI reports them missing.
"""
import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "x")

from WebUI.app import _studio_rewrite_output_paths as rewrite


OUT = "/home/u/.local/share/gis-co-scientist/outputs/abc/def/rerun_1234abcd/attempt_1"
WIN_RUN = r"C:\Users\me\AppData\Local\gis-co-scientist\outputs\abc\def\test_d1a355d5"


class RerunPathRewriteTests(unittest.TestCase):
    def assertRewritten(self, code, expected):
        new, count = rewrite(code, OUT)
        self.assertEqual(count, 1, new)
        self.assertEqual(new, expected)

    def test_windows_durable_root_run_directory(self):
        self.assertRewritten(f'out_dir = r"{WIN_RUN}"', f'out_dir = r"{OUT}"')

    def test_windows_durable_root_keeps_filename(self):
        self.assertRewritten(f'p = r"{WIN_RUN}\\b04.tif"', f'p = r"{OUT}/b04.tif"')

    def test_windows_durable_root_escaped_backslashes(self):
        escaped = WIN_RUN.replace("\\", "\\\\")
        self.assertRewritten(f'p = "{escaped}\\\\b04.tif"', f'p = "{OUT}/b04.tif"')

    def test_windows_durable_root_alone(self):
        self.assertRewritten(
            r'root = r"C:\Users\me\AppData\Local\gis-co-scientist\outputs"',
            f'root = r"{OUT}"')

    def test_posix_durable_root(self):
        self.assertRewritten(
            'p = "/Users/me/Library/Application Support/gis-co-scientist/outputs/abc/def/test_ab12/x.csv"',
            f'p = "{OUT}/x.csv"')

    def test_legacy_root_still_rewritten(self):
        self.assertRewritten(
            r'p = r"C:\Users\me\AppData\Local\Temp\GISHBStudio\abc\def\test_ab12\x.csv"',
            f'p = r"{OUT}/x.csv"')

    def test_unrelated_paths_untouched(self):
        code = 'url = "https://example.com/data/file.tif"\nlocal = "/home/u/data/x.tif"'
        self.assertEqual(rewrite(code, OUT), (code, 0))


if __name__ == "__main__":
    unittest.main()
