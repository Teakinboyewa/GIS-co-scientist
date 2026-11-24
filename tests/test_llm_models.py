"""Every LLM call takes its model and reasoning effort from utils/llm_models.toml by name."""
import ast
import io
import re
import tokenize
import unittest
from pathlib import Path

from utils import llm_models

ROOT = Path(__file__).resolve().parents[1]
SKIP = ("_v1.py", "_v2.py")
LOOKUPS = {"llm_model", "llm_reasoning", "llm_effort", "_llm_model", "_llm_reasoning"}
# A model or effort chosen in code instead of by name.
HARDCODED_MODEL_RE = re.compile(r"""\bmodel(?:_name)?\s*=\s*["'](?:gpt|o\d|claude)[-\w.]*["']""")
HARDCODED_EFFORT_RE = re.compile(
    r"""(?<![\w.])reasoning_effort=(?:["']\w+["']|(?:helper\.)?get_ui_reasoning_effort\(\))""")


def _sources():
    for folder in ("agents", "WebUI", "utils"):
        for path in (ROOT / folder).rglob("*.py"):
            if not path.name.endswith(SKIP) and "__pycache__" not in path.parts:
                yield path, path.read_text(encoding="utf-8", errors="replace")


def _func_name(node):
    f = node.func
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None


def _lookup(node):
    """('llm_model', 'x.y') for a call like llm_model("x.y"), else None."""
    if (isinstance(node, ast.Call) and _func_name(node) in LOOKUPS and len(node.args) == 1
            and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
        return _func_name(node), node.args[0].value
    return None


def _code_lookups():
    """Every (kind, name, location) lookup in real code (comments don't count)."""
    for path, text in _sources():
        for node in ast.walk(ast.parse(text)):
            hit = _lookup(node)
            if hit:
                yield hit[0], hit[1], f"{path.relative_to(ROOT)}:{node.lineno}"


def _code_lines(text):
    """Source lines with comments removed."""
    lines = text.splitlines()
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT:
            row, col = tok.start
            lines[row - 1] = lines[row - 1][:col]
    return lines


class LLMModelRegistry(unittest.TestCase):
    def test_every_name_used_in_code_is_configured(self):
        configured = llm_models.all_settings()
        missing = sorted({f"{name}  ({where})" for _, name, where in _code_lookups()
                          if name not in configured})
        self.assertEqual(missing, [], "LLM call names missing from llm_models.toml")

    def test_every_configured_name_is_used(self):
        used = {name for _, name, _ in _code_lookups()}
        self.assertEqual(sorted(set(llm_models.all_settings()) - used), [],
                         "llm_models.toml entries no code asks for")

    def test_calls_pass_model_and_effort_of_the_same_name(self):
        """A call given llm_model("x") as an argument also gets **llm_reasoning("x")."""
        problems = []
        for path, text in _sources():
            for node in ast.walk(ast.parse(text)):
                if not isinstance(node, ast.Call):
                    continue
                where = f"{path.relative_to(ROOT)}:{node.lineno}"
                if _lookup(node.func):           # llm_model("x")(...) calls a string
                    problems.append(f"{where}: result of {_lookup(node.func)[0]} is called")
                    continue
                args = list(node.args) + [k.value for k in node.keywords if k.arg is not None]
                models = {_lookup(a)[1] for a in args if _lookup(a) and _lookup(a)[0] in ("llm_model", "_llm_model")}
                if not models:
                    continue
                efforts = {_lookup(k.value)[1] for k in node.keywords
                           if k.arg is None and _lookup(k.value)
                           and _lookup(k.value)[0] in ("llm_reasoning", "_llm_reasoning")}
                if efforts != models:
                    problems.append(f"{where}: model {sorted(models)} but **llm_reasoning {sorted(efforts)}")
        self.assertEqual(problems, [])

    def test_lookups_never_sit_in_comments(self):
        found = []
        for path, text in _sources():
            for tok in tokenize.generate_tokens(io.StringIO(text).readline):
                if tok.type == tokenize.COMMENT and re.search(r"\*\*\s*\w*llm_reasoning\(|ning\(\"", tok.string):
                    found.append(f"{path.relative_to(ROOT)}:{tok.start[0]}: {tok.string.strip()}")
        self.assertEqual(found, [])

    def test_no_call_site_hardcodes_a_model_or_effort(self):
        found = []
        for path, text in _sources():
            for i, line in enumerate(_code_lines(text), 1):
                if HARDCODED_MODEL_RE.search(line) or HARDCODED_EFFORT_RE.search(line):
                    found.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()}")
        self.assertEqual(found, [], "use llm_model('<name>') / **llm_reasoning('<name>')")

    def test_unknown_name_fails_loudly(self):
        for lookup in (llm_models.llm_model, llm_models.llm_effort, llm_models.llm_reasoning):
            with self.assertRaises(KeyError):
                lookup("no_such.call")

    def test_settings_are_well_formed(self):
        settings = llm_models.all_settings()
        self.assertTrue(settings)
        for name, s in settings.items():
            self.assertRegex(name, r"^\w+\.\w+$")
            self.assertTrue(s["model"].strip())
            self.assertIn(s["reasoning_effort"], (None,) + llm_models.EFFORTS)

    def test_reasoning_kwargs_are_empty_without_an_effort(self):
        # The chat-completion helpers forward kwargs as-is, so "no effort" must be
        # no argument at all, never reasoning_effort=None.
        for name, s in llm_models.all_settings().items():
            kwargs = llm_models.llm_reasoning(name)
            expected = {"reasoning_effort": s["reasoning_effort"]} if s["reasoning_effort"] else {}
            self.assertEqual(kwargs, expected, name)

    def test_bad_entries_are_rejected(self):
        for bad in ({"model": ""}, {"model": "m", "reasoning_effort": "extreme"},
                    {"model": "m", "temperature": 0}):
            with self.assertRaises(ValueError):
                llm_models._entry("x.y", bad)
        self.assertEqual(llm_models._entry("x.y", "m"), {"model": "m", "reasoning_effort": None})


if __name__ == "__main__":
    unittest.main()
