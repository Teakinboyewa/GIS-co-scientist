"""Only page assets may be downloaded from WebUI/: never data, code or databases."""
import ast
import os
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load():
    tree = ast.parse((ROOT / 'WebUI/app.py').read_text(encoding='utf-8'))
    names = {'_STATIC_EXTENSIONS', '_STATIC_SUBDIRS', '_is_static_asset'}
    body = [n for n in tree.body if getattr(n, 'name', None) in names
            or (isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', None) in names)]
    scope = {'os': os}
    exec(compile(ast.Module(body=body, type_ignores=[]), 'app.py', 'exec'), scope)
    return scope['_is_static_asset']


class StaticFiles(unittest.TestCase):
    def setUp(self):
        self.allowed = load()

    def test_private_files_are_refused(self):
        for f in ('conversations.db', 'conversations.db-wal', 'app.py', 'uploads/a.csv', 'uploads/a.png',
                  'conversation_DBs/x.db', 'conversations/x.html', '.env', 'ui/.env', '../utils/config.ini',
                  'icon/../uploads/a.png', 'icon\\..\\uploads\\a.png', '/abs.html', 'key.keys', 'x'):
            with self.subTest(f):
                self.assertFalse(self.allowed(f))

    def test_every_asset_the_page_links_to_is_served(self):
        html = (ROOT / 'WebUI/index.html').read_text(encoding='utf-8')
        links = {m.split('?')[0].lstrip('/') for m in re.findall(r'(?:src|href)="([^"#{]+)"', html)
                 if '//' not in m and not m.startswith(('data:', 'javascript:', 'mailto:'))}
        files = [f for f in links if '.' in f]
        self.assertGreater(len(files), 5)
        for f in files:
            with self.subTest(f):
                self.assertTrue(self.allowed(f))

    def test_every_tracked_ui_asset_is_served(self):
        for d in ('icon', 'ui'):
            for p in (ROOT / 'WebUI' / d).rglob('*'):
                if p.is_file() and p.suffix in ('.js', '.css', '.html', '.png', '.svg'):
                    self.assertTrue(self.allowed(p.relative_to(ROOT / 'WebUI').as_posix()), p)


if __name__ == '__main__':
    unittest.main()
