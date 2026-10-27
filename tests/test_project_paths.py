"""Check public names and links without renaming Python or archived artifacts."""
import json
from pathlib import Path
import re
import unittest
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]


class ProjectPathTests(unittest.TestCase):
    def test_public_artifact_names_have_no_underscores(self):
        files = [path for path in ROOT.iterdir()
                 if path.is_file() and path.suffix in {'.md', '.json', '.tex', '.pdf'}]
        for directory in ('figures', 'output'):
            files.extend(path for path in (ROOT / directory).rglob('*')
                         if path.is_file() and path.suffix in {'.md', '.json', '.pdf', '.png'})
        for path in files:
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertNotIn('_', str(path.relative_to(ROOT)))

    def test_configuration_cross_references_resolve(self):
        expected = {
            'experiment-01.json', 'experiment-02.json', 'experiment-02-legacy.json',
            'experiment-02-tracking.json', 'experiment-03.json', 'high-rank-sensitivity.json',
        }
        self.assertEqual({path.name for path in ROOT.glob('*.json')}, expected)
        for name in expected:
            config = json.loads((ROOT / name).read_text())
            for key in ('base_config', 'paper_config', 'group_config'):
                if key in config:
                    with self.subTest(config=name, key=key):
                        self.assertTrue((ROOT / config[key]).is_file(), config[key])
                        self.assertNotIn('_', config[key])

    def test_documentation_local_links_resolve(self):
        documents = list(ROOT.glob('*.md')) + list((ROOT / 'output').rglob('README.md'))
        for document in documents:
            for link in re.findall(r'\[[^\]]+\]\(([^)]+)\)', document.read_text()):
                target = urlsplit(link.strip('<>'))
                if target.scheme or not target.path:
                    continue
                with self.subTest(document=document.relative_to(ROOT), link=link):
                    self.assertTrue((document.parent / unquote(target.path)).exists())

    def test_report_figure_paths_resolve(self):
        report = ROOT / 'group sparsity project report.tex'
        figures = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', report.read_text())
        self.assertGreaterEqual(len(figures), 5)
        for name in figures:
            with self.subTest(figure=name):
                self.assertNotIn('_', name)
                self.assertTrue((ROOT / name).is_file())


if __name__ == '__main__':
    unittest.main()
