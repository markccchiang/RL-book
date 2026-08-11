"""Sphinx configuration for the notes under ``notes/``.

The Markdown lives in ``notes/`` and is the canonical copy: it carries no
Sphinx-only markup, so it stays readable on its own in any Markdown viewer.
Sphinx needs its sources inside the source directory to resolve the relative
``*.md`` links between them, so the notes are mirrored into ``docs/notes/``
below, on every build. That mirror is generated -- edit ``notes/``, never it.

(A symlink does not work here: MyST resolves a link through the file's real
path, which lands outside the source directory and leaves every cross-file
reference dangling.)

Build with::

    make -C docs html          # or: .venv/bin/sphinx-build -b html docs docs/_build/html
"""

import filecmp
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).parent
_SRC = _HERE.parent / 'notes'
_MIRROR = _HERE / 'notes'


def _sync_notes() -> None:
    """Mirror notes/**/*.md into docs/notes/, dropping anything stale.

    Files are copied only when they differ, so unchanged pages keep their
    mtime and Sphinx's incremental build stays incremental.
    """
    wanted = {p.relative_to(_SRC) for p in _SRC.rglob('*.md')}
    for rel in wanted:
        dst = _MIRROR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or not filecmp.cmp(_SRC / rel, dst, shallow=False):
            shutil.copy2(_SRC / rel, dst)
    for stale in _MIRROR.rglob('*.md'):
        if stale.relative_to(_MIRROR) not in wanted:
            stale.unlink()


_sync_notes()

project = 'RL-book Notes'
author = 'Cheng-Chin Chiang'
copyright = 'Notes on Rao & Jelvis, Foundations of Reinforcement Learning with Applications in Finance'

sys.path.insert(0, str(_HERE / '_ext'))

extensions = [
    'myst_parser',
    'sphinx.ext.mathjax',
    'mathify',        # docs/_ext/mathify.py: Unicode maths -> LaTeX
]

# The notes are Markdown only; the .org files in notes/ are ignored.
source_suffix = {'.md': 'markdown'}
root_doc = 'index'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'README.md']

# Generate anchors for h1-h3 so the cross-file "...md#chapter-4--dynamic-..."
# links in the notes resolve to real targets.
myst_heading_anchors = 3
myst_enable_extensions = ['deflist', 'dollarmath', 'smartquotes',
                          'substitution']

# MathJax is vendored under _static so equations render with no network:
# the v3 SVG bundle compiles the TeX fonts in, so it needs no font files.
mathjax_path = 'mathjax/tex-mml-svg.js'
mathjax3_config = {
    'options': {'enableMenu': False},   # the menu lazy-loads a11y components
    'tex': {'packages': {'[+]': ['ams']}},
}

html_theme = 'furo'
html_title = 'Foundations of RL with Applications in Finance — Notes'
html_static_path = ['_static']
html_css_files = ['custom.css']
