# Foundations of Reinforcement Learning with Applications in Finance — Notes

A condensed reference for the manuscript in `book/`: the central equations, what every symbol
means, and how each equation is actually solved — analytically, by dynamic programming, or by
reinforcement learning — with pointers to the implementing function in the `rl/` package.

These pages are built from the Markdown in `notes/`. That Markdown is written to be read
directly as plain text too: there is no LaTeX in it, and all the math is Unicode.

```{toctree}
:maxdepth: 2
:caption: Contents

notes/book-summary
notes/book-summary/00-notation
notes/book-summary/01-processes-and-planning
notes/book-summary/02-financial-applications
notes/book-summary/03-rl-algorithms
notes/book-summary/04-finishing-touches
notes/book-summary/05-appendices
notes/book-summary/06-cross-cutting
```

## Where to start

- New to the material — read [the overview](notes/book-summary.md), then the modules in order.
- Looking up a symbol — [Notation and Variable Glossary](notes/book-summary/00-notation.md).
- Know the equation, want the method or the code —
  [Cross-Cutting Reference](notes/book-summary/06-cross-cutting.md).

## Rebuilding

```
make -C docs html          # output in docs/_build/html
make -C docs clean html    # full rebuild
```

The toolchain is Sphinx + MyST + furo in the repo's `.venv`, entirely separate from the Nix
toolchain that builds the book PDF.
