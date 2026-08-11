"""Typeset the notes' plain-Unicode math as real LaTeX, at build time.

``notes/*.md`` deliberately contains no LaTeX: equations are Unicode inside
fenced blocks, and inline formulas are code spans. That keeps the files
readable as plain text in any editor or Markdown viewer. This extension
translates that back into LaTeX for the HTML build only -- nothing on disk
changes -- so the site typesets like a LaTeX document.

It runs on the ``source-read`` event and touches only documents under
``notes/``. Two jobs:

1. every fenced block in the notes is an equation -> ``$$ ... $$``;
2. inline code spans that hold math -> ``$ ... $``, leaving code identifiers
   (``evaluate_mrp``, ``rl/td.py``, ``FunctionApprox``) as literal code.

Telling those apart is a heuristic, so it is deliberately conservative and
its two judgement calls are explicit below: ``_FORCE_CODE`` and
``_FORCE_MATH`` override it for anything it gets wrong.
"""

from __future__ import annotations

import re

# --- symbol translation ----------------------------------------------------

_GREEK = {
    'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta', 'ε': 'epsilon',
    'ζ': 'zeta', 'η': 'eta', 'θ': 'theta', 'ι': 'iota', 'κ': 'kappa',
    'λ': 'lambda', 'μ': 'mu', 'ν': 'nu', 'ξ': 'xi', 'π': 'pi', 'ρ': 'rho',
    'σ': 'sigma', 'τ': 'tau', 'υ': 'upsilon', 'φ': 'phi', 'χ': 'chi',
    'ψ': 'psi', 'ω': 'omega', 'Γ': 'Gamma', 'Δ': 'Delta', 'Θ': 'Theta',
    'Λ': 'Lambda', 'Ξ': 'Xi', 'Π': 'Pi', 'Φ': 'Phi',
    'Ψ': 'Psi', 'Ω': 'Omega',
}

_SYMBOL = {
    '∈': r'\in ', '∉': r'\notin ', '≤': r'\leq ', '≥': r'\geq ',
    '≠': r'\neq ', '≈': r'\approx ', '≡': r'\equiv ', '∝': r'\propto ',
    '±': r'\pm ', '×': r'\times ', '·': r'\cdot ', '∘': r'\circ ',
    '⊕': r'\oplus ', '⊖': r'\ominus ', '∪': r'\cup ', '∩': r'\cap ',
    '⊂': r'\subset ', '⊆': r'\subseteq ', '∅': r'\emptyset ',
    '∀': r'\forall ', '∃': r'\exists ', '†': r'\dagger ',
    '→': r'\to ', '←': r'\leftarrow ', '↦': r'\mapsto ',
    '⟹': r'\Rightarrow ', '⟺': r'\iff ',
    '∞': r'\infty ', '∇': r'\nabla ', '∂': r'\partial ',
    '∑': r'\sum ', 'Σ': r'\sum ', '∏': r'\prod ', '∫': r'\int ',
    'ℝ': r'\mathbb{R}', 'ℤ': r'\mathbb{Z}', 'ℕ': r'\mathbb{N}',
    '‖': r'\Vert ', '~': r'\sim ', '...': r'\ldots ', '✓': r'\checkmark ',
}

# Multi-letter names that must not be typeset as a product of italic letters.
_OPERATOR = ['argmax', 'argmin', 'max', 'min', 'log', 'exp', 'lim', 'sup',
             'inf', 'det', 'dim', 'sign', 'ln']
_UPRIGHT = ['Count', 'Obj', 'Out', 'KL', 'FIM', 'RMSE', 'IG', 'Beta',
            'Poisson', 'SP', 'SB', 'GMVP', 'PBE', 'TDE', 'BE']

# Whole phrases that are prose sitting inside an equation.
_PHRASE = {
    'Optimal bid-ask spread': r'\text{Optimal bid-ask spread}\quad ',
    'Optimal pseudo-mid': r'\text{Optimal pseudo-mid}\quad ',
}

# Spans the heuristic would get wrong.
_FORCE_CODE = {'*', '^', '_', '.md', 'structure', 'table', 'sample', 'update'}
_FORCE_MATH: set[str] = set()

_MATH_CHARS = set(_GREEK) | set('∈∉≤≥≠≈≡∝±×·∘⊕⊖∪∩⊂⊆∅∀∃†→←↦⟹⟺∞∇∂∑Σ∏∫ℝℤℕ‖√̂̄')
_MATH_WORD = re.compile(
    r'^(max|min|log|exp|lim|sup|inf|argmax|argmin|det|dim|sign|ln|and|or|for'
    r'|all|where|otherwise|if|else|then|with|the|is|to|st|dt|ds)$')


def _is_code(span: str) -> bool:
    """True when an inline code span is code rather than math."""
    if span in _FORCE_MATH:
        return False
    if span in _FORCE_CODE or span.startswith('\\'):
        return True                                   # literal LaTeX examples
    if re.fullmatch(r'[\W_]+', span):
        return True                                   # bare punctuation
    if re.search(r'\.(py|md|template)\b', span) or span.endswith('/'):
        return True
    if set(span) & _MATH_CHARS or '^' in span or '_{' in span:
        return False
    if '/' in span:
        return True                                   # a path, not a fraction
    # Otherwise: a run of three or more lowercase letters that is not a maths
    # operator marks an identifier (``evaluate_mrp``) rather than a formula
    # (``S_t``, ``dR_t = rR_t dt``).
    return any(not _MATH_WORD.match(w) for w in re.findall(r'[a-z]{3,}', span))


def _sqrt(s: str) -> str:
    """√(...) -> \\sqrt{...}, matching the closing parenthesis by counting."""
    while (k := s.find('√')) != -1:
        if k + 1 < len(s) and s[k + 1] == '(':
            depth, j = 0, k + 1
            while j < len(s):
                depth += (s[j] == '(') - (s[j] == ')')
                if depth == 0:
                    break
                j += 1
            s = s[:k] + r'\sqrt{' + s[k + 2:j] + '}' + s[j + 1:]
        else:
            m = re.match(r'√(\w+)', s[k:])
            arg = m.group(1) if m else ''
            s = s[:k] + r'\sqrt{' + arg + '}' + s[k + 1 + len(arg):]
    return s


def _to_latex(s: str) -> str:
    """Unicode maths -> LaTeX."""
    for phrase, rep in _PHRASE.items():
        s = s.replace(phrase, rep)

    s = _sqrt(s)

    # combining circumflex / macron -> \hat{} / \bar{}
    s = re.sub(r'(\w)\u0302', r'\\hat{\1}', s)
    s = re.sub(r'(\w)\u0304', r'\\bar{\1}', s)

    s = s.replace('...', r'\ldots ')
    for ch, name in _GREEK.items():
        s = s.replace(ch, '\\' + name + ' ')
    for ch, rep in _SYMBOL.items():
        s = s.replace(ch, rep)

    # blackboard letters the notes spell out as words
    s = re.sub(r'\bPr\b', r'\\mathbb{P}', s)
    s = re.sub(r'\bE(?=[\[_])', r'\\mathbb{E}', s)
    s = re.sub(r'(?<![\w\\])1(?=\[)', r'\\mathbf{1}', s)

    for name in _UPRIGHT:
        rep = r'\mathrm{%s}' % name
        s = re.sub(r'(?<![\w\\])' + name + r'(?![A-Za-z])', lambda _m, r=rep: r, s)
    for name in _OPERATOR:
        rep = (r'\operatorname*{arg%s}' % name[3:]) if name.startswith('arg') \
            else '\\' + name
        s = re.sub(r'(?<![\w\\])' + name + r'(?![A-Za-z])', lambda _m, r=rep: r, s)

    # parenthesised prose, e.g. "(risk-neutral expectation)"
    s = re.sub(r'\(([A-Za-z][A-Za-z -]{4,})\)',
               lambda m: r'\text{(%s)}' % m.group(1)
               if (' ' in m.group(1) or '-' in m.group(1)
                   or len(m.group(1)) >= 5) else m.group(0), s)
    s = re.sub(r'(?<=[)}\w]) and (?=[(\\\w])', r' \\text{ and } ', s)

    s = s.replace(r'+=', r'\mathrel{+}=')
    # A literal bar breaks Markdown tables and reads as a relation anyway.
    s = s.replace(r'\|', '|').replace('|', r'\mid ')
    s = re.sub(r'\s+([_^])', r'\1', s)
    return re.sub(r'[ ]{2,}', ' ', s).strip()


_FENCE = re.compile(r'^(?P<pre>(?:[ \t]|>)*)```[ \t]*$')


def _convert(text: str) -> str:
    out, lines, i = [], text.split('\n'), 0
    while i < len(lines):
        m = _FENCE.match(lines[i])
        if not m:
            out.append(_inline(lines[i]))
            i += 1
            continue
        pre, body, j = m.group('pre'), [], i + 1
        while j < len(lines) and not _FENCE.match(lines[j]):
            body.append(re.sub(r'^' + re.escape(pre), '', lines[j]))
            j += 1
        eq = r' \\ '.join(_to_latex(b) for b in body if b.strip())
        # dollarmath only sees a display block when it starts its own block, so
        # fence it with blank lines -- carrying the '>' when inside a quote.
        if '>' in pre:
            # Inside a blockquote a multi-line $$ block escapes the quote and
            # drags the '>' markers into the maths; the one-line form doesn't.
            gap = pre.rstrip()
            out += [gap, pre + '$$' + eq + '$$', gap]
        else:
            out += ['', pre + '$$', pre + eq, pre + '$$', '']
        i = j + 1
    return '\n'.join(out)


def _inline(line: str) -> str:
    def repl(m):
        span = m.group(1)
        return m.group(0) if _is_code(span) else '$' + _to_latex(span) + '$'
    return re.sub(r'`([^`\n]+)`', repl, line)


def _on_source_read(app, docname, source):
    if docname.startswith('notes/'):
        source[0] = _convert(source[0])


def setup(app):
    app.connect('source-read', _on_source_read)
    return {'version': '1.0', 'parallel_read_safe': True}
