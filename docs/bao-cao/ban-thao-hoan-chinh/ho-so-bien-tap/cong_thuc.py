"""Convert the report's LaTeX formulas to UnicodeMath, the linear form Word builds equations from.

Word turns UnicodeMath into a real equation (stacked fraction bar, radicals, n-ary sums) when
OMath.BuildUp() runs, so formulas stay editable instead of being images or plain "a/b" text.
Grouping uses parentheses: in UnicodeMath "x_(i,j)" sets i,j as a subscript without printing
the parentheses, and "{a}/{b}" is the same for a fraction.
"""

from __future__ import annotations

import re

MATH_START = "\ue000"
MATH_END = "\ue001"

SYMBOLS = {
    r"\times": "×", r"\cdot": "⋅", r"\approx": "≈", r"\neq": "≠", r"\ne": "≠",
    r"\geq": "≥", r"\ge": "≥", r"\leq": "≤", r"\le": "≤", r"\pm": "±", r"\mp": "∓",
    r"\rightarrow": "→", r"\to": "→", r"\leftarrow": "←", r"\in": "∈", r"\notin": "∉",
    r"\odot": "⊙", r"\otimes": "⊗", r"\top": "⊤", r"\infty": "∞", r"\ldots": "…", r"\dots": "…",
    r"\sigma": "σ", r"\tau": "τ", r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\epsilon": "ε", r"\lambda": "λ", r"\mu": "μ", r"\theta": "θ", r"\eta": "η", r"\rho": "ρ",
    r"\phi": "φ", r"\varphi": "φ", r"\psi": "ψ", r"\omega": "ω", r"\Omega": "Ω", r"\pi": "π",
    r"\nabla": "∇", r"\partial": "∂", r"\forall": "∀", r"\exists": "∃", r"\emptyset": "∅",
    r"\cup": "∪", r"\cap": "∩", r"\subset": "⊂", r"\subseteq": "⊆", r"\equiv": "≡",
    r"\{": "{", r"\}": "}",
    r"\lfloor": "⌊", r"\rfloor": "⌋", r"\lceil": "⌈", r"\rceil": "⌉",
    # Word collapses ordinary spaces inside an equation, so wide LaTeX spacing uses em spaces.
    r"\qquad": "  ", r"\quad": " ", r"\,": " ", r"\;": " ", r"\:": " ",
    r"\!": "", r"\ ": " ",
    r"\left": "", r"\right": "", r"\%": "%", r"\$": "$",
}
# Multi-letter names stay upright inside quotes; Word keeps quoted text as literal text.
FUNCTIONS = ("softmax", "Attention", "FFN", "ReLU", "LN", "MHA", "Dropout", "sigmoid",
             "Accuracy", "Precision", "Recall", "Latency", "tf", "idf", "df")


def take_group(text: str, start: int) -> tuple[str, int]:
    """Read a balanced {...} group beginning at `start`; return its body and the next index."""
    while start < len(text) and text[start] == " ":
        start += 1  # "\widehat y" passes its argument after a space
    if start >= len(text) or text[start] != "{":
        return text[start:start + 1], start + 1
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:index], index + 1
    raise ValueError(f"Ngoac khong can: {text}")


def convert(latex: str) -> str:
    text = latex.strip()

    # Environments first: they hold rows separated by \\ and columns by &.
    def rows(body: str) -> str:
        parts = [convert(part) for part in re.split(r"\\\\", body) if part.strip()]
        return "@".join(part.rstrip(",") for part in parts)

    while r"\begin{cases}" in text:
        head = text.index(r"\begin{cases}")
        tail = text.index(r"\end{cases}", head)
        body = text[head + len(r"\begin{cases}"):tail]
        text = text[:head] + "{■(" + rows(body) + ")┤" + text[tail + len(r"\end{cases}"):]
    for environment in ("aligned", "align", "array", "matrix"):
        token = r"\begin{%s}" % environment
        while token in text:
            head = text.index(token)
            tail = text.index(r"\end{%s}" % environment, head)
            body = text[head + len(token):tail]
            text = text[:head] + "■(" + rows(body) + ")" + text[tail + len(r"\end{%s}" % environment):]

    # Structures that need their arguments parsed.
    # Parentheses, not braces: Word prints "{...}" literally but removes the parentheses that
    # wrap a whole numerator or denominator when it builds the fraction.
    for command, builder in ((r"\frac", lambda a, b: f"({convert(a)})/({convert(b)})"),
                             (r"\dfrac", lambda a, b: f"({convert(a)})/({convert(b)})")):
        while command in text:
            head = text.index(command)
            first, after = take_group(text, head + len(command))
            second, end = take_group(text, after)
            built = builder(first, second)
            # "\log\frac{a}{b}" means log of the whole fraction; written bare, Word applies log to
            # the numerator only. Wrap the fraction in visible parentheses: log((a)/(b)).
            # (The ▒〖〗 function-apply form was tried: it swallowed the tab and "(" of the
            # equation number and set "log" in italics.)
            if re.search(r"\\(log|ln|exp)\s*$", text[:head]):
                built = f"({built})"
            text = text[:head] + built + text[end:]
    # Blackboard-bold and script letters: "\mathbb{R}" and "\mathbb R" both reach here.
    alphabets = {r"\mathbb": {"R": "ℝ", "N": "ℕ", "Z": "ℤ", "Q": "ℚ", "C": "ℂ", "E": "𝔼", "P": "ℙ"},
                 r"\mathcal": {"L": "ℒ", "D": "𝒟", "F": "ℱ", "H": "ℋ", "N": "𝒩", "X": "𝒳"}}
    for command, table in alphabets.items():
        while command in text:
            head = text.index(command)
            body, end = take_group(text, head + len(command))
            letter = body.strip()
            text = text[:head] + table.get(letter, letter) + text[end:]
    for command, template in ((r"\sqrt", "√({0})"), (r"\widehat", "{0}\u0302"),
                              (r"\hat", "{0}\u0302"), (r"\bar", "{0}\u0304"),
                              (r"\operatorname", '"{0}"'), (r"\mathrm", '"{0}"'),
                              (r"\text", '"{0}"'), (r"\mathbf", "{0}")):
        while command in text:
            head = text.index(command)
            body, end = take_group(text, head + len(command))
            inner = convert(body) if command in (r"\sqrt", r"\widehat", r"\hat", r"\bar") else body
            text = text[:head] + template.format(inner) + text[end:]

    for command, symbol in sorted(SYMBOLS.items(), key=lambda item: -len(item[0])):
        text = text.replace(command, symbol)
    # A trailing \b fails before "_", so use an explicit "not a letter" lookahead.
    text = re.sub(r"\\(sum|prod|int)(?![a-zA-Z])",
                  lambda m: {"sum": "∑", "prod": "∏", "int": "∫"}[m.group(1)], text)
    text = re.sub(r"\\(log|ln|exp|max|min|sin|cos|tanh|tan)(?![a-zA-Z])", r"\1", text)

    # Grouping: UnicodeMath hides parentheses used as script arguments.
    text = re.sub(r"_\{([^{}]*)\}", r"_(\1)", text)
    text = re.sub(r"\^\{([^{}]*)\}", r"^(\1)", text)
    for name in FUNCTIONS:
        # An underscore may follow (Latency_(ms/tin)), so only letters, digits and quotes block it.
        text = re.sub(rf'(?<![a-zA-Z0-9"]){name}(?![a-zA-Z0-9"])', f'"{name}"', text)
    text = text.replace("{,}", ",")
    return re.sub(r"[ ]{3,}", "  ", text).strip()


TALL = ("∑", "∏", "∫", "√", "■")


def wrap(latex: str, inline: bool = False) -> str:
    """Wrap a converted formula in the sentinels the Word step looks for.

    Inside a sentence a stacked fraction would stretch the line, so a simple inline formula uses
    the division slash U+2215, which Word keeps linear. When the formula already contains a tall
    part (a sum, a root, a case list) the line is tall anyway, so the stacked bar reads better.
    """
    text = convert(latex)
    if inline and not any(symbol in text for symbol in TALL):
        text = text.replace("/", "∕")
    return f"{MATH_START}{text}{MATH_END}"


def unconverted(text: str) -> bool:
    """True when a backslash command survived the conversion, which would print as raw LaTeX."""
    return bool(re.search(r"\\[a-zA-Z]", text))


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    samples = [
        r"idf(t)=\log\frac{1+N}{1+df(t)}+1,\qquad u_{t,d}=tf(t,d)\,idf(t).",
        r"p=\sigma(z)=\frac{1}{1+e^{-z}},\qquad \widehat y=\begin{cases}1,&p\geq\tau,\\0,&p<\tau,\end{cases}\quad\tau=0{,}5.",
        r"\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.",
        r"v=\frac{\sum_{t=0}^{L-1}m_t Z_t}{\max\!\left(1,\sum_{t=0}^{L-1}m_t\right)}",
        r"Latency_{ms/tin}=\frac{1000\,(t_{end}-t_{start})}{N_{SMS}}.",
        r"Accuracy=\frac{TP+TN}{N},\quad F1=\frac{2TP}{2TP+FP+FN}.",
    ]
    for sample in samples:
        print(f"{sample}\n  -> {convert(sample)}\n")
