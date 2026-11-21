"""Utilidades de formatação dos relatórios de auditoria em português."""

from __future__ import annotations

import re

_NUMBER_RE = re.compile(
    r"(?<![\w.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w])"   # 9,680 / 1,234.56
    r"|(?<![\w.,])\d+\.\d+(?![\w])"                       # 0.993506
)


def localize_numbers(markdown: str) -> str:
    """Converte os números de um relatório para a notação decimal em português.

    Os textos são escritos com separadores no padrão inglês (``9,680`` e
    ``0.51``) e convertidos de uma vez só no fim. A troca não toca em trechos
    entre crases nem em blocos de código, e ignora títulos — onde ``2.1`` é
    numeração de seção, não número decimal.
    """
    def swap(match: re.Match) -> str:
        return match.group(0).translate(str.maketrans({",": ".", ".": ","}))

    out, in_fence = [], False
    for line in markdown.split("\n"):
        if line.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence or line.startswith("#"):
            out.append(line)
            continue
        parts = line.split("`")
        for i in range(0, len(parts), 2):  # fora das crases
            parts[i] = _NUMBER_RE.sub(swap, parts[i])
        out.append("`".join(parts))
    return "\n".join(out)


__all__ = ["localize_numbers"]
