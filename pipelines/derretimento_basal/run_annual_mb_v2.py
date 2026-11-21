"""Derretimento basal ANUAL por plataforma, com janelas independentes (substitui a janela móvel na Fig. 4).

A série trimestral usava DH/Dt numa janela móvel de 5 estações: estações vizinhas compartilhavam
pontos, e isso criava mínimos e máximos aparentes (o "mínimo de 2020–21" da PIG não sobreviveu ao
teste com janelas independentes, ``run_calving_response_test.py``). Aqui cada ano usa só as suas 4
estações (DJF a SON do mesmo ano; DJF recebe o ano de janeiro); anos vizinhos não dividem pontos.

DH/Dt(ano) = inclinação ponderada (1/σ²) da série de espessura por efeitos fixos nas 4 estações.
m_b(ano) = SMB(ano) − DH/Dt(ano) − H·∇·u(ano) (+ Δ fixo da correção na Thwaites).
σ(ano) = σ da inclinação (estatístico). σ estrutural: quase constante no tempo; ver o fatorial.

Saída: ``outputs/series_v2/mb_anual_v2.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "outputs" / "series_v2"


def main() -> int:
    h = pd.read_csv(S / "espessura_trimestral_v2.csv")
    q = pd.read_csv(S / "mb_trimestral_v2.csv")
    rows = []
    for (u, ano), g in h.groupby(["unidade", "ano"]):
        g = g[np.isfinite(g.dH_m) & np.isfinite(g.se_m)]
        if len(g) < 3:
            continue
        t, y, w = g.t_meio.to_numpy(), g.dH_m.to_numpy(), 1 / np.maximum(g.se_m.to_numpy(), 1e-3) ** 2
        tb = np.sum(w * t) / w.sum()
        sxx = np.sum(w * (t - tb) ** 2)
        slope = np.sum(w * (t - tb) * (y - np.sum(w * y) / w.sum())) / sxx
        se = np.sqrt(1 / sxx)
        f = q[(q.unidade == u) & (q.ano == ano)]
        smb, hd, corr = f.SMB_mpa.mean(), f.Hdiv_mpa.mean(), f.correcao_mpa.mean()
        rows.append({"unidade": u, "ano": int(ano), "n_estacoes": len(g), "DHDt_mpa": slope, "se_mpa": se,
                     "SMB_mpa": smb, "Hdiv_mpa": hd, "correcao_mpa": corr,
                     "mb_mpa": smb - slope - hd + corr})
    t = pd.DataFrame(rows)
    t.to_csv(S / "mb_anual_v2.csv", index=False)
    pd.set_option("display.width", 200)
    print(t.pivot_table(index="unidade", columns="ano", values="mb_mpa").round(1).to_string())
    print(t.pivot_table(index="unidade", columns="ano", values="se_mpa").round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
