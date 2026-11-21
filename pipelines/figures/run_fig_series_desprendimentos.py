"""Figura 4: derretimento basal ANUAL por plataforma (janelas independentes), com desprendimentos.

Cada painel = uma unidade. Pontos: m_b anual (``run_annual_mb_v2.py``; cada ano usa só as suas
4 estações; barras = ±σ estatístico). Faixa horizontal: média ponderada 2019–2025 ± σ.
Texto: teste χ² de que os anos diferem entre si mais do que o ruído (p < 0,05 = variabilidade
interanual detectada). Tracejado: Meng et al. (2025), 2019–2023.
Eixo direito (cinza): posição da face da frente (IceLines, corredor central). Faixas amarelas:
recuos ≥ 2 km entre épocas consecutivas (desprendimentos). Quadrados vermelhos (escala própria):
fração de água/mélange no setor da frente (ICESat-2; cobre 2025).

A versão anterior usava uma janela móvel de 5 estações, que compartilhava pontos entre estações
vizinhas e criava mínimos aparentes (o de 2020–21 na PIG não sobreviveu ao teste com janelas
independentes).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "outputs" / "series_v2"
OUT = ROOT / "outputs" / "figuras_publicacao"
MM = 1 / 25.4
UNITS = ["Pine_Island", "Thwaites_TEIS", "Thwaites_TWGT", "Crosson", "Dotson", "Cosgrove"]
LABEL = {"Pine_Island": "Pine Island", "Thwaites_TEIS": "Thwaites (TEIS)",
         "Thwaites_TWGT": "Thwaites (TWGT, indicativo)", "Crosson": "Crosson", "Dotson": "Dotson",
         "Cosgrove": "Cosgrove"}
COL = {"Pine_Island": "#0F4D92", "Thwaites_TEIS": "#B64342", "Thwaites_TWGT": "#B64342",
       "Crosson": "#42949E", "Dotson": "#9A4D8E", "Cosgrove": "#767676"}
MENG = {"Pine_Island": 16.28, "Thwaites_TEIS": 7.13, "Crosson": 7.79, "Dotson": 7.15, "Cosgrove": 1.22}
EVENT_KM = 2.0
NOISY_FRONT = {"Thwaites_TWGT", "Crosson"}

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 6.5,
    "axes.spines.top": False, "axes.linewidth": 0.6, "legend.frameon": False,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.labelsize": 6.5,
})


def main() -> int:
    a = pd.read_csv(S / "mb_anual_v2.csv")
    pos = pd.read_csv(S / "frente_posicao.csv")
    ev = pd.read_csv(S / "desprendimentos.csv")
    wat = pd.read_csv(S / "agua_setor_frente.csv")
    fig, axes = plt.subplots(len(UNITS), 1, figsize=(183 * MM, 200 * MM), sharex=True,
                             gridspec_kw={"hspace": 0.2})
    rows, stats = [], []
    for ax, u in zip(axes, UNITS):
        g = a[a.unidade == u].sort_values("ano")
        x = g.ano.to_numpy() + 0.5
        y, e = g.mb_mpa.to_numpy(), g.se_mpa.to_numpy()
        w = 1 / e ** 2
        mean = np.sum(w * y) / w.sum()
        smean = np.sqrt(1 / w.sum())
        c2 = float(np.sum(((y - mean) / e) ** 2))
        p = float(chi2.sf(c2, len(y) - 1))
        stats.append({"unidade": u, "media_ponderada": mean, "sigma_media": smean, "chi2": c2,
                      "gl": len(y) - 1, "p": p})
        ax.axhspan(mean - smean, mean + smean, color=COL[u], alpha=0.15, lw=0)
        ax.axhline(mean, color=COL[u], lw=0.6)
        ax.errorbar(x, y, yerr=e, fmt="o", color=COL[u], ms=3, elinewidth=0.8, capsize=1.5, capthick=0.6)
        if u in MENG:
            ax.plot([2019, 2024], [MENG[u]] * 2, "--", color="k", lw=0.6)
        ax.text(0.995, 0.93, f"χ² = {c2:.1f} ({len(y) - 1} gl), p = {p:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=5.6)
        for _, r in ev[(ev.unidade == u) & (ev.recuo_km >= EVENT_KM) & (ev.e >= 2018.5)].iterrows():
            ax.axvspan(max(r.entre, 2018.95), r.e, color="#F2C14E", alpha=0.35, lw=0, zorder=0)
            rows.append({"unidade": u, "inicio": r.entre, "fim": r.e, "recuo_km": r.recuo_km})
        b = ax.twinx()
        pp = pos[(pos.unidade == u) & (pos.epoca >= 2018.9)]
        b.plot(pp.epoca, pp.posicao_frente_km, "-", color="#767676", lw=0.8)
        b.set_ylabel("frente (km)", color="#4D4D4D", fontsize=5.8)
        b.tick_params(axis="y", colors="#4D4D4D", labelsize=5.5)
        b.spines["right"].set_color("#767676")
        ww = wat[wat.unidade == u]
        c = ax.twinx()
        c.plot(ww.t_meio, 100 * ww.frac_agua, "s", ms=1.8, color="#E53935", alpha=0.7)
        c.set_ylim(0, max(30, 100 * ww.frac_agua.max() * 1.15) if len(ww) else 30)
        c.axis("off")
        ax.set_ylabel("m_b (m/ano)")
        ax.text(0.005, 0.93, LABEL[u] + (" — frente de mélange (ruidosa)" if u in NOISY_FRONT else ""),
                transform=ax.transAxes, fontsize=6.5, fontweight="bold", va="top")
        ax.set_xlim(2018.9, 2026.1)
    axes[-1].set_xlabel("Ano")
    h = [mpl.lines.Line2D([], [], color="#4D4D4D", marker="o", ms=3, ls=""),
         mpl.patches.Patch(color="#767676", alpha=0.25),
         mpl.lines.Line2D([], [], color="k", ls="--", lw=0.6),
         mpl.lines.Line2D([], [], color="#767676", lw=0.8),
         mpl.patches.Patch(color="#F2C14E", alpha=0.35),
         mpl.lines.Line2D([], [], color="#E53935", marker="s", ls="", ms=2)]
    axes[0].legend(h, ["m_b anual (janelas independentes; ±σ estatístico)",
                       "Média ponderada 2019–2025 ± σ", "Meng et al. 2025 (2019–2023)",
                       "Posição da frente (IceLines)", f"Desprendimento (recuo ≥ {EVENT_KM:g} km)",
                       "Água/mélange no setor da frente (ICESat-2)"],
                   loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=5.6)
    for ext in ("svg", "pdf"):
        fig.savefig(OUT / f"fig4_series_desprendimentos.{ext}", bbox_inches="tight")
    fig.savefig(OUT / "fig4_series_desprendimentos.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "fig4_series_desprendimentos_previa.png", dpi=200, bbox_inches="tight")
    pd.DataFrame(rows).to_csv(OUT / "dados" / "fig4_desprendimentos.csv", index=False)
    a.to_csv(OUT / "dados" / "fig4_mb_anual.csv", index=False)
    st = pd.DataFrame(stats)
    st.to_csv(OUT / "dados" / "fig4_teste_variabilidade.csv", index=False)
    print(st.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
