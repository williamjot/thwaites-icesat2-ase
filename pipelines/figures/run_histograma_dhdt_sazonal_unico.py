"""
Figura única com dois painéis de dh/dt: inverno (JJA) e verão (DJF).

Cada painel repete exatamente o estilo dos histogramas individuais
(`run_histogramas_dhdt_individuais.py`): barras coloridas pela paleta
divergente do dh/dt — vermelho para afinamento, cinza no zero, azul para
espessamento — classes de 0,1 m/ano, média (linha cheia) e mediana (linha
pontilhada).

Os painéis são empilhados e compartilham os eixos x e y, de modo que as
alturas das classes sejam diretamente comparáveis entre as estações. Sem
título, sem contagem de nós e sem nota de rodapé: a identificação fica em
a)/b) dentro do painel e no rótulo da estação na legenda.

Entradas
--------
data/<estacao>/dhdt/dhdt_nodes_qc.parquet

Saídas
------
outputs/comparacao_sazonal/histograma_dhdt_jja_djf_painel_unico.{png,svg}

Uso: python pipelines/figures/run_histograma_dhdt_sazonal_unico.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from pipelines.figures.run_produto_figuras import DIV_THIN


OUT = ROOT / "outputs" / "comparacao_sazonal"
INK = "#1A1A1A"
INK2 = "#4A4A4A"
MUTED = "#777777"
GRID = "#D9D9D6"
BG = "#FCFCFB"
STAT = "#0D47A1"

PANELS = (
    ("jja", "a", "JJA · inverno austral"),
    ("djf", "b", "DJF · verão austral"),
)

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "legend.frameon": True,
        "svg.fonttype": "none",
    }
)


def _load() -> dict[str, np.ndarray]:
    configs = {"jja": load_config("jja"), "djf": load_config("djf")}
    result = {}
    for season, cfg in configs.items():
        values = pd.read_parquet(
            cfg.paths.dhdt_dir / "dhdt_nodes_qc.parquet", columns=["dhdt"]
        )["dhdt"].to_numpy(float)
        result[season] = values[np.isfinite(values)]
    return result


def _panel(ax, values, bins, norm, tag, season_label):
    """Um painel no estilo do histograma individual."""
    density, edges = np.histogram(values, bins=bins, density=True)
    for index, height in enumerate(density):
        left, right = edges[index], edges[index + 1]
        color = DIV_THIN(np.clip(norm(0.5 * (left + right)), 0.0, 1.0))
        ax.bar(
            left,
            height,
            width=right - left,
            align="edge",
            color=color,
            edgecolor="#333333",
            linewidth=0.48,
            zorder=2,
        )

    ax.axvline(0, color=INK, linewidth=1.0, linestyle="--", alpha=0.65,
               zorder=3)
    mean_line = ax.axvline(
        float(np.mean(values)), color=STAT, linewidth=2.2, linestyle="-",
        zorder=4, label=f"média {np.mean(values):+.3f} m/ano")
    median_line = ax.axvline(
        float(np.median(values)), color=STAT, linewidth=1.8, linestyle=":",
        zorder=4, label=f"mediana {np.median(values):+.3f} m/ano")

    ax.set_ylabel("Densidade de probabilidade")
    ax.grid(axis="y", color=GRID, linewidth=0.75, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[mean_line, median_line],
        loc="upper left",
        title=season_label,
        fontsize=10.2,
        title_fontsize=10.2,
        handlelength=2.6,
        borderpad=0.65,
        labelspacing=0.5,
        facecolor=BG,
        edgecolor="#B9B9B5",
        framealpha=0.96,
    )
    # Identificador do painel, canto superior direito, dentro dos eixos.
    ax.text(
        0.988, 0.94, f"{tag})", transform=ax.transAxes, ha="right", va="top",
        fontsize=15, fontweight="bold", color=INK, zorder=6,
        bbox=dict(boxstyle="square,pad=0.30", fc=BG, ec="#B9B9B5", lw=0.7,
                  alpha=0.96),
    )


def main() -> None:
    values = _load()
    lower = np.floor(min(v.min() for v in values.values()) * 10) / 10
    upper = np.ceil(max(v.max() for v in values.values()) * 10) / 10
    bins = np.arange(lower, upper + 0.1001, 0.1)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-3.0, vmax=1.0)

    fig, axes = plt.subplots(
        2, 1, figsize=(12.8, 9.6), dpi=150, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.088, right=0.972, bottom=0.085, top=0.985,
                        hspace=0.11)

    for ax, (season, tag, label) in zip(axes, PANELS):
        _panel(ax, values[season], bins, norm, tag, label)

    axes[0].set_xlim(float(lower), float(upper))
    axes[0].set_ylim(bottom=0)
    axes[-1].set_xlabel("Taxa de mudança de elevação, dh/dt (m ano⁻¹)")

    OUT.mkdir(parents=True, exist_ok=True)
    stem = "histograma_dhdt_jja_djf_painel_unico"
    png, svg = OUT / f"{stem}.png", OUT / f"{stem}.svg"
    fig.savefig(png, dpi=150, facecolor=fig.get_facecolor())
    fig.savefig(svg, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"-> {png}")
    print(f"-> {svg}")


if __name__ == "__main__":
    main()
