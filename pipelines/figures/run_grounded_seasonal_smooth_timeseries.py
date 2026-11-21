# -*- coding: utf-8 -*-
"""Series sazonais em celula fixa de gelo aterrado, no estilo da figura TEIS.

Nao reprocessa nada: le os artefatos ja auditados de
``outputs/grounded_H_timeseries`` (CSV por janela + relatorio JSON) e apenas
replota. A curva PCHIP e guia visual entre janelas observadas; nenhum ponto
novo e criado.

Gera duas variantes da mesma celula:

``h``       elevacao de superficie bruta (7 janelas, 2019-2022);
``h_ice``   elevacao corrigida de firn, h - FAC (6 janelas). O cubo
            GSFC-FDM v1.2.1 local (data/firn_thwaites.nc) termina em
            2022.484, antes do centro de JJA 2022 (2022.541), entao essa
            janela nao tem FAC e fica de fora -- limite do modelo de firn,
            nao escolha de recorte.

Diferenca fisica em relacao a figura da TEIS: sobre gelo aterrado a relacao
hidrostatica nao se aplica, entao nenhuma das duas grandezas e espessura de
gelo; ambas sao elevacao.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "grounded_H_timeseries"
SERIES_CSV = OUT_DIR / "grounded_cell_por_janela.csv"
REPORT_JSON = OUT_DIR / "grounded_timeseries_report.json"
PNG_DPI = 360


@dataclass(frozen=True)
class Variant:
    """Uma grandeza plotavel: coluna do CSV + chave de tendencia + rotulos."""

    column: str
    trend_key: str
    output_stem: str
    title: str
    ylabel: str
    slope_symbol: str
    footer: str


VARIANTS = (
    Variant(
        column="h",
        trend_key="tendencia_h",
        output_stem="grounded_h_JJA_DJF_timeseries",
        title="Evolução temporal da elevação de superfície em um ponto fixo "
              "de gelo aterrado",
        ylabel=r"Elevação de superfície $h$ (m)",
        slope_symbol="dh/dt",
        footer="Elevação de superfície sobre gelo aterrado; a relação "
               "hidrostática não se aplica e $h$ não é espessura de gelo."
               "\nInclui variação de firn: parte do sinal sazonal é neve, "
               "não gelo.",
    ),
    Variant(
        column="h_ice",
        trend_key="tendencia_gelo_equivalente",
        output_stem="grounded_h_menos_FAC_JJA_DJF_timeseries",
        title="Evolução temporal da elevação corrigida de firn em um ponto "
              "fixo de gelo aterrado",
        ylabel=r"Elevação gelo-equivalente $h-$FAC (m)",
        slope_symbol="d(h-FAC)/dt",
        footer="Elevação corrigida de firn (FAC do GSFC-FDM v1.2.1, "
               "amostrado co-temporalmente); sobre gelo aterrado a relação "
               "hidrostática não se aplica."
               "\nJJA 2022 ausente: o cubo de firn termina em 2022.484, "
               "antes do centro da janela (2022.541).",
    ),
)

PCHIP_NOTE = ("\nCurva PCHIP apenas como guia visual entre janelas; não "
              "representa novas observações.")


def _load() -> tuple[pd.DataFrame, dict]:
    for path in (SERIES_CSV, REPORT_JSON):
        if not path.exists():
            raise SystemExit(
                f"artefato ausente: {path}\n"
                "rode antes pipelines/figures/run_grounded_seasonal_mass_report.py"
            )
    series = pd.read_csv(SERIES_CSV)
    if series.empty:
        raise SystemExit(f"{SERIES_CSV} sem linhas")
    series = series.sort_values("t_center_year").reset_index(drop=True)
    return series, json.loads(REPORT_JSON.read_text(encoding="utf-8"))


def plot_variant(series: pd.DataFrame, metrics: dict, cell: dict,
                 variant: Variant) -> tuple[list[Path], int]:
    usable = series[np.isfinite(series[variant.column])].reset_index(drop=True)
    if len(usable) < 3:
        raise SystemExit(f"{variant.column}: apenas {len(usable)} janelas com "
                         "valor finito; nada a plotar")
    if len(usable) != metrics["n_windows"]:
        raise SystemExit(
            f"{variant.column}: CSV traz {len(usable)} janelas finitas mas a "
            f"tendencia em {variant.trend_key} foi ajustada sobre "
            f"{metrics['n_windows']}; artefatos fora de sincronia"
        )

    values = usable[variant.column].to_numpy(float)
    dates = pd.to_datetime(usable.t_center, utc=True).dt.tz_convert(None)
    fig, ax = plt.subplots(figsize=(10.4, 6.2), facecolor="white")

    date_numbers = mdates.date2num(dates.to_numpy())
    smooth_dates = np.linspace(date_numbers.min(), date_numbers.max(), 500)
    smooth_values = PchipInterpolator(date_numbers, values)(smooth_dates)
    ax.plot(mdates.num2date(smooth_dates), smooth_values,
            color="#3E4A59", lw=1.45, zorder=2)

    for season, (marker, color) in {"JJA": ("o", "#1F77B4"),
                                    "DJF": ("^", "#D97706")}.items():
        choose = usable.season.eq(season).to_numpy()
        ax.plot(dates[choose], values[choose], marker=marker, ms=6.5,
                mfc="white", mec=color, mew=1.4, linestyle="none",
                label=season, zorder=4)

    t = usable.t_center_year.to_numpy(float)
    trend_y = (metrics["intercept_at_mean_time"]
               + metrics["slope_m_per_year"] * (t - metrics["t_reference"]))
    ax.plot(dates, trend_y, ls="--", lw=1.5, color="#8B1E3F",
            label="Tendência interanual", zorder=3)

    ax.set_xlim(pd.Timestamp("2019-01-01"), pd.Timestamp("2022-12-31"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    low, high = float(values.min()), float(values.max())
    pad = max(0.05 * (high - low), 5.0)
    ax.set_ylim(low - pad, high + pad)
    ax.set_xlabel("Tempo")
    ax.set_ylabel(variant.ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.13))

    slope = metrics["slope_m_per_year"]
    slope_sigma = metrics["slope_sigma_m_per_year"]
    ci0, ci1 = metrics["slope_ci95_low"], metrics["slope_ci95_high"]
    ax.text(
        0.02, 0.97,
        (rf"${variant.slope_symbol}={slope:+.2f}\pm{slope_sigma:.2f}$ "
         rf"m ano$^{{-1}}$"
         f"\nIC 95% [{ci0:+.2f}, {ci1:+.2f}]  ·  {len(usable)} janelas"),
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(facecolor="white", edgecolor="0.75",
                  boxstyle="round,pad=0.3"),
    )
    if metrics["thinning_supported"]:
        ax.text(0.98, 0.04, "tendência de afinamento", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, color="#8B1E3F")
    elif metrics["thickening_supported"]:
        ax.text(0.98, 0.04, "tendência de espessamento",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                color="#245C3A")

    fig.suptitle(
        f"{variant.title}\n"
        f"JJA e DJF — ICESat-2, 2019–2022  ·  {cell['dist_gl_m'] / 1e3:.1f} km "
        "da linha de aterramento",
        fontsize=13, y=0.98,
    )
    fig.text(0.5, 0.005, variant.footer + PCHIP_NOTE, ha="center",
             va="bottom", fontsize=8.5, color="0.30")
    fig.subplots_adjust(bottom=0.23, top=0.84, left=0.11, right=0.98)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / variant.output_stem
    paths = [base.with_suffix(".png"), base.with_suffix(".pdf")]
    fig.savefig(paths[0], dpi=PNG_DPI, facecolor="white")
    fig.savefig(paths[1], facecolor="white")
    plt.close(fig)
    return paths, len(usable)


def main() -> int:
    series, report = _load()
    cell = report["celulas"]["primary"]
    print(f"celula ({cell['cell_x']:.0f}, {cell['cell_y']:.0f}) EPSG:3031 - "
          f"{len(series)} janelas no CSV, "
          f"{series.N_obs.sum()} observacoes ATL06")
    for variant in VARIANTS:
        metrics = report[variant.trend_key]["primary"]
        paths, n = plot_variant(series, metrics, cell, variant)
        print(f"\n{variant.column}: {n} janelas, "
              f"{metrics['slope_m_per_year']:+.4f} "
              f"+/- {metrics['slope_sigma_m_per_year']:.4f} m/ano")
        for path in paths:
            print(f"  gravado: {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
