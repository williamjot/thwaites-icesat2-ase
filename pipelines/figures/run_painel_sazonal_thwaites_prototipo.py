# -*- coding: utf-8 -*-
"""PROTOTIPO - figura sazonal de tres paineis em Thwaites (padrao Nature).

a  gelo aterrado, afinamento rapido perto da linha de aterramento
b  gelo aterrado, interior de afinamento lento
c  gelo flutuante (plataforma leste, TEIS), espessura H_i

Paineis a e b leem as series produzidas por run_grounded_teis_method_series.py
(mesma metodologia, mesmas opcoes de linha de comando). O painel c le a serie
de H_i ja existente (run_teis_hi_seasonal_timeseries.py), sem reprocessar.

O que este prototipo NAO e
--------------------------
* Nao ha celula de ESPESSAMENTO em Thwaites: nos produtos JJA e DJF, todos os
  1 483 nos de 5 km da bacia (proxy) afinam; o maior dh/dt e -0,14 m/ano. O
  painel b mostra o interior de menor afinamento como contraste.
* O painel c NAO e derretimento basal. m_b = SMB - dH/dt - div(uH); aqui so ha
  dH/dt euleriano. O termo de divergencia de fluxo falta e, em Thwaites, e
  grande. Alem disso a serie para em JJA 2022 (fim do IMAU-FDM).

Estilo: 183 mm (coluna dupla), Helvetica/Arial 7 pt, eixos 0,5 pt, sem grade,
Okabe-Ito, PDF vetorial + PNG 600 dpi. Sem titulo: isso vai na legenda.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.stats import theilslopes
from shapely.geometry import Point
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
GH = ROOT / "outputs" / "grounded_H_timeseries"
OUT = ROOT / "outputs" / "painel_sazonal_thwaites"
OUT.mkdir(parents=True, exist_ok=True)

MM = 1.0 / 25.4
AZUL, LARANJA = "#0072B2", "#D55E00"          # Okabe-Ito
CINZA, PRETO, CINZA_TXT = "#8C8C8C", "#1A1A1A", "#6E6E6E"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.direction": "out", "ytick.direction": "out",
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def janelas(ultimo=2025):
    out = []
    for ano in range(2019, ultimo + 1):
        out.append(f"JJA {ano}")
        if ano < ultimo:
            out.append(f"DJF {ano}/{ano + 1}")
    return out


JANELAS = janelas()
POS = {w: i for i, w in enumerate(JANELAS)}


def menos(s: str) -> str:
    return s.replace("-", "−")


def rotulo(w):
    return ("JJA\n%s" % w[4:]) if w.startswith("JJA") else ("DJF\n%s/%s" % (w[6:8], w[-2:]))


# Distancia a linha de aterramento OBSERVADA (InSAR). O shapefile derivado do
# BedMachine inclui a borda do recorte como fronteira do poligono aterrado, e
# para celulas do interior devolve a distancia ate essa borda, nao ate a GL.
_GL = unary_union(
    gpd.read_file(ROOT / "data/grounding/processed/InSAR_GL_ASE_v02.1.gpkg")
    .query("Year >= 2015").geometry.tolist())


def dist_gl_km(x, y):
    return Point(x, y).distance(_GL) / 1000.0


def carrega_aterrado(pasta):
    d = GH / pasta
    ser = pd.read_csv(d / "serie_metodo_TEIS.csv").sort_values("t_center_year")
    rep = json.loads((d / "relatorio_metodo_TEIS.json").read_text(encoding="utf-8"))
    h0 = float(ser.loc[ser.window == "JJA 2019", "h_ice_res"].iloc[0])
    met = rep["tendencia"]
    t = ser.t_center_year.to_numpy(float)
    tend = met["intercept_at_mean_time"] + met["slope_m_per_year"] * (t - met["t_reference"]) - h0
    return dict(
        janela=ser.window.to_list(), t=t,
        y=ser.dh_ice_m.to_numpy(float), s=ser.sigma_dh_ice_m.to_numpy(float),
        tend=tend, taxa=met["slope_m_per_year"], sigma=met["slope_sigma_m_per_year"],
        gtm=rep["Gt_por_metro"], cel=rep["celula"], rep=rep)


def carrega_flutuante(arquivo, chave_rel):
    d = pd.read_csv(ROOT / "data/processed" / arquivo).sort_values("t_center_year")
    rel = json.loads((ROOT / "outputs/seasonal_H_timeseries/TEIS_Hi_timeseries_report.json")
                     .read_text(encoding="utf-8"))["metrics"][chave_rel]
    h0 = float(d.loc[d.window == "JJA 2019", "H_i"].iloc[0])
    t = d.t_center_year.to_numpy(float)
    y = d.H_i.to_numpy(float) - h0
    # Linha: Theil-Sen dos proprios pontos; a taxa anotada e a do relatorio
    # (Monte Carlo condicional a sigma), que e a que carrega a incerteza.
    sl, ic, _, _ = theilslopes(y, t)
    s = d.sigma_H_i.to_numpy(float).copy()
    s[0] = 0.0                                   # referencia: anomalia nula por definicao
    return dict(janela=d.window.to_list(), t=t, y=y, s=s, tend=ic + sl * t,
                taxa=rel["slope_m_per_year"], sigma=rel["slope_sigma_m_per_year"],
                p=rel["slope_p_two_sided"], lon=float(d.lon.iloc[0]),
                lat=float(d.lat.iloc[0]), gl=float(d.distance_to_GL_m.iloc[0]) / 1e3)


def desenha(ax, s, com_massa):
    x = np.array([POS[w] for w in s["janela"]], float)
    jja = np.array([w.startswith("JJA") for w in s["janela"]])
    xs = np.linspace(x.min(), x.max(), 400)
    ax.plot(xs, PchipInterpolator(x, s["y"])(xs), color=CINZA, lw=0.6, zorder=2)
    ax.plot(x, s["tend"], ls=(0, (4, 2)), lw=0.7, color=PRETO, zorder=3)
    for m, mk, cor in ((jja, "o", AZUL), (~jja, "^", LARANJA)):
        ax.errorbar(x[m], s["y"][m], yerr=s["s"][m], fmt=mk, ms=3.4, mfc=cor,
                    mec="white", mew=0.4, color=cor, ecolor=cor, elinewidth=0.6,
                    capsize=1.3, capthick=0.6, ls="none", zorder=4)
    ax.axhline(0, color="#BDBDBD", lw=0.4, zorder=1)
    ax.set_xlim(-0.5, len(JANELAS) - 0.5)
    ax.spines[["top"]].set_visible(False)
    if com_massa:
        g = s["gtm"]
        sec = ax.secondary_yaxis("right", functions=(lambda h: h * g, lambda m: m / g))
        sec.set_ylabel("Δ$M$ (Gt)", labelpad=3)
        sec.spines["right"].set_linewidth(0.5)
        sec.tick_params(width=0.5, size=2.5)
    else:
        ax.spines[["right"]].set_visible(False)
    lo = np.nanmin(s["y"] - s["s"]); hi = np.nanmax(s["y"] + s["s"])
    pad = 0.10 * (hi - lo)
    ax.set_ylim(lo - pad, hi + 2.2 * pad)
    return x


def main():
    a = carrega_aterrado("thwaites_nucleo_metodo_TEIS")
    b = carrega_aterrado("thwaites_interior_metodo_TEIS")
    c = carrega_flutuante("TEIS_Hi_outer_shelf_cell_9696_timeseries.csv", "outer_shelf")

    fig, axs = plt.subplots(3, 1, figsize=(183 * MM, 150 * MM), sharex=True,
                            facecolor="white")
    fig.subplots_adjust(left=0.075, right=0.915, top=0.975, bottom=0.095, hspace=0.22)

    for ax, s, massa in ((axs[0], a, True), (axs[1], b, True)):
        desenha(ax, s, massa)
        ax.set_ylabel("Δ$h$ (m)", labelpad=3)
        c_ = s["cel"]
        ax.text(0.012, 0.965,
                menos("Gelo aterrado · %.2f° W, %.2f° S · %.0f km da linha de aterramento"
                      % (abs(c_["lon"]), abs(c_["lat"]), dist_gl_km(c_["x"], c_["y"]))),
                transform=ax.transAxes, ha="left", va="top", fontsize=6.3, color=CINZA_TXT)
        # Casas decimais pela incerteza: com 3 fixas, o painel b imprimia "± 0.000".
        dm, sdm = s["taxa"] * s["gtm"], s["sigma"] * s["gtm"]
        nd = max(3, int(-np.floor(np.log10(sdm))) + 1) if sdm > 0 else 3
        ax.text(0.012, 0.06,
                menos(("$dh$/$dt$ = %+.2f ± %.2f m ano$^{-1}$    $dM$/$dt$ = %+." + str(nd)
                       + "f ± %." + str(nd) + "f Gt ano$^{-1}$")
                      % (s["taxa"], s["sigma"], dm, sdm)),
                transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5)

    desenha(axs[2], c, False)
    axs[2].set_ylabel("Δ$H_i$ (m)", labelpad=3)
    axs[2].text(0.012, 0.965,
                menos("Gelo flutuante (plataforma leste de Thwaites) · %.2f° W, %.2f° S · "
                      "%.0f km da linha de aterramento" % (abs(c["lon"]), abs(c["lat"]), c["gl"])),
                transform=axs[2].transAxes, ha="left", va="top", fontsize=6.3, color=CINZA_TXT)
    axs[2].text(0.012, 0.06,
                menos("$dH_i$/$dt$ = %+.1f ± %.1f m ano$^{-1}$ (euleriano; não é derretimento basal)"
                      % (c["taxa"], c["sigma"])),
                transform=axs[2].transAxes, ha="left", va="bottom", fontsize=6.5)
    ult = POS[c["janela"][-1]]
    axs[2].axvspan(ult + 0.5, len(JANELAS) - 0.5, color="#F2F2F2", zorder=0, lw=0)
    axs[2].text((ult + len(JANELAS)) / 2, 0.5, "série existente termina em JJA 2022",
                transform=axs[2].get_xaxis_transform(), ha="center", va="center",
                fontsize=6.2, color=CINZA_TXT, style="italic")

    for ax, tag in zip(axs, "abc"):
        ax.text(-0.062, 1.0, tag, transform=ax.transAxes, ha="left", va="top",
                fontsize=8, weight="bold")

    axs[2].set_xticks(np.arange(len(JANELAS)))
    axs[2].set_xticklabels([rotulo(w) for w in JANELAS])
    for tick, w in zip(axs[2].get_xticklabels(), JANELAS):
        tick.set_color(AZUL if w.startswith("JJA") else LARANJA)
    axs[2].set_xlabel("Temporada", labelpad=2)

    # legenda unica, no painel a
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls=(0, (4, 2)), lw=0.7, color=PRETO, label="Tendência (Theil–Sen)"),
         Line2D([], [], marker="o", ls="none", ms=3.4, mfc=AZUL, mec="white", mew=0.4,
                label="Inverno (JJA)"),
         Line2D([], [], marker="^", ls="none", ms=3.4, mfc=LARANJA, mec="white", mew=0.4,
                label="Verão (DJF)")]
    axs[0].legend(handles=h, loc="upper right", ncol=3, handlelength=1.8,
                  columnspacing=1.2, borderaxespad=0.2)

    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(OUT / f"prototipo_tres_paineis.{ext}", facecolor="white", **kw)
    print("->", OUT / "prototipo_tres_paineis.png")
    for nome, s in (("a", a), ("b", b)):
        c_ = s["cel"]
        print(f"{nome}: {abs(c_['lon']):.2f}W {abs(c_['lat']):.2f}S  GL={dist_gl_km(c_['x'], c_['y']):.1f} km  "
              f"dh/dt={s['taxa']:+.3f}±{s['sigma']:.3f}  n={len(s['janela'])}")
    print(f"c: {abs(c['lon']):.2f}W {abs(c['lat']):.2f}S  GL={c['gl']:.1f} km  "
          f"dHi/dt={c['taxa']:+.2f}±{c['sigma']:.2f} (p={c['p']:.4f})  n={len(c['janela'])}")


if __name__ == "__main__":
    main()
