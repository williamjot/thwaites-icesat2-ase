# -*- coding: utf-8 -*-
"""Grafico unico: anomalia de massa acumulada, inverno (JJA) x verao (DJF)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(r"D:/WILLIAM/ARTIGO PIBIC ANTARTICA")
OUT = ROOT / "outputs" / "grounded_H_timeseries" / "nucleo_afinamento_thwaites"

ser = pd.read_csv(OUT / "serie_inverno_verao.csv")
rep = json.loads((OUT / "relatorio.json").read_text(encoding="utf-8"))
GTM = rep["Gt_por_metro"]
ser["date"] = pd.to_datetime(ser.t_center, utc=True).dt.tz_convert(None)
H_REF = float(ser.loc[ser.window == "JJA 2019", "h"].iloc[0])

STYLE = {"JJA": dict(color="#1F5F99", marker="o", label="Inverno (JJA)"),
         "DJF": dict(color="#C0620B", marker="^", label="Verão (DJF)")}


def short(w):
    return ("JJA " + w[-4:]) if w.startswith("JJA") else ("DJF " + w[6:8] + "/" + w[-2:])


fig, ax = plt.subplots(figsize=(12.5, 7.6), facecolor="white")
fig.subplots_adjust(left=0.085, right=0.905, top=0.845, bottom=0.175)
ax.axhline(0, color="0.55", lw=0.8, zorder=1)

for season in ("JJA", "DJF"):
    st = STYLE[season]
    s = ser[ser.season == season].sort_values("t_center_year")
    full = s[~s.partial]
    part = s[s.partial]
    ax.plot(s.date, s.dM_Gt, color=st["color"], lw=1.3, alpha=0.9, zorder=2)
    ax.errorbar(full.date, full.dM_Gt, yerr=full.sigma_dM_Gt, fmt=st["marker"],
                ms=7.5, color=st["color"], mfc=st["color"], mec="white", mew=0.9,
                ecolor=st["color"], elinewidth=1.1, capsize=3, zorder=4)
    if len(part):
        ax.errorbar(part.date, part.dM_Gt, yerr=part.sigma_dM_Gt, fmt=st["marker"],
                    ms=7.5, color=st["color"], mfc="white", mec=st["color"], mew=1.5,
                    ecolor=st["color"], elinewidth=1.1, capsize=3, zorder=4)
    tr = rep["tendencias"][season]
    tt = full.t_center_year.to_numpy(float)
    line = (tr["intercept_at_mean_time"] + tr["slope_m_per_year"]
            * (tt - tr["t_reference"]) - H_REF) * GTM
    ax.plot(full.date, line, ls="--", lw=1.4, color=st["color"], alpha=0.75, zorder=3,
            label="%s, tendência: %+.2f ± %.2f m/ano  =  %+.3f ± %.3f Gt/ano"
                  % (st["label"], tr["slope_m_per_year"], tr["slope_sigma_m_per_year"],
                     tr["slope_m_per_year"] * GTM, tr["slope_sigma_m_per_year"] * GTM))
    ax.plot([], [], st["marker"], color=st["color"], ms=7.5, mec="white",
            label=st["label"] + " — janela (±1σ)")
    for _, r in s.iterrows():
        dy = 13 if season == "JJA" else -20
        ax.annotate(short(r.window), (r.date, r.dM_Gt), textcoords="offset points",
                    xytext=(0, dy), ha="center", fontsize=7.4, color=st["color"])

if ser.partial.any():
    ax.plot([], [], "^", mfc="white", mec="0.35", mew=1.3, ms=7.5,
            label="marcador vazado = janela parcial (DJF 18/19: só jan–fev 2019; "
                  "DJF 25/26: só dez 2025)")

ax.set_ylabel("Anomalia de massa acumulada da célula (Gt)\nreferência: JJA 2019",
              fontsize=10.5)
sec = ax.secondary_yaxis("right", functions=(lambda m: m / GTM, lambda h: h * GTM))
sec.set_ylabel("Δh de superfície (m)", fontsize=10.5)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_xlabel("Tempo")
ax.grid(axis="y", color="0.9", lw=0.7)
ax.spines[["top"]].set_visible(False)
lo = float((ser.dM_Gt - ser.sigma_dM_Gt).min())
hi = float((ser.dM_Gt + ser.sigma_dM_Gt).max())
pad = 0.10 * (hi - lo)
ax.set_ylim(lo - pad, hi + 1.6 * pad)
ax.legend(frameon=False, fontsize=8.6, loc="lower left", ncol=1)

c = rep["celula"]
off = rep["desvio_sazonal"]
fig.suptitle(
    "Gelo aterrado — núcleo de afinamento de Thwaites · anomalia de massa acumulada, "
    "inverno × verão\n"
    "célula fixa de 5 km em (%.0f, %.0f) m EPSG:3031 · %.2f°W, %.2f°S · %.1f km da "
    "linha de aterramento\n"
    "ICESat-2 ATL06, 2019–2025 · dh/dt no mapa regional: %.2f m/ano (JJA), %.2f m/ano (DJF)"
    % (c["cx"], c["cy"], abs(rep["lon"]), abs(rep["lat"]), c["dist_gl_m"] / 1000.0,
       c["dhdt_mapa_jja"], c["dhdt_mapa_djf"]), fontsize=11.2, y=0.985)
fig.text(
    0.5, 0.018,
    "Massa = ρ$_i$·A·Δh, ρ$_i$ = 917 kg m$^{-3}$, A = 25 km² (1 m = %.4f Gt) — premissa "
    "de afinamento dinâmico; FAC NÃO descontado (o GSFC-FDM termina em jun/2022).\n"
    "Elevação por janela ajustada sobre h − REMA: intercepto por janela + plano local, "
    "Huber, σ cluster-robusta por faixa orbital.\n"
    "Desvio DJF − JJA com tendência comum: %+.2f ± %.2f m (não significativo). "
    "Sinal euleriano; GIA (Caron 2018) = %+.1f mm/ano."
    % (GTM, off["DJF_menos_JJA_m"], off["sigma_m"], rep["gia_mm_ano"]),
    ha="center", fontsize=8.4, color="0.3")
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"anomalia_massa_inverno_verao.{ext}",
                dpi=300 if ext == "png" else None, facecolor="white")
print("ok")
