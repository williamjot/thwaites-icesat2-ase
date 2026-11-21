# -*- coding: utf-8 -*-
"""Figuras do relatorio de gelo aterrado. Recebe constantes por argumento."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def short(w):
    return ("JJA " + w[-2:]) if w.startswith("JJA") \
        else ("DJF " + w[6:8] + "/" + w[-2:])


def _bars(ax, values, sigmas, labels, colors, ylabel, title, fmt):
    xp = np.arange(len(values))
    ax.bar(xp, values, color=colors, edgecolor="0.25", lw=0.6, width=0.62,
           zorder=3)
    if sigmas is not None:
        ax.errorbar(xp, values, yerr=sigmas, fmt="none", ecolor="0.25",
                    elinewidth=1.1, capsize=4, zorder=4)
    ax.axhline(0, color="0.3", lw=0.9, zorder=2)
    ax.set_xticks(xp)
    ax.set_xticklabels(labels, fontsize=8.0)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_title(title, fontsize=10.5, loc="left")
    ax.grid(axis="y", color="0.92", lw=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    sig = np.zeros_like(values) if sigmas is None else np.asarray(sigmas)
    lim = float(np.nanmax(np.abs(values) + sig) * 1.45)
    ax.set_ylim(-lim, lim)
    for x, v, s in zip(xp, values, sig):
        if not np.isfinite(v):
            ax.annotate("sem FAC", (x, 0.0), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=8.0,
                        color="0.45", style="italic")
            continue
        off = 10 if v >= 0 else -18
        ax.annotate(fmt.format(v), (x, v + np.sign(v) * s),
                    textcoords="offset points", xytext=(0, off), ha="center",
                    fontsize=8.4, fontweight="bold")


def figure_main(ser, tr, met, met_ice, info, gia, out_base,
                GT_PER_M, GT_PER_MM_SLE):
    dates = pd.to_datetime(ser.t_center, utc=True).dt.tz_convert(None)
    lab = [short(a) + "\n↓\n" + short(b) for a, b in zip(tr.de, tr.para)]
    fig = plt.figure(figsize=(13.4, 11.6), facecolor="white")
    gs = fig.add_gridspec(3, 2, height_ratios=[1.20, 1.0, 1.0], hspace=0.62,
                          wspace=0.28, left=0.078, right=0.918, top=0.882,
                          bottom=0.135)

    # ---------------------------------------------------------------- (a)
    ax = fig.add_subplot(gs[0, :])
    ax.plot(dates, ser.h, color="#3E4A59", lw=1.2, zorder=2)
    for season, (mk, c) in {"JJA": ("o", "#1F77B4"),
                            "DJF": ("^", "#D97706")}.items():
        m = ser.season.eq(season)
        ax.errorbar(dates[m], ser.h[m], yerr=ser.sigma_h[m], fmt=mk, ms=7,
                    mfc="white", mec=c, mew=1.5, ecolor=c, elinewidth=1.0,
                    capsize=3, ls="none", label="h " + season, zorder=4)
    tt = ser.t_center_year.to_numpy(float)
    ax.plot(dates, met["intercept_at_mean_time"]
            + met["slope_m_per_year"] * (tt - met["t_reference"]),
            ls="--", lw=1.5, color="#8B1E3F", label="Theil–Sen (h)",
            zorder=3)
    shift = float(np.nanmean(ser.h - ser.h_ice))
    ax.plot(dates, ser.h_ice + shift, color="#245C3A", lw=1.3, marker="s",
            ms=4.5, mfc="white", mew=1.2,
            label="$h-$FAC (deslocado %+.1f m)" % shift, zorder=5)
    for d, hv, sv, mv in zip(dates, ser.h, ser.sigma_h, ser.dM_cum_Gt):
        tag = ("%.2f m\n%+.4f Gt" % (hv, mv) if np.isfinite(mv)
               else "%.2f m\nsem FAC" % hv)
        ax.annotate(tag, (d, hv + sv), textcoords="offset points",
                    xytext=(0, 6), ha="center", va="bottom", fontsize=7.9,
                    color="#333")
    ax.set_ylabel("Elevação de superfície $h$ (m)")
    h0 = float(ser.h.iloc[0])
    sec = ax.secondary_yaxis(
        "right", functions=(lambda v: (v - h0) * GT_PER_M,
                            lambda m: m / GT_PER_M + h0))
    sec.set_ylabel("Massa da célula (Gt) para $\\Delta h$ bruto")
    lo = float((ser.h - ser.sigma_h).min())
    hi = float((ser.h + ser.sigma_h).max())
    rng = max(hi - lo, 1.0)
    ax.set_ylim(lo - 0.60 * rng, hi + 0.80 * rng)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color="0.91", lw=0.7)
    ax.spines[["top"]].set_visible(False)
    ax.legend(frameon=False, ncol=4, loc="lower left", fontsize=8.5)
    ax.set_title("(a) Elevação por janela, série corrigida de firn e massa "
                 "acumulada da célula (25 km²)", fontsize=11, loc="left",
                 pad=22)

    cols = ["#2A6F97" if v < 0 else "#B36B00" for v in tr.dh_m]

    # ---------------------------------------------------------------- (b)
    ax = fig.add_subplot(gs[1, 0])
    _bars(ax, tr.dhdt_m_ano.values, tr.sigma_dhdt_m_ano.values, lab, cols,
          r"$dh/dt$ (m ano$^{-1}$)",
          "(b) Taxa em cada intervalo entre temporadas", "{:+.2f}")
    sl, ss = met["slope_m_per_year"], met["slope_sigma_m_per_year"]
    ax.axhspan(sl - ss, sl + ss, color="#8B1E3F", alpha=0.10, zorder=1)
    ax.axhline(sl, color="#8B1E3F", ls="--", lw=1.3, zorder=2)
    ax.plot(np.arange(len(tr)), tr.dh_icedt_m_ano, "s", ms=5.5,
            color="#245C3A", zorder=6, label="gelo-equivalente ($-$FAC)")
    ax.legend(frameon=False, fontsize=7.8, loc="upper right")
    ax.text(0.015, 0.968, "faixa: tendência = %+.2f $\\pm$ %.2f m/ano"
            % (sl, ss), transform=ax.transAxes, fontsize=8.2,
            color="#8B1E3F", ha="left", va="top")

    # ---------------------------------------------------------------- (c)
    ax = fig.add_subplot(gs[1, 1])
    _bars(ax, tr.dM_Gt.values, tr.sigma_dM_Gt.values, lab, cols,
          "ΔM da célula (Gt)",
          "(c) Massa por intervalo — gelo-equivalente", "{:+.4f}")

    # ---------------------------------------------------------------- (d)
    ax = fig.add_subplot(gs[2, 0])
    ax.axhline(0, color="0.3", lw=0.9)
    ax.fill_between(dates, ser.dM_cum_Gt - ser.sigma_dM_cum_Gt,
                    ser.dM_cum_Gt + ser.sigma_dM_cum_Gt, color="#2A6F97",
                    alpha=0.16, label=r"$\pm1\sigma$")
    ax.plot(dates, ser.dM_cum_Gt, color="#2A6F97", lw=1.6, marker="o", ms=5.5,
            mfc="white", mew=1.4, label="observado (gelo-equivalente)")
    base_ice = float(ser.h_ice.iloc[0])
    ax.plot(dates, (met_ice["intercept_at_mean_time"]
                    + met_ice["slope_m_per_year"]
                    * (tt - met_ice["t_reference"]) - base_ice) * GT_PER_M,
            ls="--", lw=1.4, color="#8B1E3F",
            label="Theil–Sen: %+.4f Gt/ano"
                  % (met_ice["slope_m_per_year"] * GT_PER_M))
    ax.set_ylabel("ΔM acumulada (Gt)", fontsize=9.5)
    sec2 = ax.secondary_yaxis("right",
                              functions=(lambda m: -m / GT_PER_MM_SLE,
                                         lambda s: -s * GT_PER_MM_SLE))
    sec2.set_ylabel("Nível do mar (mm)", fontsize=9)
    ax.set_title("(d) Massa acumulada e equivalente de nível do mar",
                 fontsize=10.5, loc="left")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color="0.92", lw=0.7)
    ax.spines[["top"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.0, loc="lower left")

    # ---------------------------------------------------------------- (e)
    ax = fig.add_subplot(gs[2, 1])
    xp = np.arange(len(tr))
    ax.bar(xp, tr.z, color=["#B23A48" if z <= 1 else "#245C3A" for z in tr.z],
           edgecolor="0.25", lw=0.6, width=0.62, zorder=3)
    ax.axhline(1.0, color="0.25", ls="--", lw=1.1, zorder=4,
               label=r"limiar $|\Delta h| = 1\sigma$")
    ax.set_xticks(xp)
    ax.set_xticklabels(lab, fontsize=8.0)
    ax.set_ylabel(r"$|\Delta h| \, / \, \sigma(\Delta h)$", fontsize=9.5)
    n_ok = int((tr.z > 1).sum())
    ax.set_title("(e) %d de %d intervalos excedem 1$\\sigma$"
                 % (n_ok, len(tr)), fontsize=10.5, loc="left")
    ax.grid(axis="y", color="0.92", lw=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(1.35, float(tr.z.max()) * 1.3))
    ax.legend(frameon=False, fontsize=8.4, loc="upper left")
    for x, z in zip(xp, tr.z):
        ax.annotate("%.2f" % z, (x, z), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8.4,
                    fontweight="bold")

    fig.suptitle("Gelo aterrado — célula fixa em (%.0f, %.0f) m EPSG:3031, "
                 "%.1f km da linha de aterramento\n$dh/dt$ e variação de "
                 "massa por temporada — ICESat-2 ATL06, JJA/DJF 2019–2022"
                 % (ser.cell_x.iloc[0], ser.cell_y.iloc[0],
                    info["dist_gl_m"] / 1000.0), fontsize=13, y=0.972)
    fig.text(0.5, 0.014,
             "Sem relação hidrostática: em gelo aterrado a grandeza é a "
             "elevação de superfície, sem a amplificação "
             "$\\rho_w/(\\rho_w-\\rho_i)$ = 9,34 da plataforma. "
             "Massa = $\\rho_i\\cdot A\\cdot(\\Delta h-\\Delta$FAC), "
             "$\\rho_i$ = 917 kg m$^{-3}$, A = 25 km², FAC do GSFC-FDM v1.2.1.\n"
             "Sinal euleriano em posição fixa. GIA (Caron 2018) aqui = "
             "%+.1f mm/ano de soerguimento; entra na tendência corrigida do "
             "relatório, não nas barras." % (gia["vlm_mm_per_year"]),
             ha="center", fontsize=8.5, color="0.30")
    for ext in ("png", "pdf"):
        fig.savefig(str(out_base) + "." + ext,
                    dpi=300 if ext == "png" else None, facecolor="white")
    plt.close(fig)


def figure_transect(rows, out_base, GT_PER_M):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), facecolor="white")
    names = [r["label"] for r in rows]
    sl = np.array([r["slope_m_per_year"] for r in rows])
    ss = np.array([r["slope_sigma_m_per_year"] for r in rows])
    pv = [r["slope_p_two_sided"] for r in rows]
    cols = ["#245C3A" if p < 0.05 else "#9AA5B1" for p in pv]
    xp = np.arange(len(rows))
    for ax, vals, sig, ylab, ttl, fmt in [
        (axes[0], sl, ss, r"$dh/dt$ (m ano$^{-1}$)",
         "(a) Tendência 2019–2022 por célula", "{:+.2f}"),
        (axes[1], sl * GT_PER_M, ss * GT_PER_M,
         "dM/dt da célula (Gt ano$^{-1}$)",
         "(b) Mesma tendência expressa em massa", "{:+.4f}")]:
        ax.bar(xp, vals, color=cols, edgecolor="0.25", lw=0.6, width=0.6,
               zorder=3)
        ax.errorbar(xp, vals, yerr=sig, fmt="none", ecolor="0.25",
                    elinewidth=1.1, capsize=4, zorder=4)
        ax.axhline(0, color="0.3", lw=0.9)
        ax.set_xticks(xp)
        ax.set_xticklabels(names, fontsize=8.2)
        ax.set_ylabel(ylab)
        ax.set_title(ttl, fontsize=11, loc="left")
        ax.grid(axis="y", color="0.92", lw=0.7, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        lo = min(0.0, float((vals - sig).min()))
        hi = max(0.0, float((vals + sig).max()))
        rng = max(hi - lo, 1e-6)
        ax.set_ylim(lo - 0.55 * rng, hi + 0.10 * rng)
        for x, v, s, p in zip(xp, vals, sig, pv):
            ax.annotate(fmt.format(v) + "\np = %.4f" % p, (x, v - s),
                        textcoords="offset points", xytext=(0, -28),
                        ha="center", fontsize=8.4)
    if any(p < 0.05 for p in pv):
        axes[0].plot([], [], "s", color="#245C3A", label="p < 0,05")
    if any(p >= 0.05 for p in pv):
        axes[0].plot([], [], "s", color="#9AA5B1", label="não significativo")
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.suptitle("Gelo aterrado — tendência de elevação e de massa nas quatro "
                 "células fixas (distância à linha de aterramento)",
                 fontsize=12.5, y=0.98)
    fig.text(0.5, 0.045, "Barras = $\\pm1\\sigma$ (Theil–Sen, Monte Carlo "
             "condicional a $\\sigma_h$). Células escolhidas por cobertura "
             "orbital, nunca por magnitude de $dh/dt$.",
             ha="center", fontsize=8.6, color="0.30")
    fig.subplots_adjust(bottom=0.21, top=0.84, wspace=0.24, left=0.075,
                        right=0.97)
    for ext in ("png", "pdf"):
        fig.savefig(str(out_base) + "." + ext,
                    dpi=300 if ext == "png" else None, facecolor="white")
    plt.close(fig)
