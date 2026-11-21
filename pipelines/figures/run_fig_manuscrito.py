"""Figuras do manuscrito ASE/ICESat-2 2019–2025 (manuscrito/manuscrito_ASE_ICESat2_2019-2025.md).

Só lê resultados já calculados (JSON, CSV, Parquet, NetCDF); nada é reprocessado. As únicas contas
são de exibição: mediana regional e correlação célula a célula das duas grades de dh/dt (Fig. 2c).

Fig. 1 área de estudo + esquema do balanço de massa da plataforma
Fig. 2 dh/dt do gelo apoiado, JJA × DJF
Fig. 3 anomalia de massa (célula do núcleo da Thwaites; domínio) e dM/dt com L e GIA
Fig. 4 mapa do derretimento basal
Fig. 5 zona de flexão, radar, correção e ApRES
Fig. 6 comparação por plataforma, decomposição da incerteza, massa
Fig. 7 variabilidade interanual e teste do desprendimento
Fig. 8 evento de 2024 no tronco oeste da Thwaites

Saída: manuscrito/figuras/{figN}.{svg,pdf,tiff,_previa.png} e manuscrito/figuras/dados/*.csv.
Tamanho final 183 mm (coluna dupla); texto 6–7 pt; vírgula decimal.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import SymLogNorm, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from matplotlib.ticker import FuncFormatter
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
I = ROOT / "data" / "interim"
O = ROOT / "outputs"
OUT = ROOT / "manuscrito" / "figuras"
SRC = OUT / "dados"
PUB = O / "figuras_publicacao" / "dados"
EV = ROOT / "evento_2024" / "outputs"
BM_PATH = ROOT / "data" / "NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc"
MM = 1 / 25.4
GT_PER_MM = 361.8
UNITS = ["Pine_Island", "Thwaites_TEIS", "Crosson", "Dotson", "Cosgrove"]
LABEL = {"Pine_Island": "Pine Island", "Thwaites_TEIS": "Thwaites (TEIS)", "Crosson": "Crosson",
         "Dotson": "Dotson", "Cosgrove": "Cosgrove", "Thwaites": "Thwaites",
         "Thwaites_TWGT": "Thwaites (TWGT)"}
COL = {"Pine_Island": "#0F4D92", "Thwaites": "#B64342", "Thwaites_TEIS": "#B64342",
       "Crosson": "#42949E", "Dotson": "#9A4D8E", "Cosgrove": "#767676"}
JJA_C, DJF_C = "#0072B2", "#D55E00"            # Okabe–Ito
FURO = (-104.825, -75.207)                      # Davis et al. 2023
CELL = (-1547500.0, -412500.0)                  # célula do núcleo da Thwaites (Fig. 3a)

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 6.5,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "legend.frameon": False, "axes.labelsize": 6.5,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 5.8,
    "axes.formatter.use_locale": False,
})


def br(v, n=1):
    """Número com vírgula decimal e sinal de menos tipográfico."""
    return f"{v:.{n}f}".replace(".", ",").replace("-", "−")


COMMA = FuncFormatter(lambda v, _: br(v, 0) if float(v).is_integer() else br(v, 1))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.tiff", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(OUT / f"{name}_previa.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("ok", name)


def tag(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")


def frame(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)


def scalebar(ax, L, frac=0.05):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xa, ya = x1 - frac * (x1 - x0) - L, y0 + frac * (y1 - y0)
    ax.plot([xa, xa + L], [ya, ya], "k-", lw=1.2, solid_capstyle="butt")
    ax.text(xa + L / 2, ya + 0.02 * (y1 - y0), f"{L} km", ha="center", va="bottom", fontsize=5.6)


def bedmachine(xmin, xmax, ymin, ymax, step=1):
    bm = xr.open_dataset(BM_PATH)
    sub = bm["mask"].sel(x=slice(xmin, xmax), y=slice(ymax, ymin))
    if step > 1:
        sub = sub[::step, ::step]
    return sub.x.values, sub.y.values, sub.values


# =========================================================================================== Fig. 1
def fig1():
    xs, ys, mk = bedmachine(-1900e3, -1150e3, -800e3, -50e3, step=4)      # 2 km
    hm = xr.open_dataset(I / "hydrostatic_mask_500m.nc")
    v2 = xr.open_dataset(I / "basal_melt" / "mb_final_v2_500m.nc")
    unit = v2.unidade.values.astype(str)
    fig = plt.figure(figsize=(183 * MM, 82 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.9, 1.25], wspace=0.06)
    a = fig.add_subplot(gs[0, 0])
    img = np.full(mk.shape + (3,), 1.0)
    img[np.isin(mk, [1, 2, 4])] = mpl.colors.to_rgb("#D9D9D9")       # apoiado e rocha
    img[mk == 3] = mpl.colors.to_rgb("#DCEBF5")                       # flutuante
    ext = [xs.min() / 1e3, xs.max() / 1e3, ys.min() / 1e3, ys.max() / 1e3]
    a.imshow(img, extent=ext, origin="upper", interpolation="nearest")
    a.contour(xs / 1e3, ys / 1e3, np.isin(mk, [1, 2, 4]).astype(float), levels=[0.5], colors="k",
              linewidths=0.45)
    # domínio do produto basal por unidade (contorno colorido)
    hx, hy = v2.x.values / 1e3, v2.y.values / 1e3
    for u in UNITS + ["Thwaites_TWGT"]:
        msk = (unit == u).astype(float)
        if msk.sum() == 0:
            continue
        a.contourf(hx, hy, msk, levels=[0.5, 1.5], colors=[COL.get(u, "#B64342")], alpha=0.35)
        if u == "Thwaites_TWGT":
            a.contourf(hx, hy, msk, levels=[0.5, 1.5], colors="none", hatches=["xxxx"])
    labs = {"Pine_Island": (-1700, -262), "Thwaites_TEIS": (-1515, -500), "Crosson": (-1500, -590),
            "Dotson": (-1590, -665), "Cosgrove": (-1790, -330), "Thwaites_TWGT": (-1655, -470)}
    for u, (x, y) in labs.items():
        a.text(x, y, LABEL[u].replace("Thwaites ", "Thw. "), fontsize=5.6, ha="center",
               bbox={"fc": "w", "ec": "none", "pad": 0.5, "alpha": 0.85})
    a.add_patch(Rectangle(((CELL[0] - 2500) / 1e3, (CELL[1] - 2500) / 1e3), 5, 5, fill=False, ec="#D55E00",
                          lw=1.0))
    a.annotate("célula da\nFig. 3a", ((CELL[0]) / 1e3, (CELL[1]) / 1e3), (-1440, -360), fontsize=5.4,
               color="#D55E00", arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "#D55E00"})
    a.text(-1860, -760, "Mar de\nAmundsen", fontsize=6, color="#3775BA", style="italic")
    a.set_xlim(-1880, -1200)
    a.set_ylim(-780, -120)
    a.set_aspect("equal")
    frame(a)
    scalebar(a, 100)
    tag(a, "a", x=0.0, y=1.01)
    a.legend(handles=[mpl.patches.Patch(color="#D9D9D9", label="Gelo apoiado"),
                      mpl.patches.Patch(color="#DCEBF5", label="Gelo flutuante"),
                      mpl.patches.Patch(fc="#B64342", alpha=0.35, label="Domínio do produto basal"),
                      mpl.patches.Patch(fc="none", ec="k", hatch="xxxx", lw=0.3,
                                        label="TWGT (fora do produto)")],
             loc="lower right", bbox_to_anchor=(1.0, 0.105), fontsize=5.0, handlelength=1.0,
             frameon=True, framealpha=0.9, edgecolor="none")
    # inset de localização (Antártica)
    ins = a.inset_axes([0.70, 0.70, 0.29, 0.29])
    xs2, ys2, mk2 = bedmachine(-3.3e6, 3.3e6, -3.3e6, 3.3e6, step=60)
    ins.imshow(np.where(mk2 > 0, 0.75, 1.0), cmap="gray", vmin=0, vmax=1, origin="upper",
               extent=[xs2.min() / 1e3, xs2.max() / 1e3, ys2.min() / 1e3, ys2.max() / 1e3])
    ins.add_patch(Rectangle((-1880, -780), 680, 660, fill=False, ec="#B64342", lw=0.8))
    frame(ins)

    # ---- (b) esquema
    b = fig.add_subplot(gs[0, 1])
    b.set_xlim(0, 100)
    b.set_ylim(-62, 30)
    b.axis("off")
    sea = Polygon([[38, 0], [100, 0], [100, -62], [38, -62]], closed=True, fc="#DCEBF5", ec="none")
    b.add_patch(sea)
    b.plot([38, 100], [0, 0], color="#3775BA", lw=0.8)
    bed = Polygon([[0, -40], [20, -44], [33, -50], [45, -56], [60, -60], [100, -62], [100, -62.5],
                   [0, -62.5]], closed=True, fc="#BFA98A", ec="#7A6548", lw=0.6)
    b.add_patch(bed)
    # gelo: superfície e base
    xi = np.linspace(0, 100, 200)
    top = np.where(xi < 40, 24 - 0.25 * xi, np.interp(xi, [40, 47, 100], [14, 7, 4]))
    base = np.where(xi < 40, np.interp(xi, [0, 20, 33, 40], [-40, -44, -50, -52]),
                    np.interp(xi, [40, 47, 100], [-52, -46, -34]))
    b.fill_between(xi, base, top, color="#F3F6FA", ec="#4D4D4D", lw=0.7, zorder=2)
    # flexão
    b.add_patch(Rectangle((40, -60), 9, 88, fc="#F2C14E", alpha=0.25, ec="none", zorder=1))
    b.axvline(40, ymin=0.02, ymax=0.97, color="k", lw=0.8, ls="-", zorder=3)
    b.text(40, 27, "linha de\naterramento", ha="center", fontsize=5.6, va="bottom")
    b.annotate("", xy=(49, 20.5), xytext=(40, 20.5),
               arrowprops={"arrowstyle": "<->", "lw": 0.6, "shrinkA": 0, "shrinkB": 0})
    b.text(44.5, 21.3, "d*", ha="center", fontsize=6, style="italic")
    b.text(44.5, -47, "zona de flexão\n(β: 0 → 1)", ha="center", fontsize=5.0, color="#7A5A00", zorder=6,
           bbox={"fc": "w", "ec": "none", "alpha": 0.8, "pad": 0.5})
    b.text(12, 4, "GELO APOIADO\nsoma ao nível do mar\n(Fig. 2–3)", fontsize=5.6, ha="center",
           color="#4D4D4D", zorder=4)
    b.text(88, -14, "PLATAFORMA\nFLUTUANTE\nnão soma\ndiretamente\n(Fig. 4–6)", fontsize=5.4,
           ha="center", color="#4D4D4D", zorder=4)
    # termos do balanço (coluna)
    b.add_patch(Rectangle((62, -41), 8, 47, fill=False, ec="#B64342", lw=0.8, ls="--", zorder=4))
    b.annotate("", xy=(66, 6), xytext=(66, 14), arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": "#0F4D92"},
               zorder=5)
    b.text(67.5, 12, "SMB", fontsize=6, color="#0F4D92")
    b.annotate("", xy=(66, -48), xytext=(66, -40.5),
               arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": "#B64342"}, zorder=5)
    b.text(67.5, -48, "m_b (derretimento basal)", fontsize=6, color="#B64342")
    b.annotate("", xy=(78, -18), xytext=(55, -18), arrowprops={"arrowstyle": "-|>", "lw": 0.8},
               zorder=5)
    b.text(52, -16, "u", fontsize=6, style="italic")
    b.text(71, -2, r"$H\,\nabla\cdot u$", fontsize=6)
    b.text(71, -8, "DH/Dt", fontsize=6)
    b.text(50, -30.5, r"$m_b = \mathrm{SMB} - DH/Dt - H\,\nabla\cdot u$", fontsize=6.4, ha="left",
           bbox={"fc": "w", "ec": "#B64342", "lw": 0.5, "pad": 1.5}, zorder=6)
    # sensores
    b.annotate("ICESat-2", xy=(88, 6.0), xytext=(92, 26), fontsize=5.6, ha="center",
               arrowprops={"arrowstyle": "-", "lw": 0.5, "ls": ":"})
    b.annotate("ICESat-2", xy=(22, 19), xytext=(24, 28), fontsize=5.6, ha="center",
               arrowprops={"arrowstyle": "-", "lw": 0.5, "ls": ":"})
    b.plot([46.5], [8.6], "v", ms=4, color="k", zorder=6)
    b.text(46.5, 2.5, "ApRES", fontsize=5.0, ha="center", zorder=6)
    b.annotate("radar aerotransportado\n(MCoRDS, espessura)", xy=(55, 8.5), xytext=(54, 23), fontsize=5.4,
               arrowprops={"arrowstyle": "-|>", "lw": 0.5})
    b.text(88, -40, "maré\n(CATS2008 + DAC)", fontsize=5.4, ha="center", color="#3775BA")
    b.text(2, -61, "esquema fora de escala", fontsize=5, color="#767676", style="italic")
    tag(b, "b", x=0.02, y=0.97)
    save(fig, "substituida_py_fig1_area_e_esquema")


# =========================================================================================== Fig. 2
def fig2():
    G = {"JJA": O / "jja" / "experiments" / "firn_qc2" / "firn_corrected_grid.parquet",
         "DJF": O / "djf" / "experiments" / "djf_20260804T091211Z_firn" / "firn_corrected_grid.parquet"}
    g = {k: pd.read_parquet(p)[["x", "y", "dhdt_ice"]] for k, p in G.items()}
    cmp_ = json.loads((O / "comparacao_sazonal" / "comparacao_tendencia_nos_comuns.json").read_text("utf-8"))
    xs, ys, mk = bedmachine(-1900e3, -1150e3, -800e3, -50e3, step=4)
    fig = plt.figure(figsize=(183 * MM, 72 * MM))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.05, 0.42, 0.95], wspace=0.08)
    norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=1)
    cmap = mpl.colormaps["RdBu"]
    ims = None
    rows = []
    for i, (k, lab) in enumerate((("JJA", "Inverno (JJA)"), ("DJF", "Verão (DJF)"))):
        ax = fig.add_subplot(gs[0, i])
        d = g[k]
        ux, uy = np.sort(d.x.unique()), np.sort(d.y.unique())
        grid = d.pivot_table(index="y", columns="x", values="dhdt_ice").reindex(index=uy, columns=ux)
        ims = ax.pcolormesh(ux / 1e3, uy / 1e3, grid.values, cmap=cmap, norm=norm, shading="nearest",
                            rasterized=True)
        ax.contour(xs / 1e3, ys / 1e3, np.isin(mk, [1, 2, 4]).astype(float), levels=[0.5], colors="k",
                   linewidths=0.45)
        ax.set_xlim(-1850, -1200)
        ax.set_ylim(-740, -100)
        ax.set_aspect("equal")
        frame(ax)
        med = float(np.nanmedian(d.dhdt_ice))
        ax.text(0.03, 0.03, f"mediana {br(med, 2)} m/ano", transform=ax.transAxes, fontsize=5.8)
        ax.set_title(lab, fontsize=6.8, pad=2)
        tag(ax, "ab"[i], x=0.0, y=1.01)
        if i == 0:
            scalebar(ax, 100)
        rows.append(d.assign(estacao=k))
    cax = fig.add_subplot(gs[0, 2])
    p_ = cax.get_position()
    cax.set_position([p_.x0, p_.y0 + 0.12 * p_.height, p_.width, 0.76 * p_.height])
    cb = fig.colorbar(ims, cax=cax, extend="both", ticks=[-3, -2, -1, 0, 0.5, 1])
    cb.ax.set_yticklabels(["−3", "−2", "−1", "0", "0,5", "1"])
    cb.set_label("dh/dt do gelo (m/ano)")
    cb.outline.set_linewidth(0.4)
    pd.concat(rows).to_csv(SRC / "fig2ab_grades_dhdt.csv", index=False)
    # (c) célula a célula
    c = fig.add_subplot(gs[0, 4])
    m = g["JJA"].merge(g["DJF"], on=["x", "y"], suffixes=("_jja", "_djf")).dropna()
    r = float(np.corrcoef(m.dhdt_ice_jja, m.dhdt_ice_djf)[0, 1])
    hb = c.hexbin(m.dhdt_ice_djf, m.dhdt_ice_jja, gridsize=55, bins="log", cmap="Greys", mincnt=1,
                  extent=(-6, 1.5, -6, 1.5), lw=0)
    c.plot([-6, 1.5], [-6, 1.5], color="#B64342", lw=0.7, ls="--")
    c.set_xlim(-6, 1.5)
    c.set_ylim(-6, 1.5)
    c.set_aspect("equal")
    c.set_xlabel("dh/dt verão, DJF (m/ano)")
    c.set_ylabel("dh/dt inverno, JJA (m/ano)")
    c.xaxis.set_major_formatter(COMMA)
    c.yaxis.set_major_formatter(COMMA)
    d_ = cmp_["por_bloco"]["34.1"]["JJA_menos_DJF"]
    nfmt = f"{len(m):,}".replace(",", ".")
    c.text(0.04, 0.97, f"r = {br(r, 3)} ({nfmt} células;\ngrades interpoladas)", transform=c.transAxes,
           va="top", fontsize=5.6)
    c.text(0.96, 0.04, f"nós comuns, sem firn:\nJJA − DJF = {br(d_['media_m_ano'], 3)}\n± "
           f"{br(d_['sigma_m_ano'], 3)} m/ano\n(≈ {br(d_['Gt_ano'], 1)} Gt/ano)", transform=c.transAxes,
           va="bottom", ha="right", fontsize=5.4, color="#4D4D4D")
    tag(c, "c", x=-0.22, y=1.01)
    m.to_csv(SRC / "fig2c_celula_a_celula.csv", index=False)
    save(fig, "fig2_dhdt_inverno_verao")
    return r, len(m)


# =========================================================================================== Fig. 3
def fig3():
    cell = pd.read_csv(O / "grounded_H_timeseries" / "thwaites_nucleo_metodo_TEIS" / "serie_metodo_TEIS.csv")
    rep = json.loads((O / "grounded_H_timeseries" / "thwaites_nucleo_metodo_TEIS" /
                      "relatorio_metodo_TEIS.json").read_text("utf-8"))
    tr = rep["tendencia"]
    gpm = rep["Gt_por_metro"]
    fig = plt.figure(figsize=(183 * MM, 66 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.05], wspace=0.78)
    a = fig.add_subplot(gs[0, 0])
    t = cell.t_center_year.values
    col = np.where(cell.season == "JJA", JJA_C, DJF_C)
    a.axhline(0, color="#9A9A9A", lw=0.5)
    tm = rep["tendencia"]["t_reference"]
    a.plot(t, (tr["intercept_at_mean_time"] - cell.h_ice_res.iloc[0] + tr["slope_m_per_year"] * (t - tm)) * gpm,
           ls="--", color="k", lw=0.7, zorder=1)
    for s, cc in (("JJA", JJA_C), ("DJF", DJF_C)):
        k = cell.season == s
        a.errorbar(t[k], cell.dM_Gt[k], yerr=cell.sigma_dM_Gt[k], fmt="o", ms=3, color=cc, mec="w", mew=0.4,
                   elinewidth=0.6, capsize=0, label="Inverno (JJA)" if s == "JJA" else "Verão (DJF)", zorder=3)
    a.set_xlabel("Ano")
    a.set_ylabel("Anomalia de massa da célula (Gt)")
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: br(v, 1)))
    sec = a.secondary_yaxis("right", functions=(lambda g_: g_ / gpm, lambda h: h * gpm))
    sec.set_ylabel("Δh (m)")
    sec.yaxis.set_major_formatter(FuncFormatter(lambda v, _: br(v, 0)))
    a.spines["right"].set_visible(True)
    a.legend(loc="lower left", fontsize=5.4)
    a.text(0.97, 0.97, f"Theil–Sen: {br(tr['slope_m_per_year'], 2)} ± {br(tr['slope_sigma_m_per_year'], 2)} m/ano\n"
           f"≈ {br(rep['dM_dt_Gt_ano'], 3)} Gt/ano (25 km²)\nfirn não descontado",
           transform=a.transAxes, ha="right", va="top", fontsize=5.4,
           bbox={"fc": "w", "ec": "none", "pad": 0.8})
    a.set_title("Núcleo da Thwaites, célula fixa de 5 km", fontsize=6.3, pad=3)
    tag(a, "a", x=-0.28, y=1.02)
    cell.to_csv(SRC / "fig3a_celula_thwaites.csv", index=False)

    # (b) série sazonal do domínio
    b = fig.add_subplot(gs[0, 1])
    rows = []
    for s, cc in (("jja", JJA_C), ("djf", DJF_C)):
        d = json.loads((O / s / "tables" / "serie_massa_temporada.json").read_text("utf-8"))
        tt = np.array([x["epoca_media"] for x in d["serie"]])
        mm = np.array([x["massa_acumulada_Gt"] for x in d["serie"]])
        b.plot(tt, mm, "-o", color=cc, ms=2.8, lw=0.9, mec="w", mew=0.3,
               label=f"{s.upper()}: {br(d['taxa_Gt_por_ano_ajuste_linear'], 1)} Gt/ano")
        rows += [{"estacao": s.upper(), "epoca": x, "massa_acumulada_Gt": y} for x, y in zip(tt, mm)]
    b.axhline(0, color="#9A9A9A", lw=0.5)
    b.set_xlabel("Ano")
    b.set_ylabel("Massa acumulada do gelo apoiado (Gt)")
    b.legend(loc="lower left", fontsize=5.2)
    b.text(0.97, 0.62, "h − REMA, nós completos,\nsem firn, sem σ por temporada", transform=b.transAxes,
           ha="right", va="top", fontsize=5.2, color="#4D4D4D", bbox={"fc": "w", "ec": "none", "pad": 0.8})
    tag(b, "b", x=-0.25, y=1.02)
    pd.DataFrame(rows).to_csv(SRC / "fig3b_serie_dominio.csv", index=False)

    # (c) dM/dt com L e GIA
    c = fig.add_subplot(gs[0, 2])
    mb = json.loads((O / "djf" / "experiments" / "djf_20260804T091211Z_mass_firn" / "mass_balance.json")
                    .read_text("utf-8"))
    raw = json.loads((O / "djf" / "experiments" / "djf_20260804T091211Z_firn" / "mass_balance_firn.json")
                     .read_text("utf-8"))["raw"]
    gia = json.loads((O / "djf" / "experiments" / "gia_djf" / "mass_balance_gia.json").read_text("utf-8"))
    s34 = mb["sigma_dMdt_Gt_yr_correlated"]
    L0 = mb["correlation_length_m"] / 1e3
    s154, s20 = s34 * 154.4 / L0, s34 * 20.5 / L0
    firn = mb["dMdt_Gt_yr"]
    e_car = gia["sensibilidade_estrutural"]["efeito_na_massa_caron_Gt_ano"]
    e_bar = gia["sensibilidade_estrutural"]["efeito_na_massa_escala_barletta_Gt_ano"]
    bars = [("Só altura\n(sem firn)", raw["dMdt_Gt_yr"], "#BDBDBD", None),
            ("Corrigido por firn\n(valor adotado)", firn, "#B64342", None),
            ("+ GIA (Caron 2018)", firn + e_car, "#E9A6A1", None),
            ("+ GIA, escala do\nGPS (Barletta 2018)", firn + e_bar, "#F6CFCB", "////")]
    y = np.arange(len(bars))[::-1]
    for yi, (lab, v, cc, h) in zip(y, bars):
        c.barh(yi, v, color=cc, height=0.6, hatch=h, ec="w" if h else cc, lw=0)
        c.errorbar(v, yi, xerr=s154, color="#767676", lw=0.6, capsize=2)
        c.errorbar(v, yi, xerr=s20, color="k", lw=0.6, ls="none", capsize=0)
        c.errorbar(v, yi, xerr=s34, color="k", lw=2.0, capsize=0)
        c.text(v - 3, yi + 0.33, br(v, 1), ha="right", va="center", fontsize=5.6,
               fontweight="bold" if "adotado" in lab else None)
    c.set_yticks(y)
    c.set_yticklabels([b_[0] for b_ in bars], fontsize=5.6)
    c.axvline(0, color="k", lw=0.6)
    c.set_xlim(-165, 5)
    c.set_xlabel("dM/dt do gelo apoiado (Gt/ano)")
    c.xaxis.set_major_formatter(COMMA)
    top = c.secondary_xaxis("top", functions=(lambda g_: -g_ / GT_PER_MM, lambda s_: -s_ * GT_PER_MM))
    top.set_xlabel("nível do mar (mm/ano)", fontsize=6)
    top.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    top.xaxis.set_major_formatter(FuncFormatter(lambda v, _: br(v, 1)))
    c.text(-0.62, -0.2, f"Barra grossa: ±σ, L = {br(L0)} km (±{br(s34)}); fina preta: L = 20,5 km "
           f"(±{br(s20)});\ncinza: L = 154,4 km (±{br(s154)}). DJF 2019/20–2025; "
           f"{br(mb['area_total_km2'] / 1e3, 1)} mil km².", transform=c.transAxes, fontsize=5.0,
           color="#4D4D4D", va="top")
    tag(c, "c", x=-0.48, y=1.02)
    pd.DataFrame([{"versao": b_[0].replace("\n", " "), "dMdt_Gt_ano": b_[1], "sigma_L34": s34,
                   "sigma_L20": s20, "sigma_L154": s154} for b_ in bars]).to_csv(SRC / "fig3c_balanco.csv",
                                                                                 index=False)
    save(fig, "substituida_py_fig3_massa_gelo_apoiado")


# =========================================================================================== Fig. 4
_FLAGS = {}


def mark_negatives(ax, xs, ys, lw=0.3):
    if not _FLAGS:                       # uma leitura só: reabrir o NetCDF no Windows dá HDF error
        sys.path.insert(0, str(ROOT / "pipelines"))
        from run_blue_spots_ase import negative_flags
        _FLAGS["v"] = negative_flags()
    nr, rb = _FLAGS["v"]
    mpl.rcParams["hatch.linewidth"] = lw
    ax.contourf(xs / 1e3, ys / 1e3, nr.astype(float), levels=[0.5, 1.5], colors="none", hatches=["////"])
    ax.contour(xs / 1e3, ys / 1e3, rb.astype(float), levels=[0.5], colors="k", linewidths=lw * 2.5)


def fig4():
    with xr.open_dataset(I / "basal_melt" / "mb_final_v2_500m.nc") as v2:
        xs, ys, m = v2.x.values, v2.y.values, v2.mb.values
    with xr.open_dataset(I / "hydrostatic_mask_500m.nc") as mask:
        cls, plat = mask.classe.values, mask.plataforma.values
    bm = xr.open_dataset(BM_PATH)["mask"].sel(x=slice(xs.min(), xs.max()), y=slice(ys.max(), ys.min())).values
    ext = [xs.min() / 1e3, xs.max() / 1e3, ys.min() / 1e3, ys.max() / 1e3]
    norm = SymLogNorm(linthresh=1.0, linscale=0.6, vmin=-5, vmax=60, base=10)
    cmap = mpl.colormaps["RdBu_r"]
    fig = plt.figure(figsize=(183 * MM, 108 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], wspace=0.08, hspace=0.12)
    axa, axb, axc = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    bg = np.full(bm.shape, np.nan)
    bg[np.isin(bm, [1, 2, 4])] = 0.86
    bg[(bm == 3) & (cls == 0)] = 0.95
    bg[cls > 0] = 0.68
    tf = Transformer.from_crs(4326, 3031, always_xy=True)
    bx, by = tf.transform(*FURO)
    w = pd.read_csv(ROOT / "data" / "validation" / "wild2024_apres_teis.csv", comment="#")
    wx, wy = tf.transform(w.lon.values, w.lat.values)
    im = None
    for ax, box in ((axa, None), (axb, (-1600, -1500, -515, -395)), (axc, (-1680, -1570, -345, -240))):
        ax.imshow(bg, cmap="gray", vmin=0, vmax=1, extent=ext, interpolation="nearest")
        im = ax.imshow(m, cmap=cmap, norm=norm, extent=ext, interpolation="nearest")
        mark_negatives(ax, xs, ys)
        ax.set_aspect("equal")
        frame(ax)
        if box:
            ax.set_xlim(box[0], box[1])
            ax.set_ylim(box[2], box[3])
            axa.add_patch(Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2], fill=False, lw=0.6, ec="k"))
    ii, jj = np.nonzero(np.isfinite(m))
    axa.set_xlim(xs[jj].min() / 1e3 - 15, xs[jj].max() / 1e3 + 15)
    axa.set_ylim(ys[ii].min() / 1e3 - 15, ys[ii].max() / 1e3 + 15)
    for k, n, (dx, dy) in ((1, "Pine Island", (-60, 30)), (2, "Thwaites", (-62, 0)), (3, "Crosson", (25, 15)),
                           (4, "Dotson", (-75, -25)), (5, "Cosgrove", (-10, 45))):
        ii, jj = np.nonzero((plat == k) & (cls > 0))
        axa.text(np.median(xs[jj]) / 1e3 + dx, np.median(ys[ii]) / 1e3 + dy, n, fontsize=6.2,
                 bbox={"fc": "w", "ec": "none", "pad": 0.6, "alpha": 0.8})
    axb.plot(wx / 1e3, wy / 1e3, "o", ms=2.0, mfc="none", mec="k", mew=0.4, label="ApRES (Wild 2024)")
    axb.plot(bx / 1e3, by / 1e3, "*", ms=6, mfc="#F2C14E", mec="k", mew=0.4, label="Furo e ApRES (Davis 2023)")
    axb.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=5.4, handletextpad=0.3)
    for ax, t in ((axa, "a"), (axb, "b"), (axc, "c")):
        tag(ax, t, x=0.0, y=1.01)
    axa.legend(handles=[mpl.patches.Patch(color="0.86", label="Gelo apoiado"),
                        mpl.patches.Patch(color="0.68", label="Plataforma não medida"),
                        mpl.patches.Patch(fc="0.95", ec="0.6", lw=0.3, label="Fora da máscara de 2021"),
                        mpl.patches.Patch(fc="none", ec="k", lw=0.3, hatch="////",
                                          label="m_b < 0 que muda com a escala de div(u)"),
                        mpl.patches.Patch(fc="none", ec="k", lw=0.8, label="m_b < 0 robusto (sem explicação)")],
               loc="lower left", bbox_to_anchor=(0.0, 0.3), fontsize=5.2, handlelength=1.0)
    axb.set_title("Thwaites", fontsize=6.5, pad=2)
    axc.set_title("Pine Island", fontsize=6.5, pad=2)
    for ax, L in ((axa, 50), (axb, 10), (axc, 10)):
        scalebar(ax, L)
    cb = fig.colorbar(im, ax=[axa, axb, axc], orientation="horizontal", fraction=0.035, pad=0.03, aspect=55,
                      ticks=[-5, -1, 0, 1, 5, 10, 20, 50])
    cb.ax.set_xticklabels(["−5", "−1", "0", "1", "5", "10", "20", "50"])
    cb.set_label("Derretimento basal médio 2019–2025 (m de gelo por ano)")
    cb.outline.set_linewidth(0.4)
    ok = np.isfinite(m)
    pd.DataFrame({"x_m": np.repeat(xs[None, :], len(ys), 0)[ok], "y_m": np.repeat(ys[:, None], len(xs), 1)[ok],
                  "mb_mpa": m[ok]}).to_csv(SRC / "fig4_mapa_mb.csv", index=False)
    save(fig, "fig4_mapa_derretimento_basal")


# =========================================================================================== Fig. 5
def fig5():
    beta = pd.read_csv(PUB / "fig2a_beta.csv")
    rad = pd.read_csv(PUB / "fig2b_radar.csv")
    cor = pd.read_csv(PUB / "fig2c_correcao.csv")
    ap = pd.read_csv(O / "validation" / "apres_pontos.csv")
    box = pd.read_csv(PUB / "fig3c_apres.csv")
    dstar = {k: v["d_star_km"]["f090"] for k, v in
             json.loads(xr.open_dataset(I / "hydrostatic_mask_500m.nc").classe.attrs["parametros"]).items()}
    fig, ax = plt.subplots(1, 4, figsize=(183 * MM, 52 * MM), gridspec_kw={"wspace": 0.5,
                                                                          "width_ratios": [1.2, 1, 1, 0.85]})
    for s in ["Pine_Island", "Thwaites", "Crosson", "Dotson", "Cosgrove"]:
        g = beta[beta.plataforma == s].sort_values("dist_km")
        ax[0].plot(g.dist_km, g.beta_mediano, color=COL[s], lw=1.0, label=LABEL[s])
        ax[0].plot([dstar[s]], [np.interp(dstar[s], g.dist_km, g.beta_mediano)], "o", ms=3, color=COL[s], mec="w",
                   mew=0.4)
    ax[0].axvspan(0, 2, color="#CFCECE", alpha=0.45, lw=0)
    ax[0].text(1, 1.3, "recorte fixo\nde 2 km", ha="center", fontsize=5.4, color="#4D4D4D")
    ax[0].axhline(1, color="#767676", lw=0.5, ls="--")
    ax[0].axvline(0, color="k", lw=0.5)
    ax[0].set_xlim(-4, 10)
    ax[0].set_ylim(-0.2, 1.5)
    ax[0].set_xlabel("Distância à linha de aterramento (km)\n← apoiado   flutuante →")
    ax[0].set_ylabel("Resposta à maré, β")
    ax[0].legend(loc="lower right", fontsize=5.0, handlelength=1.2)
    ax[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: br(v, 1)))
    for s in ["Pine_Island", "Thwaites", "Crosson", "Dotson", "Cosgrove"]:
        g = rad[(rad.plataforma == s) & (rad["mid"] > 0)].sort_values("mid")
        ax[1].plot(g["mid"], g.razao_norm_med, "-o", color=COL[s], lw=0.9, ms=2.0)
    ax[1].axhspan(0.97, 1.03, color="#CFCECE", alpha=0.45, lw=0)
    ax[1].axhline(1, color="#767676", lw=0.5, ls="--")
    ax[1].set_xlabel("Distância à linha (km)")
    ax[1].set_ylabel("Espessura radar / hidrostática\n(normalizada por voo)")
    ax[1].set_ylim(0.8, 1.1)
    ax[1].yaxis.set_major_formatter(FuncFormatter(lambda v, _: br(v, 2)))
    for v, cc, lab in (("sem_correcao", "#767676", "Hidrostático"), ("corrigido", "#B64342", "Corrigido")):
        g = cor[cor.versao == v]
        ax[2].plot(g.dist_km, g.mb_mediana, "-o", color=cc, ms=2.2, lw=1.0, label=lab)
    ax[2].set_xlim(0, 12)
    ax[2].set_xlabel("Distância à linha (km)")
    ax[2].set_ylabel("m_b na TEIS, mediana (m/ano)")
    ax[2].legend(loc="upper right", fontsize=5.4)
    d = ax[3]
    d.add_patch(Rectangle((-0.4, 2.0), 0.8, 3.4, color="#F2C14E", alpha=0.6, lw=0))
    a3 = ap[ap.ponto != "G2"]
    d.errorbar(np.linspace(-0.15, 0.15, len(a3)), a3.melt_mpa,
               yerr=[a3.melt_mpa - a3.ic95_lo, a3.ic95_hi - a3.melt_mpa], fmt="o", ms=2.2, color="k", elinewidth=0.5)
    for xi, grp, cc in ((1, "hidrostatico", "#767676"), (2, "corrigido", "#B64342")):
        v = box[box.grupo == grp].mb_mpa.values
        d.boxplot([v], positions=[xi], widths=0.45, showfliers=False, patch_artist=True,
                  boxprops={"facecolor": cc, "alpha": 0.35, "lw": 0.5}, medianprops={"color": cc, "lw": 1.2},
                  whiskerprops={"lw": 0.5}, capprops={"lw": 0.5})
    d.set_xticks([0, 1, 2])
    d.set_xticklabels(["ApRES\n2020", "Hidro-\nstático", "Corri-\ngido"], fontsize=5.6)
    d.set_ylabel("m_b (m/ano)")
    d.set_title("TEIS, ≤ 3 km do furo", fontsize=6, pad=3)
    for a_, t in zip(ax, "abcd"):
        tag(a_, t, x=-0.28)
    for f in ("fig2a_beta", "fig2b_radar", "fig2c_correcao", "fig3c_apres"):
        pd.read_csv(PUB / f"{f}.csv").to_csv(SRC / f"fig5_{f}.csv", index=False)
    save(fig, "fig5_zona_aterramento_validacao")


# =========================================================================================== Fig. 6
def fig6():
    c = pd.read_csv(PUB / "fig3a_comparacao.csv").set_index("unidade")
    inc = pd.read_csv(PUB / "fig3b_incerteza.csv")
    T = pd.read_csv(O / "basal_melt_ase" / "TABELA_FINAL_derretimento_basal.csv").set_index("unidade")
    fig = plt.figure(figsize=(183 * MM, 64 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.05, 0.95], wspace=0.5)
    a, b, m = (fig.add_subplot(gs[0, i]) for i in range(3))
    y = np.arange(len(UNITS))[::-1]
    for yi, u in zip(y, UNITS):
        a.errorbar(c.loc[u, "este_trabalho"], yi + 0.2, xerr=c.loc[u, "sigma_total"], fmt="o", ms=3.2, color=COL[u],
                   elinewidth=0.8)
        a.errorbar(c.loc[u, "este_trabalho"], yi + 0.2, xerr=c.loc[u, "sigma_estat"], fmt="none", color="k",
                   elinewidth=2.0)
        a.errorbar(c.loc[u, "meng"], yi, xerr=c.loc[u, "meng_sigma"], fmt="s", ms=2.6, mfc="w", mec="k", ecolor="k",
                   elinewidth=0.6, mew=0.5)
        a.plot(c.loc[u, "paolo_1992_2017_mesmos_pixels"], yi - 0.2, "^", ms=3, mfc="#CFCECE", mec="k", mew=0.5)
    a.set_yticks(y)
    a.set_yticklabels([LABEL[u] for u in UNITS])
    a.set_xlabel("Derretimento basal médio (m/ano)")
    a.set_xlim(-1, 24)
    h = [mpl.lines.Line2D([], [], marker="o", color="#4D4D4D", ls="", ms=3.2),
         mpl.lines.Line2D([], [], marker="s", mfc="w", mec="k", ls="", ms=2.6),
         mpl.lines.Line2D([], [], marker="^", mfc="#CFCECE", mec="k", ls="", ms=3)]
    a.legend(h, ["Este trabalho 2019–2025", "Meng et al. 2025 (2019–2023)", "Paolo et al. 2023 (1992–2017)"],
             loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=5.2, handletextpad=0.2)
    facs = list(dict.fromkeys(inc.fator))
    pal = ["#0F4D92", "#42949E", "#9A4D8E", "#B64342", "#8BCF8B", "#CFCECE", "#767676"]
    left = np.zeros(len(UNITS))
    for f, cc in zip(facs, pal):
        v = np.array([inc[(inc.unidade == u) & (inc.fator == f)].efeito_mpa.iloc[0] for u in UNITS])
        b.barh(y, v, left=left, color=cc, height=0.62, label=f, lw=0)
        left += v
    b.set_yticks(y)
    b.set_yticklabels([LABEL[u].replace(" (TEIS)", "\n(TEIS)") for u in UNITS], fontsize=5.6)
    b.set_xlabel("Efeito sobre a média (m/ano)")
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), fontsize=4.9, handlelength=0.9, ncol=2,
             columnspacing=0.6, handletextpad=0.3)
    for yi, u in zip(y, UNITS):
        g, sg = T.loc[u, "Gt_medido"], T.loc[u, "sigma_Gt_medido"]
        lo, hi, sgi = T.loc[u, "Gt_inteira_min"], T.loc[u, "Gt_inteira_max"], T.loc[u, "sigma_Gt_inteira"]
        m.plot([g - sg, g + sg], [yi + 0.15] * 2, color=COL[u], lw=0.8)
        m.plot(g, yi + 0.15, "o", ms=3.4, color=COL[u])
        m.plot([lo - sgi, hi + sgi], [yi - 0.15] * 2, color=COL[u], lw=0.6, alpha=0.6)
        m.plot([lo, hi], [yi - 0.15] * 2, color=COL[u], lw=3.0, alpha=0.45, solid_capstyle="butt")
    tot = T.loc["TOTAL (sem TWGT)"]
    m.set_yticks(y)
    m.set_yticklabels([])
    m.set_xlim(0, 115)
    m.set_xlabel("Massa derretida (Gt/ano)")
    m.set_ylim(-0.6, 4.6)
    m.text(0.98, 0.42, f"Total sem TWGT\nmedido: {br(tot.Gt_medido, 0)} ± {br(tot.sigma_Gt_medido, 0)}\n"
           f"inteira: {br(tot.Gt_inteira_min, 0)}–{br(tot.Gt_inteira_max, 0)} ± {br(tot.sigma_Gt_inteira, 0)}",
           transform=m.transAxes, ha="right", va="bottom", fontsize=5.2)
    m.legend(handles=[mpl.lines.Line2D([], [], marker="o", color="#4D4D4D", ms=3, lw=0.8,
                                       label="Domínio medido (± σ)"),
                      mpl.lines.Line2D([], [], color="#4D4D4D", lw=3, alpha=0.45,
                                       label="Plataforma inteira\n(faixa de extrapolação; fina: ± σ)")],
             loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=5.0, handlelength=1.6)
    for ax, t, x in ((a, "a", -0.34), (b, "b", -0.25), (m, "c", -0.06)):
        tag(ax, t, x=x, y=1.03)
    c.to_csv(SRC / "fig6a_comparacao.csv")
    inc.to_csv(SRC / "fig6b_incerteza.csv", index=False)
    T.to_csv(SRC / "fig6c_massa.csv")
    save(fig, "fig6_plataformas_incerteza_massa")


# =========================================================================================== Fig. 7
def fig7():
    a_ = pd.read_csv(PUB / "fig4_mb_anual.csv")
    chi = pd.read_csv(PUB / "fig4_teste_variabilidade.csv").set_index("unidade")
    ev = pd.read_csv(PUB / "fig4_desprendimentos.csv")
    te = pd.read_csv(O / "series_v2" / "teste_resposta_desprendimento.csv").set_index("conjunto")
    fig = plt.figure(figsize=(183 * MM, 92 * MM))
    gs = fig.add_gridspec(2, 3, wspace=0.42, hspace=0.55)
    pos = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    for (r, cc_), u, t in zip(pos, UNITS, "abcde"):
        ax = fig.add_subplot(gs[r, cc_])
        g = a_[a_.unidade == u].sort_values("ano")
        mu, smu = chi.loc[u, "media_ponderada"], chi.loc[u, "sigma_media"]
        ax.axhspan(mu - smu, mu + smu, color=COL[u], alpha=0.12, lw=0)
        ax.axhline(mu, color=COL[u], lw=0.6)
        for _, e in ev[ev.unidade == u].iterrows():
            ax.axvspan(e.inicio, e.fim, color="#F2C14E", alpha=0.45, lw=0)
        ax.errorbar(g.ano + 0.5, g.mb_mpa, yerr=g.se_mpa, fmt="o", ms=2.8, color=COL[u], mec="w", mew=0.3,
                    elinewidth=0.6, capsize=0)
        ax.set_xlim(2019, 2026)
        ax.set_xticks([2019, 2021, 2023, 2025])
        ax.set_title(LABEL[u], fontsize=6.5, pad=2)
        p = chi.loc[u, "p"]
        ptxt = "p < 0,001" if p < 0.001 else f"p = {br(p, 2)}"
        ax.text(0.03, 0.96, f"χ² = {br(chi.loc[u, 'chi2'], 1)}, {ptxt}", transform=ax.transAxes, va="top",
                fontsize=5.4, fontweight="bold" if p < 0.05 else None)
        if cc_ == 0:
            ax.set_ylabel("m_b anual (m/ano)")
        tag(ax, t, x=-0.12)
    f = fig.add_subplot(gs[1, 2])
    rows = [("PIG inteira", "Pine_Island"), ("PIG > 10 km", "Pine_Island_longe_>10km"),
            ("PIG ≤ 10 km", "Pine_Island_perto_<=10km"), ("TEIS", "Thwaites_TEIS"), ("Crosson", "Crosson"),
            ("Dotson", "Dotson"), ("Cosgrove", "Cosgrove")]
    yy = np.arange(len(rows))[::-1]
    for yi, (lab, k) in zip(yy, rows):
        cc = "#0F4D92" if k.startswith("Pine") else "#767676"
        f.errorbar(te.loc[k, "delta"], yi, xerr=te.loc[k, "sigma_delta"], fmt="o", ms=2.8, color=cc, elinewidth=0.7)
    f.axvline(0, color="k", lw=0.5)
    f.set_yticks(yy)
    f.set_yticklabels([r[0] for r in rows], fontsize=5.6)
    f.set_xlabel("m_b depois − antes (m/ano)")
    f.set_title("Desprendimento da PIG, 2020", fontsize=6.5, pad=2)
    tag(f, "f", x=-0.3)
    a_.to_csv(SRC / "fig7_mb_anual.csv", index=False)
    te.to_csv(SRC / "fig7f_teste_desprendimento.csv")
    save(fig, "fig7_variabilidade_resultados_negativos")


# =========================================================================================== Fig. 8
def fig8():
    s = pd.read_csv(EV / "fase7_triagem" / "S3_series.csv")
    res = json.loads((EV / "fase7c_degrau" / "resumo.json").read_text("utf-8"))
    st = {(r["dominio"], r["variavel"]): r for r in res["teste"] if r["populacao"] == "com QC"}
    s1 = pd.read_csv(EV / "fase7e_sentinel1" / "s1_fracao_brilhante.csv", parse_dates=["data"])
    fig, ax = plt.subplots(1, 3, figsize=(183 * MM, 54 * MM), gridspec_kw={"wspace": 0.45})
    for i, (var, ylab, sc) in enumerate((("gelo", "Cobertura de gelo (%)", 100), ("H", "Espessura efetiva (m)", 1))):
        a = ax[i]
        for dom, cc in (("TEIS coerente", "#9A9A9A"), ("tronco e mélange", "#B64342")):
            g = s[(s.dominio == dom) & (s.variavel == var)]
            a.plot(g.t, g.valor * sc, "o-", ms=2.2, lw=0.6, color=cc, mec="w", mew=0.3,
                   label="Tronco oeste e mélange" if dom.startswith("tronco") else "TEIS (controle)")
        r = st[("tronco e mélange", var)]
        t0 = r["data_degrau"]
        a.hlines(r["antes"] * sc, 2019.0, t0, color="k", lw=0.8)
        a.hlines(r["depois"] * sc, t0, 2025.9, color="k", lw=0.8)
        a.axvspan(2024.10, 2024.19, color="#F2C14E", alpha=0.6, lw=0)
        unit = " pontos" if var == "gelo" else " m"
        a.text(0.03, 0.05 if var == "gelo" else 0.5, f"degrau {br(r['degrau'] * sc, 1)} ± {br(r['sigma_jk'] * sc, 1)}"
               f"{unit}\n({br(abs(r['z']), 1)}σ; forma abrupta não\nestabelecida só com ICESat-2)",
               transform=a.transAxes, fontsize=5.2, va="bottom")
        a.set_ylabel(ylab)
        a.set_xlabel("Ano")
        a.set_xlim(2019, 2026)
        tag(a, "ab"[i], x=-0.22)
    ax[0].legend(loc="center left", fontsize=5.2)
    c = ax[2]
    w = s1[~s1.data.dt.month.isin([12, 1, 2])]
    c.plot(w.data, w.fracao_brilhante_fixo * 100, "o-", ms=2.2, lw=0.6, color="#4D4D4D", label="215 blocos")
    c.plot(w.data, w.fixo_blocos_que_perdem * 100, "o-", ms=2.2, lw=0.6, color="#B64342",
           label="97 blocos que perdem gelo")
    c.axvspan(pd.Timestamp("2024-02-05"), pd.Timestamp("2024-03-12"), color="#F2C14E", alpha=0.6, lw=0)
    c.set_ylabel("Pixels de alto retroespalhamento (%)")
    c.set_xlabel("Data (Sentinel-1, órbita 21)")
    c.legend(loc="lower left", fontsize=5.2)
    c.xaxis.set_major_locator(mpl.dates.YearLocator(2))
    c.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y"))
    tag(c, "c", x=-0.22)
    s.to_csv(SRC / "fig8ab_series_icesat2.csv", index=False)
    s1.to_csv(SRC / "fig8c_sentinel1.csv", index=False)
    save(fig, "fig8_evento_2024_tronco_thwaites")


def main() -> int:
    SRC.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:])
    for n, f in (("1", fig1), ("2", fig2), ("3", fig3), ("4", fig4), ("5", fig5), ("6", fig6), ("7", fig7),
                 ("8", fig8)):
        if not only or n in only:
            out = f()
            if out is not None:
                print("fig", n, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
