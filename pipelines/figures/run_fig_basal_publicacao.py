"""Figuras de publicação do derretimento basal da ASE (2019–2025).

Fig. 1 — mapa do derretimento basal médio (a: ASE; b: Thwaites com ApRES; c: Pine Island).
Fig. 2 — por que o recorte fixo de 2 km não serve: β × distância (a), radar/hidrostático (b),
         efeito da correção na Thwaites (c).
Fig. 3 — comparação e incerteza: m_b por plataforma × Meng 2025 × Paolo 2023 (a), decomposição
         da incerteza estrutural (b), validação ApRES (c).

Tamanho final 183 mm; texto 6–7 pt; SVG (texto editável), PDF e TIFF 600 dpi; um CSV de dados de
origem por painel em ``outputs/figuras_publicacao/dados/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import SymLogNorm
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
I = ROOT / "data" / "interim"
O = ROOT / "outputs"
OUT = O / "figuras_publicacao"
SRC = OUT / "dados"
MM = 1 / 25.4
UNITS = ["Pine_Island", "Thwaites_TEIS", "Crosson", "Dotson", "Cosgrove"]
LABEL = {"Pine_Island": "Pine Island", "Thwaites_TEIS": "Thwaites (TEIS)", "Crosson": "Crosson",
         "Dotson": "Dotson", "Cosgrove": "Cosgrove", "Thwaites_TWGT": "Thwaites (TWGT)",
         "Thwaites": "Thwaites"}
# uma cor por plataforma, reutilizada em todas as figuras
SHELF_COL = {"Pine_Island": "#0F4D92", "Thwaites": "#B64342", "Thwaites_TEIS": "#B64342",
             "Crosson": "#42949E", "Dotson": "#9A4D8E", "Cosgrove": "#767676"}
FURO = (-104.825, -75.207)                    # Davis et al. 2023

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 6.5,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "legend.frameon": False, "axes.labelsize": 6.5,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 5.8,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / f"{name}_previa.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def tag(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")


# ------------------------------------------------------------------------------------------ dados
def final_map():
    """Produto central v2 (frente por data; Thwaites com correção β·maré + radar)."""
    v2 = xr.open_dataset(I / "basal_melt" / "mb_final_v2_500m.nc")
    mask = xr.open_dataset(I / "hydrostatic_mask_500m.nc")
    return v2.x.values, v2.y.values, v2.mb.values, mask.classe.values, mask.plataforma.values


def mark_negatives(ax, xs, ys, lw=0.3):
    """Hachura nos pixels com m_b < 0 cujo sinal muda com a escala de ∇·u (1–5 km) e contorno nos
    negativos robustos. Diagnóstico: ``run_blue_spots_ase.py``."""
    import sys
    sys.path.insert(0, str(ROOT / "pipelines"))
    from run_blue_spots_ase import negative_flags
    nr, rb = negative_flags()
    mpl.rcParams["hatch.linewidth"] = lw
    ax.contourf(xs / 1e3, ys / 1e3, nr.astype(float), levels=[0.5, 1.5], colors="none", hatches=["////"])
    ax.contour(xs / 1e3, ys / 1e3, rb.astype(float), levels=[0.5], colors="k", linewidths=lw * 2.5)


def bedmachine_mask(xs, ys):
    bm = xr.open_dataset(ROOT / "data" / "NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc")
    return bm["mask"].sel(x=slice(xs.min(), xs.max()), y=slice(ys.max(), ys.min())).values


# ------------------------------------------------------------------------------------------ Fig. 1
def fig1():
    xs, ys, m, cls, plat = final_map()
    bmm = bedmachine_mask(xs, ys)
    ext = [xs.min() / 1e3, xs.max() / 1e3, ys.min() / 1e3, ys.max() / 1e3]
    norm = SymLogNorm(linthresh=1.0, linscale=0.6, vmin=-5, vmax=60, base=10)
    cmap = mpl.colormaps["RdBu_r"].copy()
    fig = plt.figure(figsize=(183 * MM, 108 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], wspace=0.08, hspace=0.12)
    axa = fig.add_subplot(gs[:, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 1])
    bg = np.full(bmm.shape, np.nan)
    bg[np.isin(bmm, [1, 2, 4])] = 0.86                       # gelo apoiado e rocha
    bg[(bmm == 3) & (cls == 0)] = 0.95                       # flutuante no BedMachine, ausente em 2021
    bg[cls > 0] = 0.68                                       # plataforma (2021) fora do domínio medido
    tf = Transformer.from_crs(4326, 3031, always_xy=True)
    bx, by = tf.transform(*FURO)
    w = pd.read_csv(ROOT / "data" / "validation" / "wild2024_apres_teis.csv", comment="#")
    wx, wy = tf.transform(w.lon.values, w.lat.values)
    for ax, box in ((axa, None), (axb, (-1600, -1500, -515, -395)), (axc, (-1680, -1570, -345, -240))):
        ax.imshow(bg, cmap="gray", vmin=0, vmax=1, extent=ext, interpolation="nearest")
        im = ax.imshow(m, cmap=cmap, norm=norm, extent=ext, interpolation="nearest")
        mark_negatives(ax, xs, ys, lw=0.3)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.5)
        if box:
            ax.set_xlim(box[0], box[1])
            ax.set_ylim(box[2], box[3])
            axa.add_patch(mpl.patches.Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2],
                                                fill=False, lw=0.6, ec="k"))
    ii, jj = np.nonzero(np.isfinite(m))
    axa.set_xlim(xs[jj].min() / 1e3 - 15, xs[jj].max() / 1e3 + 15)
    axa.set_ylim(ys[ii].min() / 1e3 - 15, ys[ii].max() / 1e3 + 15)
    for k, n, (dx, dy) in ((1, "Pine Island", (-60, 30)), (2, "Thwaites", (-62, 0)), (3, "Crosson", (25, 15)),
                           (4, "Dotson", (-75, -25)), (5, "Cosgrove", (-10, 45))):
        ii, jj = np.nonzero((plat == k) & (cls > 0))
        axa.text(np.median(xs[jj]) / 1e3 + dx, np.median(ys[ii]) / 1e3 + dy, n, fontsize=6.2,
                 bbox={"fc": "w", "ec": "none", "pad": 0.6, "alpha": 0.8})
    axb.plot(wx / 1e3, wy / 1e3, "o", ms=2.0, mfc="none", mec="k", mew=0.4, label="ApRES (Wild 2024)")
    axb.plot(bx / 1e3, by / 1e3, "*", ms=6, mfc="#F2C14E", mec="k", mew=0.4,
             label="Furo e ApRES (Davis 2023)")
    axb.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=5.4, handletextpad=0.3)
    for ax, t in ((axa, "a"), (axb, "b"), (axc, "c")):
        tag(ax, t, x=0.0, y=1.01)
    axa.legend(handles=[mpl.patches.Patch(color="0.86", label="Apoiado"),
                        mpl.patches.Patch(color="0.68", label="Não medida"),
                        mpl.patches.Patch(fc="0.95", ec="0.6", lw=0.3,
                                          label="Fora da máscara 2021"),
                        mpl.patches.Patch(fc="none", ec="k", lw=0.3, hatch="////",
                                          label="m_b < 0 que muda com a escala da divergência"),
                        mpl.patches.Patch(fc="none", ec="k", lw=0.8, label="m_b < 0 robusto (não explicado)")],
               loc="center left", fontsize=5.4, handlelength=1.0)
    axb.set_title("Thwaites", fontsize=6.5, pad=2)
    axc.set_title("Pine Island", fontsize=6.5, pad=2)
    # barra de escala de 50 km (painel a) e 10 km (b, c)
    for ax, L in ((axa, 50), (axb, 10), (axc, 10)):
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        xa, ya = x1 - 0.05 * (x1 - x0) - L, y0 + 0.05 * (y1 - y0)
        ax.plot([xa, xa + L], [ya, ya], "k-", lw=1.2)
        ax.text(xa + L / 2, ya + 0.02 * (y1 - y0), f"{L} km", ha="center", va="bottom", fontsize=5.6)
    cb = fig.colorbar(im, ax=[axa, axb, axc], orientation="horizontal", fraction=0.035, pad=0.03,
                      aspect=55, ticks=[-5, -1, 0, 1, 5, 10, 20, 50])
    cb.ax.set_xticklabels(["−5", "−1", "0", "1", "5", "10", "20", "50"])
    cb.set_label("Derretimento basal médio 2019–2025 (m de gelo por ano)")
    cb.outline.set_linewidth(0.4)
    save(fig, "fig1_mapa_derretimento")
    pd.DataFrame({"x_m": np.repeat(xs[None, :], len(ys), 0)[np.isfinite(m)],
                  "y_m": np.repeat(ys[:, None], len(xs), 1)[np.isfinite(m)],
                  "mb_mpa": m[np.isfinite(m)]}).to_csv(SRC / "fig1_mapa.csv", index=False)


# ------------------------------------------------------------------------------------------ Fig. 2
def fig2():
    b = pd.read_parquet(O / "tide_response_ase" / "beta_lagr_todas.parquet")
    params = json.loads(xr.open_dataset(I / "hydrostatic_mask_500m.nc").classe.attrs["parametros"])
    rad = pd.read_csv(O / "mcords_hydro_test" / "perfil_razao_normalizado_por_voo.csv")
    cells = pd.read_parquet(O / "gz_recovery" / "celulas_Thwaites.parquet")
    fig, ax = plt.subplots(1, 3, figsize=(183 * MM, 58 * MM), gridspec_kw={"wspace": 0.38})
    bins = np.arange(-4, 10.01, 0.5)
    rows = []
    for s in ["Pine_Island", "Thwaites", "Crosson", "Dotson", "Cosgrove"]:
        g = b[b.shelf == s]
        prof = g.groupby(pd.cut(g.dist_gl_km, bins), observed=True).beta
        x = np.array([i.mid for i in prof.median().index])
        med = prof.median().to_numpy()
        ax[0].plot(x, med, color=SHELF_COL[s], lw=1.1, label=LABEL[s])
        ds = params[s]["d_star_km"]["f090"]
        ax[0].plot([ds], [np.interp(ds, x, med)], "o", ms=3, color=SHELF_COL[s], mec="w", mew=0.4)
        rows += [{"plataforma": s, "dist_km": xi, "beta_mediano": mi, "n": ni}
                 for xi, mi, ni in zip(x, med, prof.size().to_numpy())]
    ax[0].axvspan(0, 2, color="#CFCECE", alpha=0.45, lw=0)
    ax[0].text(1, 1.17, "recorte fixo\nde 2 km", ha="center", fontsize=5.6, color="#4D4D4D")
    ax[0].axhline(1, color="#767676", lw=0.5, ls="--")
    ax[0].axvline(0, color="k", lw=0.5)
    ax[0].set_xlim(-4, 10)
    ax[0].set_ylim(-0.2, 1.45)
    ax[0].set_xlabel("Distância à linha de aterramento (km)\n← apoiado    flutuante →")
    ax[0].set_ylabel("Resposta à maré, β")
    ax[0].legend(loc="lower right", fontsize=5.4, handlelength=1.4)
    pd.DataFrame(rows).to_csv(SRC / "fig2a_beta.csv", index=False)

    rad = rad.copy()
    lo = rad.dist_gl_km.str.extract(r"\(([-\d.]+),").astype(float)[0]
    hi = rad.dist_gl_km.str.extract(r",\s*([-\d.]+)\]").astype(float)[0]
    rad["mid"] = (lo + hi) / 2
    for s in ["Pine_Island", "Thwaites", "Crosson", "Dotson", "Cosgrove"]:
        g = rad[(rad.plataforma == s) & (rad["mid"] > 0)].sort_values("mid")
        ax[1].plot(g["mid"], g.razao_norm_med, "-o", color=SHELF_COL[s], lw=1.0, ms=2.2)
    ax[1].axhspan(0.97, 1.03, color="#CFCECE", alpha=0.45, lw=0)
    ax[1].axhline(1, color="#767676", lw=0.5, ls="--")
    ax[1].set_xlabel("Distância à linha de aterramento (km)")
    ax[1].set_ylabel("Espessura radar / hidrostática\n(normalizada por voo)")
    ax[1].set_ylim(0.8, 1.1)
    rad.to_csv(SRC / "fig2b_radar.csv", index=False)

    cells = cells[cells.unidade == "Thwaites_TEIS"]
    p = cells.pivot_table(index=["x", "y"], columns="versao", values=["mb", "dist_gl_km"]).dropna()
    p.columns = [f"{a}_{c}" for a, c in p.columns]
    p = p.reset_index()
    tf = Transformer.from_crs(4326, 3031, always_xy=True)
    bx, by = tf.transform(*FURO)
    bins2 = np.arange(0, 12.01, 1.0)
    out = []
    for v, col, lab in (("sem_correcao", "#767676", "Hidrostático"),
                        ("corrigido", "#B64342", "Corrigido (β·maré, radar)")):
        g = p.groupby(pd.cut(p[f"dist_gl_km_{v}"], bins2), observed=True)[f"mb_{v}"]
        x = np.array([i.mid for i in g.median().index])
        ax[2].plot(x, g.median().to_numpy(), "-o", color=col, ms=2.2, lw=1.0, label=lab)
        ax[2].fill_between(x, g.quantile(.25).to_numpy(), g.quantile(.75).to_numpy(), color=col,
                           alpha=0.15, lw=0)
        out += [{"versao": v, "dist_km": xi, "mb_mediana": mi} for xi, mi in zip(x, g.median())]
    ax[2].set_xlim(0, 12)
    ax[2].set_xlabel("Distância à linha de aterramento (km)")
    ax[2].set_ylabel("Derretimento basal, TEIS (m/ano)")
    ax[2].legend(loc="upper right", fontsize=5.4)
    pd.DataFrame(out).to_csv(SRC / "fig2c_correcao.csv", index=False)
    for a, t in zip(ax, "abc"):
        tag(a, t, x=-0.18)
    save(fig, "fig2_zona_flexao")


# ------------------------------------------------------------------------------------------ Fig. 3
def fig3():
    T = pd.read_csv(O / "basal_melt_ase" / "TABELA_FINAL_derretimento_basal.csv").set_index("unidade")
    unc = pd.read_csv(O / "basal_melt_ase" / "incerteza_v2_resumo.csv").set_index("unidade")   # domínio v2
    pao = pd.read_csv(O / "compare_paolo2023" / "comparacao_por_unidade.csv").set_index("unidade")
    meng = {"Pine_Island": (16.28, 1.46), "Thwaites_TEIS": (7.13, 1.25), "Crosson": (7.79, 1.65),
            "Dotson": (7.15, 1.45), "Cosgrove": (1.22, 1.02)}
    fig = plt.figure(figsize=(183 * MM, 62 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.1, 0.9], wspace=0.45)
    a, b, c = (fig.add_subplot(gs[0, i]) for i in range(3))
    y = np.arange(len(UNITS))[::-1]
    rows = []
    for yi, u in zip(y, UNITS):
        m, st, tot = T.loc[u, "mb_mpa"], T.loc[u, "sigma_estat"], T.loc[u, "sigma_total"]
        a.errorbar(m, yi + 0.18, xerr=tot, fmt="o", ms=3.2, color=SHELF_COL[u], elinewidth=0.8, capsize=0)
        a.errorbar(m, yi + 0.18, xerr=st, fmt="none", color="k", elinewidth=2.0, capsize=0)
        a.errorbar(meng[u][0], yi, xerr=meng[u][1], fmt="s", ms=2.6, mfc="w", mec="k", ecolor="k",
                   elinewidth=0.6, mew=0.5)
        a.plot(pao.loc[u, "paolo_media_mpa"], yi - 0.18, "^", ms=3, mfc="#CFCECE", mec="k", mew=0.5)
        rows.append({"unidade": u, "este_trabalho": m, "sigma_estat": st, "sigma_total": tot,
                     "meng": meng[u][0], "meng_sigma": meng[u][1],
                     "paolo_1992_2017_mesmos_pixels": pao.loc[u, "paolo_media_mpa"]})
    a.set_yticks(y)
    a.set_yticklabels([LABEL[u] for u in UNITS])
    a.set_xlabel("Derretimento basal médio (m/ano)")
    a.set_xlim(-1, 24)
    h = [mpl.lines.Line2D([], [], marker="o", color="#4D4D4D", ls="", ms=3.2),
         mpl.lines.Line2D([], [], marker="s", mfc="w", mec="k", ls="", ms=2.6),
         mpl.lines.Line2D([], [], marker="^", mfc="#CFCECE", mec="k", ls="", ms=3)]
    a.legend(h, ["Este trabalho, 2019–2025 (σ total; barra preta: estatístico)",
                 "Meng et al. 2025, 2019–2023", "Paolo et al. 2023, 1992–2017"],
             loc="upper center", bbox_to_anchor=(0.5, 1.22), fontsize=5.2, handletextpad=0.2, ncol=1)
    pd.DataFrame(rows).to_csv(SRC / "fig3a_comparacao.csv", index=False)

    facs = [("efeito_firn", "Firn (GEMB × IMAU)", "#0F4D92"),
            ("efeito_escala_km", "Escala da divergência", "#42949E"),
            ("efeito_mascara", "Máscara (d*)", "#9A4D8E"),
            ("agregacao", "Agregação", "#B64342"),
            ("velocidade", "Velocidade (ITS_LIVE × MEaSUREs)", "#8BCF8B"),
            ("efeito_smb", "SMB (GEMB × RACMO)", "#CFCECE"),
            ("efeito_thwaites", "Thwaites: correção × corte", "#F2C14E"),
            ("cortes", "Frente, perda de gelo, radar", "#767676")]
    left = np.zeros(len(UNITS))
    rows = []
    for key, lab, col in facs:
        vals = []
        for u in UNITS:
            if key == "velocidade":
                v = abs(unc.loc[u, "velocidade"])
            elif key == "cortes":
                v = unc.loc[u, ["efeito_frente_km", "efeito_perda_max", "efeito_obs_km",
                                "efeito_radar_tol"]].max()
            else:
                v = unc.loc[u, key]
            vals.append(0.0 if pd.isna(v) else float(v))
        vals = np.array(vals)
        b.barh(y, vals, left=left, color=col, height=0.62, label=lab, lw=0)
        left += vals
        rows += [{"unidade": u, "fator": lab, "efeito_mpa": v} for u, v in zip(UNITS, vals)]
    b.set_yticks(y)
    b.set_yticklabels([LABEL[u].replace(" (TEIS)", "\n(TEIS)") for u in UNITS], fontsize=5.6)
    b.set_xlabel("Efeito sobre o derretimento médio (m/ano)")
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=5.0, handlelength=0.9,
             handletextpad=0.3, ncol=2, columnspacing=0.8)
    pd.DataFrame(rows).to_csv(SRC / "fig3b_incerteza.csv", index=False)

    ap = pd.read_csv(O / "validation" / "apres_pontos.csv")
    ap = ap[ap.ponto != "G2"]
    xs = [0, 1, 2]
    c.add_patch(mpl.patches.Rectangle((-0.4, 2.0), 0.8, 3.4, color="#F2C14E", alpha=0.6, lw=0))
    c.errorbar(np.zeros(len(ap)) + np.linspace(-0.15, 0.15, len(ap)), ap.melt_mpa,
               yerr=[ap.melt_mpa - ap.ic95_lo, ap.ic95_hi - ap.melt_mpa], fmt="o", ms=2.2,
               color="k", elinewidth=0.5)
    rec = xr.open_dataset(I / "basal_melt" / "thwaites_recuperada_500m.nc")
    comp = xr.open_dataset(I / "basal_melt" / "componentes_uniao_500m.nc")
    raw = comp.SMB.values - comp.DHDt.values - comp.H_gemb.values * comp.div_2km.values
    tf = Transformer.from_crs(4326, 3031, always_xy=True)
    bx, by = tf.transform(*FURO)
    XX, YY = np.meshgrid(comp.x.values, comp.y.values)
    near = np.hypot(XX - bx, YY - by) <= 3e3
    v_raw = raw[near & np.isfinite(rec.mb_corrigido.values)]
    v_cor = rec.mb_corrigido.values[near & np.isfinite(rec.mb_corrigido.values)]
    for xi, v, col in ((1, v_raw, "#767676"), (2, v_cor, "#B64342")):
        c.boxplot([v], positions=[xi], widths=0.45, showfliers=False, patch_artist=True,
                  boxprops={"facecolor": col, "alpha": 0.35, "lw": 0.5}, medianprops={"color": col, "lw": 1.2},
                  whiskerprops={"lw": 0.5}, capprops={"lw": 0.5})
    c.set_xticks(xs)
    c.set_xticklabels(["ApRES\n2020", "Produto\nhidrostático", "Produto\ncorrigido"])
    c.set_ylabel("Derretimento basal (m/ano)")
    c.set_title("TEIS, ≤ 3 km do furo", fontsize=6, pad=3)
    pd.DataFrame({"grupo": ["apres"] * len(ap) + ["hidrostatico"] * len(v_raw) + ["corrigido"] * len(v_cor),
                  "mb_mpa": np.r_[ap.melt_mpa, v_raw, v_cor]}).to_csv(SRC / "fig3c_apres.csv", index=False)
    for ax, t in ((a, "a"), (b, "b"), (c, "c")):
        tag(ax, t, x={"a": -0.32, "b": -0.22, "c": -0.28}[t], y=1.06 if t == "a" else 1.02)
    save(fig, "fig3_comparacao_incerteza")


def main() -> int:
    SRC.mkdir(parents=True, exist_ok=True)
    fig1()
    fig2()
    fig3()
    print("figuras em", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
