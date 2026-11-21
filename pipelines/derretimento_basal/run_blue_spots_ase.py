"""Manchas azuis (m_b < 0, "recongelamento") no produto central v2: reais ou artefato?

Para cada mancha (componente conexa, 8-vizinhos, de pixels do domínio com m_b < LIMIAR, área
≥ AREA_MIN km²) mede-se:

1. **Observação**: fração de pixels com célula observada própria (dist_obs = 0) e distância média
   à célula observada → a mancha está onde há dado ou foi criada pela krigagem?
2. **Termo que a produz**: médias de SMB, −DH/Dt e −H·∇·u na mancha × média da unidade → o
   negativo vem do afinamento (DH/Dt) ou da divergência?
3. **Escala da divergência**: m_b recalculado com ∇·u em 1, 2, 3 e 5 km → se some ao suavizar,
   é ruído da derivada do campo de velocidade (Dotson, Crosson e Cosgrove usam 1 km, a escala mais
   ruidosa).
4. **Significância**: fração de pixels com m_b + 2σ < 0, σ = erro de krigagem do DH/Dt
   (``se_DHDt`` dos componentes; aproximação: o σ do H·∇·u não entra).
5. **Geometria**: distância ao gelo apoiado mais próximo (BedMachine máscara 2, inclui ilhas e
   pontos de ancoragem) — perto de ancoragem a espessura hidrostática falha.
6. **Referência independente**: Paolo et al. (2023, NSIDC-0792, média 1992–2017, outra época e
   outro método) nos mesmos pixels → recongelamento persistente apareceria lá também.

Saídas: ``outputs/blue_spots_ase/manchas.csv``, ``resumo_por_unidade.csv``, ``manchas_mapa.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import SymLogNorm
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
I = ROOT / "data" / "interim" / "basal_melt"
OUT = ROOT / "outputs" / "blue_spots_ase"
BM = ROOT / "data" / "NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc"
PAOLO = ROOT / "data" / "validation" / "paolo2023_ase_subset.nc"
UNITS = ["Pine_Island", "Thwaites_TEIS", "Crosson", "Dotson", "Cosgrove"]
OPT = {"Pine_Island": 3, "Thwaites_TEIS": 2, "Thwaites_TWGT": 2, "Crosson": 1, "Dotson": 1, "Cosgrove": 1}
LIMIAR = 0.0
AREA_MIN = 2.0          # km² (8 pixels de 500 m)
PIX_KM2 = 0.25


def negative_flags():
    """Máscaras do produto v2: m_b < 0 com sinal NÃO robusto à escala de ∇·u (algum nível de 1, 2,
    3 ou 5 km dá m_b ≥ 0) e m_b < 0 robusto (negativo nas quatro escalas). Grade do produto."""
    v2 = xr.open_dataset(I / "mb_final_v2_500m.nc")
    c = xr.open_dataset(I / "componentes_uniao_500m.nc")
    m, dom, unit = v2.mb.values, v2.dominio.values == 1, v2.unidade.values.astype(str)
    delta = np.nan_to_num(xr.open_dataset(I / "thwaites_recuperada_500m.nc").delta_correcao.values)
    corr = np.where(np.char.startswith(unit, "Thwaites"), delta, 0)
    neg_all = np.ones(m.shape, bool)
    for s in (1, 2, 3, 5):
        neg_all &= (c.SMB.values - v2.DHDt.values - v2.H.values * c[f"div_{s}km"].values + corr) < 0
    neg = dom & (m < 0)
    return neg & ~neg_all, neg & neg_all


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    v2 = xr.open_dataset(I / "mb_final_v2_500m.nc")
    c = xr.open_dataset(I / "componentes_uniao_500m.nc")
    xs, ys = v2.x.values, v2.y.values
    m, dom, unit = v2.mb.values, v2.dominio.values == 1, v2.unidade.values
    dh, H, obs = v2.DHDt.values, v2.H.values, v2.dist_obs_m.values / 1e3
    smb, se = c.SMB.values, c.se_DHDt.values
    delta = np.nan_to_num(xr.open_dataset(I / "thwaites_recuperada_500m.nc").delta_correcao.values)
    divs = {s: c[f"div_{s}km"].values for s in (1, 2, 3, 5)}
    # m_b com cada escala de divergência (mesmo domínio)
    mb_sc = {s: smb - dh - H * d + np.where(np.char.startswith(unit.astype(str), "Thwaites"), delta, 0)
             for s, d in divs.items()}
    hdiv_opt = np.full(m.shape, np.nan)
    for u, s in OPT.items():
        hdiv_opt[unit == u] = (H * divs[s])[unit == u]
    # gelo apoiado (BedMachine 2) → distância
    bm = xr.open_dataset(BM)["mask"].sel(x=slice(xs.min(), xs.max()), y=slice(ys.max(), ys.min())).values
    assert bm.shape == m.shape, (bm.shape, m.shape)
    d_ground = ndimage.distance_transform_edt(bm != 2) * 0.5
    # Paolo nos pixels (vizinho mais próximo na grade de 1.920 m; sinal: derretimento positivo)
    p = xr.open_dataset(PAOLO)
    pm = (-p.melt_mean).sel(x=xr.DataArray(xs, dims="x"), y=xr.DataArray(ys, dims="y"), method="nearest").values
    pm = pm.T if pm.shape != m.shape else pm

    blue = dom & (m < LIMIAR)
    lab, n = ndimage.label(blue, structure=np.ones((3, 3)))
    rows = []
    for k in range(1, n + 1):
        s = lab == k
        a = s.sum() * PIX_KM2
        if a < AREA_MIN:
            continue
        u = pd.Series(unit[s]).mode().iloc[0]
        uu = dom & (unit == u)
        ii, jj = np.nonzero(s)
        rows.append({
            "mancha": k, "unidade": u, "area_km2": a, "x_km": xs[jj].mean() / 1e3, "y_km": ys[ii].mean() / 1e3,
            "mb_mpa": float(np.mean(m[s])), "mb_unidade_mpa": float(np.mean(m[uu])),
            "frac_pix_observados": float(np.mean(obs[s] == 0)), "frac_obs_unidade": float(np.mean(obs[uu] == 0)),
            "dist_obs_media_km": float(np.mean(obs[s])),
            "SMB": float(np.nanmean(smb[s])), "menos_DHDt": float(np.nanmean(-dh[s])),
            "menos_Hdiv": float(np.nanmean(-hdiv_opt[s])),
            "menos_DHDt_unidade": float(np.nanmean(-dh[uu])), "menos_Hdiv_unidade": float(np.nanmean(-hdiv_opt[uu])),
            **{f"mb_div{sc}km": float(np.nanmean(mb_sc[sc][s])) for sc in (1, 2, 3, 5)},
            **{f"frac_neg_div{sc}km": float(np.nanmean(mb_sc[sc][s] < 0)) for sc in (1, 2, 3, 5)},
            "frac_signif_2sigma": float(np.nanmean((m[s] + 2 * se[s]) < 0)),
            "dist_apoiado_km": float(np.mean(d_ground[s])), "dist_apoiado_min_km": float(np.min(d_ground[s])),
            "paolo_mpa": float(np.nanmean(pm[s])), "paolo_frac_neg": float(np.nanmean(pm[s] < 0)),
            "paolo_unidade_mpa": float(np.nanmean(pm[uu])),
        })
    t = pd.DataFrame(rows).sort_values("area_km2", ascending=False)
    t.to_csv(OUT / "manchas.csv", index=False)

    summ = []
    for u in UNITS:
        uu = dom & (unit == u)
        b = uu & (m < 0)
        summ.append({"unidade": u, "area_km2": uu.sum() * PIX_KM2, "frac_area_negativa": b.sum() / uu.sum(),
                     **{f"frac_neg_div{sc}km": float(np.mean(mb_sc[sc][uu] < 0)) for sc in (1, 2, 3, 5)},
                     "frac_neg_em_pix_observados": float(np.mean(m[uu & (obs == 0)] < 0)),
                     "frac_neg_em_pix_interpolados": float(np.mean(m[uu & (obs > 0)] < 0)),
                     "frac_neg_signif_2sigma": float(np.nanmean((m[uu] + 2 * se[uu]) < 0)),
                     "paolo_frac_neg": float(np.nanmean(pm[uu] < 0)),
                     "paolo_frac_neg_nas_manchas": float(np.nanmean(pm[b] < 0)),
                     "mb_media": float(np.mean(m[uu])), "mb_media_sem_negativos_zerados": float(np.mean(np.maximum(m[uu], 0)))})
    sm = pd.DataFrame(summ)
    sm.to_csv(OUT / "resumo_por_unidade.csv", index=False)

    # figura diagnóstica: m_b (1 km × 3 km de divergência) e Paolo, nas três regiões com manchas
    boxes = {"Dotson": (-1600, -1510, -680, -610), "Cosgrove": (-1810, -1730, -380, -290),
             "Pine Island": (-1665, -1565, -340, -260)}
    norm = SymLogNorm(linthresh=1.0, linscale=0.6, vmin=-5, vmax=60, base=10)
    ext = [xs.min() / 1e3, xs.max() / 1e3, ys.min() / 1e3, ys.max() / 1e3]
    fig, axes = plt.subplots(3, 3, figsize=(12, 11), constrained_layout=True)
    for r, (name, b) in enumerate(boxes.items()):
        for col, (title, f) in enumerate((("produto (∇·u na escala ótima)", np.where(dom, m, np.nan)),
                                          ("∇·u em 3 km", np.where(dom, mb_sc[3], np.nan)),
                                          ("Paolo 2023 (1992–2017)", np.where(dom, pm, np.nan)))):
            ax = axes[r, col]
            im = ax.imshow(f, cmap="RdBu_r", norm=norm, extent=ext, interpolation="nearest")
            ax.contour(xs / 1e3, ys / 1e3, (lab > 0) & np.isin(lab, t.mancha.values), levels=[0.5],
                       colors="k", linewidths=0.6)
            ax.contour(xs / 1e3, ys / 1e3, (obs == 0).astype(float), levels=[0.5], colors="0.4",
                       linewidths=0.3, linestyles=":")
            ax.set_xlim(b[0], b[1])
            ax.set_ylim(b[2], b[3])
            ax.set_aspect("equal")
            ax.set_title(f"{name}: {title}", fontsize=9)
            ax.tick_params(labelsize=7)
    cb = fig.colorbar(im, ax=axes, shrink=0.6, ticks=[-5, -1, 0, 1, 5, 10, 20, 50])
    cb.set_label("m_b (m/ano)")
    fig.suptitle("Contorno preto: manchas m_b < 0 (≥ 2 km²); pontilhado: limite das células observadas", fontsize=9)
    fig.savefig(OUT / "manchas_mapa.png", dpi=170)
    pd.set_option("display.width", 300)
    print(sm.round(3).to_string(index=False))
    cols = ["unidade", "area_km2", "mb_mpa", "frac_pix_observados", "dist_obs_media_km", "menos_DHDt",
            "menos_DHDt_unidade", "menos_Hdiv", "menos_Hdiv_unidade", "mb_div1km", "mb_div3km", "mb_div5km",
            "frac_signif_2sigma", "dist_apoiado_km", "paolo_mpa", "paolo_frac_neg"]
    print(t[cols].head(25).round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
