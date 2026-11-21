"""Envelope de incerteza do derretimento basal por unidade e σ por pixel (passo 8 do plano basal).

Valor central: média por área do mapa contínuo (``mb_continuo_por_unidade.csv``).

Incerteza **estatística**: erro da tendência Lagrangiana da série de espessura (regressão ponderada
pelo jackknife por grupos de passagens). As passagens são a unidade independente.

Incerteza **estrutural**: cada fator é uma alternativa defensável; o efeito é a maior diferença
absoluta em m_b em relação à escolha principal:
1. firn: IMAU × GEMB em 2019–2023 (tendência DH/Dt e H·∇·u);
2. SMB: RACMO × GEMB em 2019–2022;
3. máscara: variantes f = 0,8, f = 0,95 e IC × f = 0,9 (domínio e série de espessura refeitos);
4. escala de ∇·u: 1, 2, 3 e 5 km × ótima por CV;
5. agregação: mapa contínuo × só trilhas × série ponderada.
Os fatores são tratados como independentes: σ_estr = √Σ Δ_i². É uma convenção declarada;
os fatores podem estar correlacionados (por exemplo, máscara e agregação).

σ por pixel: √(σ_DH/Dt interpolado² + σ_H·∇·u²). Nas unidades interpoladas por kernel, o σ do
kernel só leva o ruído; soma-se o erro de interpolação medido na CV,
√max(RMSE_CV² − mediana(σ_obs²), 0), nos pixels sem observação própria.

Saídas: ``outputs/basal_melt_ase/envelope_por_unidade.csv``; ``se_mb`` gravado no NetCDF contínuo.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
O = ROOT / "outputs"
UNITS = ["Pine_Island", "Thwaites_TEIS", "Thwaites_TWGT", "Crosson", "Dotson", "Cosgrove"]
VAR_OF_CLASS = {"classe_f080": "f080", "classe_f095": "f095", "classe_ic": "ic"}


def trend(g, col, secol=None, period=None):
    if period:
        g = g[(g.ano >= period[0]) & (g.ano <= period[1])]
    y = g[col].to_numpy()
    ok = np.isfinite(y)
    t = g.t_meio.to_numpy()[ok]
    if secol:
        w = 1 / np.maximum(g[secol].to_numpy()[ok], 1e-3)
        p, c = np.polyfit(t, y[ok], 1, w=w, cov=True)
        return p[0], float(np.sqrt(c[0, 0]))
    return np.polyfit(t, y[ok], 1)[0], np.nan


def base_of(u):
    return u.replace("_TEIS", "").replace("_TWGT", "")


def main() -> int:
    cont = pd.read_csv(O / "basal_melt_ase" / "mb_continuo_por_unidade.csv").set_index("unidade")
    tr = pd.read_csv(O / "basal_melt_ase" / "mb_por_unidade.csv").set_index("unidade")
    smb = pd.read_csv(O / "basal_melt_ase" / "smb_gemb_racmo.csv").set_index("unidade")
    hd = pd.read_csv(O / "flux_term_ase" / "H_div_por_unidade.csv")
    best = dict(pd.read_csv(O / "divergence_ase" / "escala_otima.csv").values)
    ser0 = pd.read_csv(O / "thickness_ase" / "serie_espessura_classe.csv")

    # H·∇·u e SMB médios em cada variante de máscara (mesma grade e mesmo H)
    m = xr.open_dataset(ROOT / "data" / "interim" / "hydrostatic_mask_500m.nc")
    fl = xr.open_dataset(ROOT / "data" / "interim" / "flux_term" / "H_div_500m.nc")
    mb_nc = xr.open_dataset(ROOT / "data" / "interim" / "basal_melt" / "mb_continuo_500m.nc")
    unit_map = mb_nc.unidade.values
    smb_map = mb_nc.SMB.values

    rows = []
    for u in UNITS:
        b = base_of(u)
        sc = int(best[b])
        g0 = ser0[ser0.plataforma == u]
        dh0, se_stat = trend(g0, "dH_lagr_m", "se_lagr_m")
        hrow = hd[(hd.unidade == u) & (hd.sigma_km == sc)].iloc[0]
        center = cont.loc[u, "mb_media_area_mpa"]

        # 1. firn
        dg, _ = trend(g0, "dH_lagr_m", period=(2019, 2023))
        di, _ = trend(g0, "dH_lagr_imau_m", period=(2019, 2023))
        d_firn = -(di - dg) - (hrow.H_div_imau_mpa - hrow.H_div_gemb_mpa)
        # 2. SMB
        d_smb = smb.loc[u, "diff"]
        # 3. máscara
        d_mask = {}
        for cvar, tag in VAR_OF_CLASS.items():
            f = O / "thickness_ase" / f"serie_espessura_{cvar}.csv"
            if not f.is_file():
                continue
            gv = pd.read_csv(f)
            gv = gv[gv.plataforma == u]
            dhv, _ = trend(gv, "dH_lagr_m")
            # o rótulo de unidade do mapa já separa TEIS/TWGT; a variante só muda o domínio
            dom_v = (m[cvar].values == 1)
            # pixels da unidade na variante: mesma plataforma; TEIS/TWGT pelo rótulo do mapa base,
            # e pixels novos da variante herdam o rótulo do vizinho mais próximo da mesma plataforma
            plat_id = {"Pine_Island": 1, "Thwaites": 2, "Crosson": 3, "Dotson": 4, "Cosgrove": 5}[b]
            if b == "Thwaites":
                from scipy.spatial import cKDTree
                XX, YY = np.meshgrid(m.x.values, m.y.values)
                lab = unit_map.copy()
                need = dom_v & (m.plataforma.values == plat_id) & (lab == "")
                src = np.isin(lab, ["Thwaites_TEIS", "Thwaites_TWGT"])
                if need.any():
                    t = cKDTree(np.c_[XX[src], YY[src]])
                    _, k = t.query(np.c_[XX[need], YY[need]])
                    lab[need] = lab[src][k]
                uni_v = dom_v & (lab == u)
            else:
                uni_v = dom_v & (m.plataforma.values == plat_id)
            hdv = fl[f"H_div_gemb_{sc}km"].values
            okv = uni_v & np.isfinite(hdv)
            smb_v = np.nanmean(np.where(uni_v, smb_map, np.nan)) if np.isfinite(
                np.where(uni_v, smb_map, np.nan)).any() else smb.loc[u, "SMB_GEMB_2019_22"]
            mb_v = smb_v - dhv - float(np.mean(hdv[okv]))
            ok0 = (unit_map == u) & np.isfinite(hdv)
            mb_0 = float(np.nanmean(smb_map[unit_map == u])) - dh0 - float(np.mean(hdv[ok0]))
            d_mask[tag] = mb_v - mb_0
        # 4. escala
        hs = hd[hd.unidade == u].set_index("sigma_km")["H_div_gemb_mpa"]
        d_scale = {f"{int(k)}km": -(v - hs[sc]) for k, v in hs.items() if int(k) != sc}
        # 5. agregação
        alts = {"trilhas": tr.loc[u, "mb_media_area_mpa"], "serie": tr.loc[u, "mb_com_DHDt_da_serie_mpa"]}
        d_agg = {k: v - center for k, v in alts.items()}

        def mx(dct):
            return max((abs(v) for v in dct.values()), default=0.0)

        comps = {"firn": abs(d_firn), "smb": abs(d_smb), "mascara": mx(d_mask),
                 "escala_div": mx(d_scale), "agregacao": mx(d_agg)}
        s_struct = float(np.sqrt(sum(v ** 2 for v in comps.values())))
        rows.append({"unidade": u, "mb_central_mpa": center, "sigma_estat_mpa": se_stat,
                     **{f"d_{k}_mpa": v for k, v in comps.items()},
                     "sigma_estrutural_mpa": s_struct,
                     "sigma_total_mpa": float(np.sqrt(se_stat ** 2 + s_struct ** 2)),
                     "fator_dominante": max(comps, key=comps.get),
                     "detalhe_mascara": {k: round(v, 2) for k, v in d_mask.items()},
                     "detalhe_escala": {k: round(v, 2) for k, v in d_scale.items()},
                     "detalhe_agregacao": {k: round(v, 2) for k, v in d_agg.items()}})
    t = pd.DataFrame(rows)
    t.to_csv(O / "basal_melt_ase" / "envelope_por_unidade.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 60)
    print(t.drop(columns=["detalhe_mascara", "detalhe_escala", "detalhe_agregacao"])
          .round(2).to_string(index=False))
    print(t[["unidade", "detalhe_mascara", "detalhe_escala", "detalhe_agregacao"]].to_string(index=False))

    # σ por pixel
    cv = pd.read_csv(O / "basal_melt_ase" / "cv_interpolacao.csv")
    src = xr.open_dataset(ROOT / "data" / "interim" / "basal_melt" / "mb_500m.nc")
    diag = pd.read_csv(O / "divergence_ase" / "escala_diagnostico.csv")
    meth = ast.literal_eval(mb_nc.attrs["metodo_por_unidade"])
    se_dh = mb_nc.se_DHDt_continuo.values.copy()
    H = fl["H_gemb"].values
    sd_hdiv = np.full(se_dh.shape, np.nan)
    for u in UNITS:
        sel = unit_map == u
        b = base_of(u)
        noise = diag[(diag.plataforma == b) & (diag.sigma_km == int(best[b]))]["ruido_media7a_1pa"]
        sd_hdiv[sel] = H[sel] * float(noise.iloc[0])
        if meth[u].startswith("kernel"):
            r = cv[(cv.unidade == u) & (cv.metodo == meth[u])]["rmse_mpa"].iloc[0]
            s_obs = src.se_DHDt.values[sel]
            extra = np.sqrt(max(r ** 2 - np.nanmedian(s_obs ** 2), 0))
            far = sel & (mb_nc.dist_obs_m.values > 0)
            se_dh[far] = np.sqrt(se_dh[far] ** 2 + extra ** 2)
            print(f"{u}: σ de interpolação somado ao kernel = {extra:.2f} m/a")
    se_mb = np.sqrt(se_dh ** 2 + sd_hdiv ** 2)
    out = mb_nc.load()
    out["se_mb"] = (("y", "x"), se_mb.astype(np.float32))
    out["se_mb"].attrs = {"descricao": "σ por pixel: DH/Dt interpolado + ruído de H·∇·u; "
                                       "sem o envelope estrutural (ver envelope_por_unidade.csv)"}
    mb_nc.close()
    out.to_netcdf(ROOT / "data" / "interim" / "basal_melt" / "mb_continuo_500m.nc")
    for u in UNITS:
        sel = (unit_map == u) & np.isfinite(se_mb)
        print(f"{u:14s} σ por pixel: mediana {np.median(se_mb[sel]):.1f} m/a")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
