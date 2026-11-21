"""Produto com a faixa da frente tratada pela frente da PRÓPRIA data (em vez da união de frentes).

Antes: pixel a ≤ 1,5 km de qualquer frente IceLines de 2019–2024,7 saía do produto. Onde a frente
recuou vários km (PIG, desprendimentos de 2020–2021), a união das posições virou uma faixa de até
~10 km, e o produto perdia área que Meng et al. (2025) mede (PIG: 445 km²).

Agora, por segmento:
1. frente do IceLines mais próxima no tempo (mensal; após 2024,7 — fim do IceLines — vale a
   última frente); o segmento entra só se estiver a > 1,5 km dela;
2. presença de gelo relativa ao estado com gelo do pixel: h ≥ 0,5 × percentil 75 da altura sobre o
   mar no pixel (a mediana falhava em pixels que viraram água depois de um desprendimento) e
   percentil 75 ≥ 5 m.
Depois, a mesma cadeia do produto: H (firn GEMB), tendência por célula material (≥ 8 passagens,
≥ 3 anos), krigagem por unidade (kernel de 8 km na Cosgrove), H·∇·u na escala ótima, SMB.

Domínio do produto central (f = 0,9): classe hidrostática; distância às linhas de 2019–2025 ≥ D_REC;
corte do radar (tolerância de 3%) fora da Thwaites; na Thwaites, a correção β·maré + radar de
``run_thwaites_recovered.py`` (Δ por pixel) em vez do corte; e, na faixa que antes saía pela
frente, só pixels com célula medida a ≤ 1 km. Uma célula que existiu só parte do período tem a
tendência no período em que era gelo (declarado).

Saída: ``data/interim/basal_melt/mb_final_v2_500m.nc``, ``outputs/basal_melt_ase/mb_final_v2_por_unidade.csv``.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipelines" / "derretimento_basal"))
sys.path.insert(0, str(ROOT))
from run_basal_melt_ase import cell_trends  # noqa: E402
from run_flux_term_ase import fill_gaps, grid_median  # noqa: E402
from run_mb_continuous_ase import fit_variogram, kernel_predict, krig_predict  # noqa: E402
from run_tide_response_lagrangian import advect_to_ref, load_velocity  # noqa: E402

THK = ROOT / "data" / "interim" / "thickness"
COMP = ROOT / "data" / "interim" / "basal_melt" / "componentes_uniao_500m.nc"
THR = ROOT / "data" / "interim" / "basal_melt" / "thwaites_recuperada_500m.nc"
FRONTS = ROOT / "data" / "calving_fronts_densified_500m.parquet"
OUT_NC = ROOT / "data" / "interim" / "basal_melt" / "mb_final_v2_500m.nc"
OUT = ROOT / "outputs" / "basal_melt_ase"
NAMES = {1: "Pine_Island", 2: "Thwaites", 3: "Crosson", 4: "Dotson", 5: "Cosgrove"}
FRONT_OF = {"Pine_Island": ["PineIsland"], "Thwaites": ["Thwaites1", "Thwaites2"],
            "Crosson": ["Crosson"], "Dotson": ["Dotson"], "Cosgrove": ["Cosgrove"]}
UNITS = ["Pine_Island", "Thwaites_TEIS", "Thwaites_TWGT", "Crosson", "Dotson", "Cosgrove"]
D_FRONT = 1.5
D_REC = {"Pine_Island": 3.0, "Thwaites_TEIS": 1.5, "Thwaites_TWGT": 1.5, "Crosson": 3.0,
         "Dotson": 2.5, "Cosgrove": 2.0}
D_RADAR = {"Pine_Island": 2.0, "Crosson": 0.0, "Dotson": 0.5, "Cosgrove": 1.5}   # tolerância 3%
OPT_SCALE = {"Pine_Island": 3, "Thwaites_TEIS": 2, "Thwaites_TWGT": 2, "Crosson": 1, "Dotson": 1,
             "Cosgrove": 1}
RHO_I, PIX = 917.0, 0.25e6


def epoch_front_distance(x, y, t, fr):
    """Distância (km) de cada segmento à frente da época mais próxima no tempo."""
    ep = np.sort(fr.epoch_year.unique())
    k = np.clip(np.searchsorted(ep, t), 1, len(ep) - 1)
    near = np.where(np.abs(ep[k] - t) < np.abs(ep[k - 1] - t), ep[k], ep[k - 1])
    near = np.where(t > ep[-1], ep[-1], near)
    out = np.full(len(x), np.nan)
    for e in np.unique(near):
        s = near == e
        f = fr[fr.epoch_year == e]
        out[s] = cKDTree(f[["x", "y"]].to_numpy()).query(np.c_[x[s], y[s]])[0] / 1e3
    return out


def to_pix(x, y, xs, ys):
    j = np.floor((x - xs[0]) / (xs[1] - xs[0]) + 0.5).astype(np.int64)
    i = np.floor((y - ys[0]) / (ys[1] - ys[0]) + 0.5).astype(np.int64)
    ok = (i >= 0) & (i < len(ys)) & (j >= 0) & (j < len(xs))
    return i, j, ok


OPT_SCALE_V3 = {"Pine_Island": 3, "Thwaites_TEIS": 2, "Thwaites_TWGT": 2, "Crosson": 3, "Dotson": 3,
                "Cosgrove": 3}
ILHA_AREA_MAX_KM2, ILHA_LA_KM, ILHA_ANEL_KM = 30.0, 2.0, 4.0
PLAT_FRENTE_REF = (1,)   # só a PIG desprendeu e voltou a avançar em 2019–25 (frentes IceLines)
BM_NC = ROOT / "data" / "NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc"


def ajustes_v3(c, xs, ys, union, unit, plat):
    """Mudanças da v3 sobre o domínio (2026-11-18). Devolve (union, unit, plat, classe, drec, dbm, info).

    1. Ilhas do BedMachine sem suporte atual: componente apoiado isolado (cercado de flutuante/oceano,
       < ILHA_AREA_MAX_KM2) e sem linha de aterramento de 2018+ (NSIDC-0498, ESA CCI, S1 2025) a
       ≤ ILHA_LA_KM → tratada como flutuante: a ilha e o
       anel de flexão que ela gera (classe 2/3 a ≤ ILHA_ANEL_KM) passam a classe 1, e a distância à LA do
       BedMachine (corte do radar) é refeita sem ela. Caso medido: PIG (−1597, −277 km), ponto só visto
       pelo ERS em 1996, a 5–6,5 km da LA recente mais próxima. Das 48 ilhas pequenas dentro das
       plataformas, 42 têm LA recente a ≤ 2 km (pinos reais). A cavidade do BedMachine NÃO serve de
       critério: sobre pixels que ele marca apoiados, o leito é forçado a encostar no gelo (cavidade ≈ 0).
    2. Frente na data de referência (T_REF = 2022,0): o produto é lagrangiano, mas a união das classes é
       euleriana; gelo que re-avançou após o desprendimento de 2020 da PIG (presente em 2021–25, seguido
       desde montante) ficava fora. Só nas plataformas de PLAT_FRENTE_REF: na Thwaites o gelo perdido não
       voltou (é mélange/água, onde a espessura hidrostática não vale). Pixels flutuantes no BedMachine com classe 0 ('fora') em todas as
       variantes, ligados à união, entram; a entrada de cada segmento continua exigindo frente da
       própria data a > D_FRONT e presença de gelo (p75), e o domínio final exige célula medida a ≤ 1 km.
    """
    from scipy import ndimage
    from thwaites.qc.grounded_mask import load_recent_gl_lines
    from thwaites.qc.grounding_zone import _coordinates
    bm = xr.open_dataset(BM_NC)
    sel = dict(x=xr.DataArray(xs, dims="x"), y=xr.DataArray(ys, dims="y"))
    m = bm["mask"].sel(**sel, method="nearest").values
    lines = load_recent_gl_lines(ROOT, 2018)
    lxy = np.vstack([_coordinates(g, 250.0) for g in lines.geometry if g is not None])
    tl = cKDTree(lxy)
    XX, YY = np.meshgrid(xs, ys)
    classe = c.classe.values.copy()
    info = {"ilhas": []}
    lab, n = ndimage.label(m == 2)
    borda = ndimage.binary_dilation(np.isin(m, [0, 1, 4]))   # toca oceano/rocha → não é ilha no gelo
    ilha_px = np.zeros(m.shape, bool)
    for k in range(1, n + 1):
        s = lab == k
        a = s.sum() * 0.25
        if a > ILHA_AREA_MAX_KM2 or (s & borda).any():
            continue
        d_la = tl.query(np.c_[XX[s], YY[s]])[0].min() / 1e3
        if d_la > ILHA_LA_KM:
            ilha_px |= s
            info["ilhas"].append({"x_km": float(XX[s].mean() / 1e3), "y_km": float(YY[s].mean() / 1e3),
                                  "area_km2": float(a), "dist_LA_recente_km": float(d_la)})
    if ilha_px.any():
        d_ilha = ndimage.distance_transform_edt(~ilha_px) * 0.5
        anel = (d_ilha <= ILHA_ANEL_KM) & np.isin(classe, [2, 3]) & (m == 3)
        fix = ilha_px | anel
        classe[fix] = 1
        union = union | fix
        gr = (m == 2) & ~ilha_px
        dbm_new = ndimage.distance_transform_edt(~gr) * 0.5
        dbm = np.where(d_ilha <= ILHA_ANEL_KM + 6, dbm_new, c.dist_LA_BedMachine_km.values)
    else:
        fix = np.zeros(m.shape, bool)
        dbm = c.dist_LA_BedMachine_km.values
    # frente na data de referência
    allz = np.ones(m.shape, bool)
    for v in ("classe", "classe_f080", "classe_f095", "classe_ic"):
        allz &= c[v].values == 0
    cand = (m == 3) & allz & ~union
    lab2, n2 = ndimage.label(cand)
    toca = np.unique(lab2[ndimage.binary_dilation(union) & cand])
    ext = np.isin(lab2, toca[toca > 0])
    ou0 = np.nonzero(union & (plat > 0))
    pe = plat[ou0][cKDTree(np.c_[XX[ou0], YY[ou0]]).query(np.c_[XX[ext], YY[ext]])[1]]
    ei, ej = np.nonzero(ext)
    ext[ei[~np.isin(pe, PLAT_FRENTE_REF)], ej[~np.isin(pe, PLAT_FRENTE_REF)]] = False
    # sem isto o domínio final (que exige classe == 1) descartava a extensão inteira (bug de 2026-11-19:
    # cobertura v3 só +83 km² sobre a v2 em vez de ~+864); flutuante longe da LA → hidrostático
    classe[ext] = 1
    novo = ext | fix
    # plataforma e unidade pelo pixel da união mais próximo
    ou = np.nonzero(union & (plat > 0) & ~fix)
    tu = cKDTree(np.c_[XX[ou], YY[ou]])
    dn, idx = tu.query(np.c_[XX[novo], YY[novo]])
    ok = dn <= 15e3
    plat = plat.copy(); unit = unit.copy().astype(object)
    ii, jj = np.nonzero(novo)
    plat[ii[ok], jj[ok]] = plat[ou][idx[ok]]
    unit[ii[ok], jj[ok]] = unit[ou][idx[ok]]
    union = union | ext
    # distância às linhas recentes (2019–2025) onde faltava
    drec = c.dist_LA_recente_km.values.copy()
    lr = load_recent_gl_lines(ROOT, 2019)
    tr = cKDTree(np.vstack([_coordinates(g, 250.0) for g in lr.geometry if g is not None]))
    falta = novo & ~np.isfinite(drec)
    drec[falta] = tr.query(np.c_[XX[falta], YY[falta]])[0] / 1e3
    chk = union & np.isfinite(c.dist_LA_recente_km.values) & ~novo
    amostra = np.flatnonzero(chk.ravel())[:: max(1, chk.sum() // 20000)]
    d_ref = c.dist_LA_recente_km.values.ravel()[amostra]
    d_new = tr.query(np.c_[XX.ravel()[amostra], YY.ravel()[amostra]])[0] / 1e3
    info.update({"pixels_ilha_anel": int(fix.sum()), "pixels_frente_ref_km2": float(ext.sum() * 0.25),
                 "drec_recalculado_vs_original_mediana_abs_km": float(np.median(np.abs(d_new - d_ref)))})
    return union, unit.astype(str), plat, classe, drec, dbm, info


def main() -> int:
    """``--d-front X`` refaz o produto com outra distância à frente (variante do fatorial): grava só
    ``mb_final_v2_dfrontX.nc`` e não toca no produto central nem nas tabelas.
    ``--v3`` grava ``mb_final_v3_500m.nc`` (∇·u a 3 km fora da Thwaites, ilhas sem suporte, frente na
    data de referência); ver ``ajustes_v3``."""
    import argparse
    import json
    global D_FRONT
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-front", type=float, default=None)
    ap.add_argument("--v3", action="store_true")
    a = ap.parse_args()
    variant = a.d_front is not None and a.d_front != D_FRONT
    if a.d_front is not None:
        D_FRONT = a.d_front
    out_nc = OUT_NC.with_name(f"mb_final_v2_dfront{D_FRONT:g}.nc") if variant else OUT_NC
    if a.v3:
        out_nc = OUT_NC.with_name("mb_final_v3_500m.nc")
    print(f"D_FRONT = {D_FRONT} km -> {out_nc.name}", flush=True)
    c = xr.open_dataset(COMP)
    xs, ys = c.x.values, c.y.values
    XX, YY = np.meshgrid(xs, ys)
    unit, plat = c.unidade.values, c.plataforma.values
    union = np.zeros(unit.shape, bool)
    for v in ("classe", "classe_f080", "classe_f095", "classe_ic"):
        union |= c[v].values == 1
    classe_dom = c.classe.values
    drec, dbm = c.dist_LA_recente_km.values, c.dist_LA_BedMachine_km.values
    opt = OPT_SCALE
    if a.v3:
        union, unit, plat, classe_dom, drec, dbm, info3 = ajustes_v3(c, xs, ys, union, unit, plat)
        opt = OPT_SCALE_V3
        print("v3:", json.dumps(info3, ensure_ascii=False, default=float), flush=True)
    fr_all = pd.read_parquet(FRONTS)
    fr_all = fr_all[fr_all.epoch_year >= 2018.5]
    dh = np.full(unit.shape, np.nan)
    se = np.full(unit.shape, np.nan)
    Hg = np.full(unit.shape, np.nan)
    rows = []
    for k, name in NAMES.items():
        cols = ["x", "y", "t", "pass_id", "h_sea", "H_gemb", "classe", "classe_f080", "classe_f095",
                "classe_ic"]
        d = pd.read_parquet(THK / f"espessura_{name}.parquet", columns=cols)
        anyv = np.zeros(len(d), bool)
        for v in ("classe", "classe_f080", "classe_f095", "classe_ic"):
            anyv |= d[v].to_numpy() == 1
        if a.v3:   # v3: o pixel do segmento na união estendida (inclui ilhas sem suporte e a frente)
            i0, j0, ok0 = to_pix(d.x.to_numpy(), d.y.to_numpy(), xs, ys)
            i0, j0 = np.clip(i0, 0, len(ys) - 1), np.clip(j0, 0, len(xs) - 1)
            anyv = ok0 & union[i0, j0] & (plat[i0, j0] == k)
        d = d[anyv & np.isfinite(d["H_gemb"])].reset_index(drop=True)
        n0 = len(d)
        fr = fr_all[fr_all.shelf.isin(FRONT_OF[name])]
        dfr = epoch_front_distance(d.x.to_numpy(), d.y.to_numpy(), d.t.to_numpy(), fr)
        i, j, ok = to_pix(d.x.to_numpy(), d.y.to_numpy(), xs, ys)
        key = np.where(ok, i * len(xs) + j, -1)
        p75 = pd.Series(d.h_sea.to_numpy()).groupby(key).transform(lambda v: np.nanpercentile(v, 75)).to_numpy()
        keep = (dfr > D_FRONT) & (d.h_sea.to_numpy() >= 0.5 * p75) & (p75 >= 5.0)
        d = d[keep].reset_index(drop=True)
        rows.append({"plataforma": name, "segmentos": n0, "frac_fora_frente_data": float(np.mean(dfr <= D_FRONT)),
                     "frac_mantida": len(d) / n0})
        g = grid_median(d.x.to_numpy(), d.y.to_numpy(), d.H_gemb.to_numpy(), xs, ys)
        Hg[plat == k] = g[plat == k]
        vel = load_velocity(d.x.to_numpy(), d.y.to_numpy())
        X0, Y0 = advect_to_ref(d.x.to_numpy(), d.y.to_numpy(), d.t.to_numpy(), vel)
        del vel
        ct = cell_trends(X0, Y0, d.t.to_numpy(), d.H_gemb.to_numpy(float), d.pass_id.to_numpy())
        i, j, ok = to_pix(ct.x.to_numpy() - 250.0, ct.y.to_numpy() - 250.0, xs, ys)
        ii, jj = np.clip(i, 0, len(ys) - 1), np.clip(j, 0, len(xs) - 1)
        ins = ok & union[ii, jj] & (plat[ii, jj] == k)
        dh[i[ins], j[ins]] = ct.dHdt.to_numpy()[ins]
        se[i[ins], j[ins]] = ct.se.to_numpy()[ins]
        print(f"{name}: {len(d):,} de {n0:,} segmentos; {ins.sum():,} células", flush=True)
        del d, X0, Y0, ct
        gc.collect()
    Hg = fill_gaps(Hg, union)
    dh_c = np.full(unit.shape, np.nan)
    dist_obs = np.full(unit.shape, np.nan)
    for u in UNITS:
        su = unit == u
        ob = su & np.isfinite(dh) & np.isfinite(se)
        xo, yo, vo, so = XX[ob], YY[ob], dh[ob], se[ob]
        if u == "Cosgrove":
            p, _ = kernel_predict(xo, yo, vo, so, XX[su], YY[su], 8e3)
        else:
            p, _ = krig_predict(xo, yo, vo, so, XX[su], YY[su], fit_variogram(xo, yo, vo, so))
        dh_c[su] = p
        dist_obs[su] = cKDTree(np.c_[xo, yo]).query(np.c_[XX[su], YY[su]])[0]

    # produto central
    delta = xr.open_dataset(THR).delta_correcao.values
    front_static_ok = (c.frente_dist_km.values > D_FRONT) & ~(c.perda_gelo_frac.values > 0.10)
    smb = c.SMB.values
    if a.v3:   # SMB onde a união antiga não tinha (mesma função do produto)
        from run_basal_melt_ase import smb_rate_field
        falta = union & ~np.isfinite(smb)
        if falta.any():
            smb = np.where(falta, smb_rate_field(xs, ys, falta)[0], smb)
    mb = np.full(unit.shape, np.nan)
    dom = np.zeros(unit.shape, bool)
    summ = []
    for u in UNITS:
        su = unit == u
        div = c[f"div_{opt[u]}km"].values
        m = smb - dh_c - Hg * div
        if u.startswith("Thwaites"):
            m = m + np.nan_to_num(delta)
            radar_ok = np.ones(su.shape, bool)
        else:
            radar_ok = dbm >= D_RADAR[u]
        base = su & (classe_dom == 1) & (drec >= D_REC[u]) & radar_ok & np.isfinite(m)
        front_ok = front_static_ok | (dist_obs <= 1000.0)
        dd = base & front_ok
        old = base & front_static_ok
        mb[dd], dom[dd] = m[dd], True
        summ.append({"unidade": u, "area_antes_km2": old.sum() * 0.25, "area_km2": dd.sum() * 0.25,
                     "recuperado_frente_km2": (dd & ~old).sum() * 0.25,
                     "mb_mpa": float(np.mean(m[dd])), "mb_antes_mpa": float(np.mean(m[old])),
                     "mb_faixa_recuperada_mpa": float(np.mean(m[dd & ~old])) if (dd & ~old).any() else np.nan,
                     "Gt": float(np.mean(m[dd]) * dd.sum() * PIX * RHO_I / 1e12)})
    t = pd.DataFrame(summ)
    if a.v3:
        t.to_csv(OUT / "mb_final_v3_por_unidade.csv", index=False)
        pd.DataFrame(rows).to_csv(OUT / "frente_por_data_segmentos_v3.csv", index=False)
        (OUT / "mb_final_v3_ajustes.json").write_text(json.dumps(info3, indent=1, ensure_ascii=False, default=float),
                                                       encoding="utf-8")
    elif not variant:
        t.to_csv(OUT / "mb_final_v2_por_unidade.csv", index=False)
        pd.DataFrame(rows).to_csv(OUT / "frente_por_data_segmentos.csv", index=False)
    xr.Dataset({"mb": (("y", "x"), mb.astype(np.float32)), "dominio": (("y", "x"), dom.astype(np.int8)),
                "DHDt": (("y", "x"), dh_c.astype(np.float32)), "H": (("y", "x"), Hg.astype(np.float32)),
                "dist_obs_m": (("y", "x"), dist_obs.astype(np.float32)),
                "unidade": (("y", "x"), unit)}, coords={"x": xs, "y": ys},
               attrs={"descricao": ("produto v3: v2 + ∇·u a 3 km fora da Thwaites, ilhas do BedMachine sem suporte "
                                    "atual tratadas como flutuante, frente na data de referência (T_REF 2022,0)")
                      if a.v3 else "produto central v2: frente por data; Thwaites com correção β·maré+radar"}
               ).to_netcdf(out_nc)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(t.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
