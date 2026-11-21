"""O derretimento basal da PIG responde ao desprendimento de fevereiro de 2020?

Desprendimento: IceLines mostra recuo de 8,8 km da face da frente da PIG entre 2019,62 e 2020,50
(``run_calving_series_ase.py``); o ICESat-2 mostra pico de água no setor da frente em meados de 2020.

Janelas (estações-ano do produto): ANTES = DJF 2019 a DJF 2020 (k 0–4); DEPOIS = JJA 2020 a SON
2021 (k 6–11). MAM 2020 (k 5), que é a transição, fica fora. Em cada janela, DH/Dt é a inclinação
da série de espessura por efeitos fixos só com os pontos da própria janela (as duas não dividem
pontos), ponderada por 1/σ²; σ da inclinação pela propagação dos σ trimestrais.
m_b = SMB − DH/Dt − H·∇·u; H·∇·u do ano (2019 antes; média 2020–2021 depois); SMB GEMB por janela.

Teste 1 (método × física): a PIG dividida em LONGE (> 10 km de qualquer frente IceLines de
2019–2024,7: nunca alcançada pelo desprendimento nem pela mudança de domínio) e PERTO (≤ 10 km).
Teste 3 (local × regional): o mesmo Δ para TEIS, TWGT, Crosson, Dotson e Cosgrove.

Leitura: queda só na PIG e maior perto da frente → compatível com resposta ao desprendimento;
queda também no setor longe e nas plataformas sem desprendimento → causa regional (oceano/ENSO/
firn). Limitação: os σ trimestrais da série não são independentes entre si (células comuns),
então o σ do Δ é aproximado.

Saída: ``outputs/series_v2/teste_resposta_desprendimento.csv``.
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
from run_front_epoch_ase import D_FRONT, FRONT_OF, OPT_SCALE, epoch_front_distance, to_pix  # noqa: E402
from run_thickness_ase import N_JACK, fe_series, season_year  # noqa: E402
from run_tide_response_lagrangian import advect_to_ref, load_velocity  # noqa: E402

THK = ROOT / "data" / "interim" / "thickness"
V2 = ROOT / "data" / "interim" / "basal_melt" / "mb_final_v2_500m.nc"
COMP = ROOT / "data" / "interim" / "basal_melt" / "componentes_uniao_500m.nc"
DIV = ROOT / "data" / "interim" / "divergence"
SMB = ROOT / "data" / "basal_forcing" / "gemb_smb_accum_roi_2018_2025.nc"
FRONTS = ROOT / "data" / "calving_fronts_densified_500m.parquet"
SERIES = ROOT / "outputs" / "series_v2" / "espessura_trimestral_v2.csv"
OUT = ROOT / "outputs" / "series_v2"
NK = 28
PRE, POST = list(range(0, 5)), list(range(6, 12))
TM = 2019 + np.arange(NK) // 4 + np.array([0.0, 0.25, 0.5, 0.75])[np.arange(NK) % 4] + 0.04
FAR_KM = 10.0


def slope(t, y, e):
    ok = np.isfinite(y) & np.isfinite(e)
    t, y, w = t[ok], y[ok], 1 / np.maximum(e[ok], 1e-3) ** 2
    tb = np.sum(w * t) / w.sum()
    s = np.sum(w * (t - tb) * (y - np.sum(w * y) / w.sum())) / np.sum(w * (t - tb) ** 2)
    return s, np.sqrt(1 / np.sum(w * (t - tb) ** 2))


def forcing(mask, unit_scale, smb_ds, tree, XX, YY, Hmap):
    near = np.unique(tree.query(np.c_[XX[mask], YY[mask]])[1])
    acc = smb_ds.smb_accum.values[:, near].mean(1)
    tsmb = smb_ds.time.values.astype(float)
    rate = lambda a, b: (np.interp(b, tsmb, acc) - np.interp(a, tsmb, acc)) / (b - a)  # noqa: E731
    dv = xr.open_dataset(DIV / f"div_{unit_scale}km.nc")
    hd = {}
    for yr in (2019, 2020, 2021):
        dm = dv.div_anual.sel(ano=yr).values
        ok = mask & np.isfinite(dm) & np.isfinite(Hmap)
        hd[yr] = float(np.mean(Hmap[ok] * dm[ok]))
    return {"smb_pre": rate(TM[PRE[0]] - 0.125, TM[PRE[-1]] + 0.125),
            "smb_post": rate(TM[POST[0]] - 0.125, TM[POST[-1]] + 0.125),
            "hd_pre": hd[2019], "hd_post": 0.5 * (hd[2020] + hd[2021])}


def delta_row(name, s, se, f):
    a, ea = slope(TM[PRE], s[PRE], se[PRE])
    b, eb = slope(TM[POST], s[POST], se[POST])
    mb_pre = f["smb_pre"] - a - f["hd_pre"]
    mb_post = f["smb_post"] - b - f["hd_post"]
    return {"conjunto": name, "mb_antes": mb_pre, "sigma_antes": ea, "mb_depois": mb_post,
            "sigma_depois": eb, "delta": mb_post - mb_pre, "sigma_delta": float(np.hypot(ea, eb)),
            "z": (mb_post - mb_pre) / float(np.hypot(ea, eb)),
            "delta_DHDt": -(b - a), "delta_Hdiv": -(f["hd_post"] - f["hd_pre"]),
            "delta_SMB": f["smb_post"] - f["smb_pre"]}


def main() -> int:
    v2 = xr.open_dataset(V2)
    xs, ys = v2.x.values, v2.y.values
    XX, YY = np.meshgrid(xs, ys)
    unit, dom, Hmap = v2.unidade.values, v2.dominio.values == 1, v2.H.values
    dfront = xr.open_dataset(COMP).frente_dist_km.values
    smb_ds = xr.open_dataset(SMB)
    tree = cKDTree(np.c_[smb_ds.x.values, smb_ds.y.values])
    rows = []
    # Teste 3: todas as unidades, a partir das séries já calculadas
    ser = pd.read_csv(SERIES)
    for u, g in ser.groupby("unidade"):
        g = g.sort_values("t_meio")
        f = forcing(dom & (unit == u), OPT_SCALE[u], smb_ds, tree, XX, YY, Hmap)
        rows.append(delta_row(u, g.dH_m.to_numpy(), g.se_m.to_numpy(), f))
    # Teste 1: PIG perto × longe da frente
    pig_far = dom & (unit == "Pine_Island") & (dfront > FAR_KM)
    pig_near = dom & (unit == "Pine_Island") & (dfront <= FAR_KM)
    d = pd.read_parquet(THK / "espessura_Pine_Island.parquet",
                        columns=["x", "y", "t", "pass_id", "h_sea", "H_gemb", "granule_month", "granule_year"])
    d = d[np.isfinite(d.H_gemb)].reset_index(drop=True)
    i, j, ok = to_pix(d.x.to_numpy(), d.y.to_numpy(), xs, ys)
    ii, jj = np.clip(i, 0, len(ys) - 1), np.clip(j, 0, len(xs) - 1)
    sel = ok & (pig_far[ii, jj] | pig_near[ii, jj])
    d, ii, jj = d[sel].reset_index(drop=True), ii[sel], jj[sel]
    fr = pd.read_parquet(FRONTS)
    fr = fr[(fr.epoch_year >= 2018.5) & fr.shelf.isin(FRONT_OF["Pine_Island"])]
    dfr = epoch_front_distance(d.x.to_numpy(), d.y.to_numpy(), d.t.to_numpy(), fr)
    key = ii * len(xs) + jj
    p75 = pd.Series(d.h_sea.to_numpy()).groupby(key).transform(lambda v: np.nanpercentile(v, 75)).to_numpy()
    keep = (dfr > D_FRONT) & (d.h_sea.to_numpy() >= 0.5 * p75) & (p75 >= 5.0)
    d, ii, jj = d[keep].reset_index(drop=True), ii[keep], jj[keep]
    s_idx, yy = season_year(d.granule_month.to_numpy(), d.granule_year.to_numpy())
    k = ((yy - 2019) * 4 + s_idx).astype(int)
    vel = load_velocity(d.x.to_numpy(), d.y.to_numpy())
    X0, Y0 = advect_to_ref(d.x.to_numpy(), d.y.to_numpy(), d.t.to_numpy(), vel)
    del vel
    gc.collect()
    for name, msk in (("Pine_Island_longe_>10km", pig_far), ("Pine_Island_perto_<=10km", pig_near)):
        m = msk[ii, jj] & (k >= 0) & (k < NK)
        px, py, H, kk, pid = X0[m], Y0[m], d.H_gemb.to_numpy(float)[m], k[m], d.pass_id.to_numpy()[m]
        s, _ = fe_series(px, py, H, kk, NK)
        upid = np.unique(pid)
        grp = np.random.default_rng(7).integers(0, N_JACK, len(upid))
        gp = grp[np.searchsorted(upid, pid)]
        jk = np.array([fe_series(px[gp != g], py[gp != g], H[gp != g], kk[gp != g], NK)[0] for g in range(N_JACK)])
        se = np.sqrt((N_JACK - 1) / N_JACK * np.nansum((jk - np.nanmean(jk, 0)) ** 2, 0))
        f = forcing(msk, OPT_SCALE["Pine_Island"], smb_ds, tree, XX, YY, Hmap)
        r = delta_row(name, s, se, f)
        r["area_km2"] = msk.sum() * 0.25
        rows.append(r)
        print(f"{name}: {m.sum():,} segmentos, {msk.sum() * 0.25:.0f} km²", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "teste_resposta_desprendimento.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
