"""Série trimestral do derretimento basal por plataforma no domínio do produto central v2.

Mesmos critérios do produto (``run_front_epoch_ase.py``): só pixels do domínio v2; segmento a
> 1,5 km da frente IceLines da mesma data; presença de gelo h ≥ 0,5 × p75 do pixel.

Por unidade:
- variação de espessura por estação-ano (28 estações, DJF 2019 a SON 2025) por efeitos fixos
  Lagrangianos (``run_thickness_ase.fe_series``), erro por jackknife em 20 grupos de passagens;
- DH/Dt(k): inclinação numa janela de 5 estações (±2), ponderada por 1/σ²; as duas estações de
  cada ponta usam janela unilateral e ficam marcadas (``ponta``);
- H·∇·u do ano (∇·u anual ITS_LIVE na escala ótima × H do produto v2);
- SMB do trimestre (GEMB, acumulado médio da unidade);
- Thwaites: soma-se o Δ médio da correção β·maré + radar no domínio (constante no tempo: a
  correção é estática, então afeta o nível e não a variabilidade).

m_b(k) = SMB − DH/Dt − H·∇·u (+ Δ). σ(k) = σ da inclinação (só o estatístico; o estrutural está
no fatorial e é quase constante no tempo).

Saída: ``outputs/series_v2/mb_trimestral_v2.csv`` e ``espessura_trimestral_v2.csv``.
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
from run_thickness_ase import N_JACK, SEASONS, fe_series, season_year  # noqa: E402
from run_tide_response_lagrangian import advect_to_ref, load_velocity  # noqa: E402

THK = ROOT / "data" / "interim" / "thickness"
V2 = ROOT / "data" / "interim" / "basal_melt" / "mb_final_v2_500m.nc"
THR = ROOT / "data" / "interim" / "basal_melt" / "thwaites_recuperada_500m.nc"
DIV = ROOT / "data" / "interim" / "divergence"
SMB = ROOT / "data" / "basal_forcing" / "gemb_smb_accum_roi_2018_2025.nc"
FRONTS = ROOT / "data" / "calving_fronts_densified_500m.parquet"
OUT = ROOT / "outputs" / "series_v2"
NAMES = {1: "Pine_Island", 2: "Thwaites", 3: "Crosson", 4: "Dotson", 5: "Cosgrove"}
NK = 28


def window_slope(t, y, e, half=2):
    s_out, e_out = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for k in range(len(y)):
        s = slice(max(k - half, 0), min(k + half + 1, len(y)))
        tw, yw, ew = t[s], y[s], np.maximum(e[s], 1e-3)
        ok = np.isfinite(yw)
        if ok.sum() < 3:
            continue
        tw, yw, W = tw[ok], yw[ok], 1 / ew[ok] ** 2
        tb = np.sum(W * tw) / W.sum()
        s_out[k] = np.sum(W * (tw - tb) * (yw - np.sum(W * yw) / W.sum())) / np.sum(W * (tw - tb) ** 2)
        e_out[k] = np.sqrt(1 / np.sum(W * (tw - tb) ** 2))
    return s_out, e_out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    v2 = xr.open_dataset(V2)
    xs, ys = v2.x.values, v2.y.values
    XX, YY = np.meshgrid(xs, ys)
    unit, dom, Hmap = v2.unidade.values, v2.dominio.values == 1, v2.H.values
    delta = np.nan_to_num(xr.open_dataset(THR).delta_correcao.values)
    fr_all = pd.read_parquet(FRONTS)
    fr_all = fr_all[fr_all.epoch_year >= 2018.5]
    smb = xr.open_dataset(SMB)
    acc_all = smb.smb_accum.values.astype(float)
    tsmb = smb.time.values.astype(float)
    tree_smb = cKDTree(np.c_[smb.x.values, smb.y.values])
    rows_h, rows_m = [], []
    for k_sh, name in NAMES.items():
        cols = ["x", "y", "t", "pass_id", "h_sea", "H_gemb", "granule_month", "granule_year"]
        d = pd.read_parquet(THK / f"espessura_{name}.parquet", columns=cols)
        d = d[np.isfinite(d.H_gemb)].reset_index(drop=True)
        i, j, ok = to_pix(d.x.to_numpy(), d.y.to_numpy(), xs, ys)
        ii, jj = np.clip(i, 0, len(ys) - 1), np.clip(j, 0, len(xs) - 1)
        indom = ok & dom[ii, jj]
        d, ii, jj = d[indom].reset_index(drop=True), ii[indom], jj[indom]
        fr = fr_all[fr_all.shelf.isin(FRONT_OF[name])]
        dfr = epoch_front_distance(d.x.to_numpy(), d.y.to_numpy(), d.t.to_numpy(), fr)
        key = ii * len(xs) + jj
        p75 = pd.Series(d.h_sea.to_numpy()).groupby(key).transform(lambda v: np.nanpercentile(v, 75)).to_numpy()
        keep = (dfr > D_FRONT) & (d.h_sea.to_numpy() >= 0.5 * p75) & (p75 >= 5.0)
        d, ii, jj = d[keep].reset_index(drop=True), ii[keep], jj[keep]
        u_seg = unit[ii, jj]
        s_idx, yy = season_year(d.granule_month.to_numpy(), d.granule_year.to_numpy())
        kk = ((yy - 2019) * 4 + s_idx).astype(int)
        good = (kk >= 0) & (kk < NK)
        d, u_seg, kk = d[good].reset_index(drop=True), u_seg[good], kk[good]
        vel = load_velocity(d.x.to_numpy(), d.y.to_numpy())
        X0, Y0 = advect_to_ref(d.x.to_numpy(), d.y.to_numpy(), d.t.to_numpy(), vel)
        del vel
        for u in sorted(set(u_seg)):
            if not u:
                continue
            m = u_seg == u
            px, py, H, k, pid = X0[m], Y0[m], d.H_gemb.to_numpy(float)[m], kk[m], d.pass_id.to_numpy()[m]
            s, n = fe_series(px, py, H, k, NK)
            upid = np.unique(pid)
            grp = np.random.default_rng(7).integers(0, N_JACK, len(upid))
            gp = grp[np.searchsorted(upid, pid)]
            jk = np.array([fe_series(px[gp != g], py[gp != g], H[gp != g], k[gp != g], NK)[0]
                           for g in range(N_JACK)])
            se = np.sqrt((N_JACK - 1) / N_JACK * np.nansum((jk - np.nanmean(jk, 0)) ** 2, 0))
            tm = 2019 + np.arange(NK) // 4 + np.array([0.0, 0.25, 0.5, 0.75])[np.arange(NK) % 4] + 0.04
            for q in range(NK):
                rows_h.append({"unidade": u, "t_meio": tm[q], "ano": 2019 + q // 4,
                               "estacao": SEASONS[q % 4], "dH_m": s[q], "se_m": se[q], "n_seg": int(n[q])})
            slope, se_slope = window_slope(tm, s, se)
            su = dom & (unit == u)
            sc = OPT_SCALE[u]
            dv = xr.open_dataset(DIV / f"div_{sc}km.nc")
            near = np.unique(tree_smb.query(np.c_[XX[su], YY[su]])[1])
            acc = acc_all[:, near].mean(1)
            dmean = float(np.mean(delta[su])) if u.startswith("Thwaites") else 0.0
            for q in range(NK):
                yr = int(np.clip(2019 + q // 4, 2019, 2025))
                dmap = dv.div_anual.sel(ano=yr).values
                okk = su & np.isfinite(dmap) & np.isfinite(Hmap)
                hd = float(np.mean(Hmap[okk] * dmap[okk]))
                t0, t1 = tm[q] - 0.125, tm[q] + 0.125
                smb_rate = (np.interp(t1, tsmb, acc) - np.interp(t0, tsmb, acc)) / (t1 - t0)
                rows_m.append({"unidade": u, "t_meio": tm[q], "ano": 2019 + q // 4, "estacao": SEASONS[q % 4],
                               "DHDt_mpa": slope[q], "se_mpa": se_slope[q], "Hdiv_mpa": hd,
                               "SMB_mpa": smb_rate, "correcao_mpa": dmean,
                               "mb_mpa": smb_rate - slope[q] - hd + dmean,
                               "ponta": q < 2 or q > NK - 3})
            print(f"{u}: {m.sum():,} segmentos", flush=True)
        del d, X0, Y0
        gc.collect()
    pd.DataFrame(rows_h).to_csv(OUT / "espessura_trimestral_v2.csv", index=False)
    t = pd.DataFrame(rows_m)
    t.to_csv(OUT / "mb_trimestral_v2.csv", index=False)
    pd.set_option("display.width", 200)
    print(t[~t.ponta].groupby("unidade")[["mb_mpa", "se_mpa"]].agg(["mean", "std", "min", "max"]).round(2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
