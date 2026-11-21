"""Derretimento basal das plataformas da ASE (passo 7 do plano basal): mapa médio e série.

Balanço de massa Lagrangiano, em metros de gelo equivalente por ano:

    m_b = SMB − DH/Dt − H·∇·u          (m_b > 0 = derretimento)

**Mapa 2019–2025 (500 m)**
- DH/Dt por célula material de 500 m (posição de 2022,0, ITS_LIVE anual, RK2):
  H = a + plano + c·(t − 2022) + ciclo anual; c = DH/Dt. Erro de c: CR1 agrupado por passagem.
  Exige ≥ 8 passagens e ≥ 3 anos de cobertura na célula. A célula fica na posição de 2022,0 da
  parcela, que é a posição euleriana do pixel do mapa nessa data.
- SMB: taxa linear 2019–2025 do acumulado do GEMB v1.5 (m gelo eq.), interpolada linearmente.
- H·∇·u: escala ótima de cada unidade (``run_flux_term_ase.py``).
- σ(m_b) por pixel = √(σ_DH/Dt² + σ_Hdiv²), com σ_Hdiv = H × ruído de ∇·u (anos pares × ímpares,
  ``escala_diagnostico.csv``). É só o ruído; o envelope estrutural (firn, máscara, escala) é o
  passo 8.

**Série trimestral por unidade**
- DH/Dt(k): inclinação local da série de efeitos fixos (``serie_espessura_classe.csv``) numa janela
  de 5 estações (±2 trimestres), ponderada por 1/σ²; σ da inclinação pelo erro das médias.
- H·∇·u do ano (campo anual de ∇·u × H médio) e SMB do trimestre (GEMB).
- A média da série (ponderada pela informação) e a média do mapa (por área) são comparadas como
  verificação de consistência de pesos.

Unidades: PIG, Thwaites_TEIS, Thwaites_TWGT, Crosson, Dotson, Cosgrove. A TWGT é fragmentada e
o mélange foi filtrado, mas a hipótese hidrostática ali é a mais frágil; ela é reportada à parte.

Saídas: ``data/interim/basal_melt/mb_500m.nc``; ``outputs/basal_melt_ase/`` (tabelas e séries).
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import Delaunay, cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipelines" / "derretimento_basal"))
from run_thickness_ase import ice_presence_filter, speed_split  # noqa: E402
from run_tide_response_lagrangian import load_velocity, advect_to_ref  # noqa: E402

MASK = ROOT / "data" / "interim" / "hydrostatic_mask_500m.nc"
THK = ROOT / "data" / "interim" / "thickness"
FLUX = ROOT / "data" / "interim" / "flux_term" / "H_div_500m.nc"
DIV = ROOT / "data" / "interim" / "divergence"
SMB = ROOT / "data" / "basal_forcing" / "gemb_smb_accum_roi_2018_2025.nc"
SERIES = ROOT / "outputs" / "thickness_ase" / "serie_espessura_classe.csv"
DIAG = ROOT / "outputs" / "divergence_ase" / "escala_diagnostico.csv"
BEST = ROOT / "outputs" / "divergence_ase" / "escala_otima.csv"
OUT_NC = ROOT / "data" / "interim" / "basal_melt" / "mb_500m.nc"
OUT = ROOT / "outputs" / "basal_melt_ase"
NAMES = {1: "Pine_Island", 2: "Thwaites", 3: "Crosson", 4: "Dotson", 5: "Cosgrove"}
CELL = 500.0
T_REF = 2022.0
MIN_PASS, MIN_SPAN = 8, 3.0


def cell_trends(px, py, t, H, pid):
    """Tendência por célula com plano e ciclo anual; CR1 por passagem."""
    key = np.c_[np.floor(px / CELL), np.floor(py / CELL)].astype(np.int64)
    u, g = np.unique(key, axis=0, return_inverse=True)
    g = g.ravel()
    nb = len(u)
    cx, cy = (u[:, 0] + 0.5) * CELL, (u[:, 1] + 0.5) * CELL
    tt = t - T_REF
    X = np.c_[np.ones_like(tt), (px - cx[g]) / 100, (py - cy[g]) / 100, tt,
              np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)]
    p = X.shape[1]
    A = np.zeros((nb, p, p))
    for a in range(p):
        for b in range(a, p):
            v = np.bincount(g, weights=X[:, a] * X[:, b], minlength=nb)
            A[:, a, b] = A[:, b, a] = v
    Xy = np.stack([np.bincount(g, weights=X[:, a] * H, minlength=nb) for a in range(p)], 1)
    span = (pd.Series(t).groupby(g).max() - pd.Series(t).groupby(g).min()).reindex(
        range(nb)).to_numpy()
    kp = np.unique(np.c_[g, pid], axis=0)
    npass = np.bincount(kp[:, 0], minlength=nb)
    cond = np.linalg.cond(A)
    ok = np.isfinite(cond) & (cond < 1e9) & (npass >= MIN_PASS) & (span >= MIN_SPAN)
    Ai = np.zeros_like(A)
    Ai[ok] = np.linalg.inv(A[ok])
    beta = np.einsum("bij,bj->bi", Ai, Xy)
    r = H - np.einsum("ij,ij->i", X, beta[g])
    # escores por (célula, passagem)
    kk, inv = np.unique(np.c_[g, pid], axis=0, return_inverse=True)
    inv = inv.ravel()
    S = np.stack([np.bincount(inv, weights=X[:, a] * r) for a in range(p)], 1)
    B = np.zeros((nb, p, p))
    np.add.at(B, kk[:, 0], S[:, :, None] * S[:, None, :])
    G = npass.astype(float)
    corr = np.where(G > 1, G / np.maximum(G - 1, 1), np.nan)
    V = corr[:, None, None] * np.einsum("bij,bjk,bkl->bil", Ai, B, Ai)
    return pd.DataFrame({"x": cx[ok], "y": cy[ok], "dHdt": beta[ok, 3],
                         "se": np.sqrt(np.maximum(V[ok, 3, 3], 0)), "n_pass": npass[ok]})


def smb_rate_field(xs, ys, mask):
    d = xr.open_dataset(SMB)
    a = d["smb_accum"].values.astype(float)
    t = d["time"].values.astype(float)
    sel = (t >= 2019.0) & (t <= 2026.0)
    rate = np.polyfit(t[sel], a[sel], 1)[0]            # m gelo eq./a por ponto
    xy = np.c_[d["x"].values, d["y"].values]
    tri = Delaunay(xy)
    ii, jj = np.nonzero(mask)
    p = np.c_[xs[jj], ys[ii]]
    s = tri.find_simplex(p)
    out = np.full(mask.shape, np.nan)
    okk = s >= 0
    T = tri.transform[s[okk]]
    b = np.einsum("ijk,ik->ij", T[:, :2], p[okk] - T[:, 2])
    w = np.c_[b, 1 - b.sum(1)]
    out[ii[okk], jj[okk]] = (rate[tri.simplices[s[okk]]] * w).sum(1)
    return out, d, sel


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_NC.parent.mkdir(parents=True, exist_ok=True)
    m = xr.open_dataset(MASK)
    xs, ys = m.x.values, m.y.values
    dom = m.classe.values == 1
    plat = m.plataforma.values
    best = dict(pd.read_csv(BEST).values)
    flux = xr.open_dataset(FLUX)
    diag = pd.read_csv(DIAG)

    # unidades (Thwaites dividida pela velocidade)
    unit = np.full(dom.shape, "", dtype=object)
    for k, n in NAMES.items():
        unit[dom & (plat == k)] = n
    ii, jj = np.nonzero(dom & (plat == 2))
    fs = speed_split(xs[jj], ys[ii])
    unit[ii[fs], jj[fs]] = "Thwaites_TWGT"
    unit[ii[~fs], jj[~fs]] = "Thwaites_TEIS"

    # ---- DH/Dt por célula material
    dhdt = np.full(dom.shape, np.nan)
    se_dh = np.full(dom.shape, np.nan)
    for k, name in NAMES.items():
        d = pd.read_parquet(THK / f"espessura_{name}.parquet",
                            columns=["x", "y", "t", "pass_id", "h_sea", "H_gemb", "classe"])
        d = d[(d["classe"] == 1) & np.isfinite(d["H_gemb"])].reset_index(drop=True)
        d = d[ice_presence_filter(d["x"].to_numpy(), d["y"].to_numpy(),
                                  d["h_sea"].to_numpy())].reset_index(drop=True)
        x, y, t = d["x"].to_numpy(), d["y"].to_numpy(), d["t"].to_numpy()
        vel = load_velocity(x, y)
        X0, Y0 = advect_to_ref(x, y, t, vel)
        del vel
        ct = cell_trends(X0, Y0, t, d["H_gemb"].to_numpy(float), d["pass_id"].to_numpy())
        j = np.floor((ct["x"].to_numpy() - xs[0]) / (xs[1] - xs[0])).astype(int)
        i = np.floor((ct["y"].to_numpy() - ys[0]) / (ys[1] - ys[0])).astype(int)
        ok = (i >= 0) & (i < len(ys)) & (j >= 0) & (j < len(xs))
        i, j, ct = i[ok], j[ok], ct[ok]
        inside = dom[i, j] & (plat[i, j] == k)
        dhdt[i[inside], j[inside]] = ct["dHdt"].to_numpy()[inside]
        se_dh[i[inside], j[inside]] = ct["se"].to_numpy()[inside]
        print(f"{name}: {inside.sum():,} células com DH/Dt (de {(dom & (plat == k)).sum():,} "
              f"pixels); σ mediano {np.median(ct['se'][inside]):.2f} m/a", flush=True)
        del d, x, y, t, X0, Y0
        gc.collect()

    # ---- SMB e H·∇·u
    smb, smb_ds, smb_sel = smb_rate_field(xs, ys, dom)
    hdiv = np.full(dom.shape, np.nan)
    sd_hdiv = np.full(dom.shape, np.nan)
    H = flux["H_gemb"].values
    for u in sorted(set(unit[dom])):
        base = u.replace("_TEIS", "").replace("_TWGT", "")
        sc = int(best[base])
        sel = unit == u
        hdiv[sel] = flux[f"H_div_gemb_{sc}km"].values[sel]
        noise = diag[(diag.plataforma == base) & (diag.sigma_km == sc)]["ruido_media7a_1pa"]
        sd_hdiv[sel] = H[sel] * float(noise.iloc[0])
    mb = smb - dhdt - hdiv
    se_mb = np.sqrt(se_dh ** 2 + sd_hdiv ** 2)

    # ---- resumo por unidade (média por área dos pixels com DH/Dt)
    ser = pd.read_csv(SERIES)
    rows = []
    for u in ["Pine_Island", "Thwaites_TEIS", "Thwaites_TWGT", "Crosson", "Dotson", "Cosgrove"]:
        sel = (unit == u)
        ok = sel & np.isfinite(mb)
        g = ser[ser.plataforma == u]
        w = 1 / np.maximum(g.se_lagr_m.to_numpy(), 1e-3) ** 2
        fe_trend = np.polyfit(g.t_meio, g.dH_lagr_m, 1, w=np.sqrt(w))[0]
        rows.append({"unidade": u, "area_km2": sel.sum() * 0.25,
                     "frac_area_com_mb": ok.sum() / sel.sum(),
                     "mb_media_area_mpa": float(np.mean(mb[ok])),
                     "mb_mediana_mpa": float(np.median(mb[ok])),
                     "mb_p90_mpa": float(np.percentile(mb[ok], 90)),
                     "SMB_mpa": float(np.nanmean(smb[ok])),
                     "DHDt_media_area_mpa": float(np.mean(dhdt[ok])),
                     "DHDt_serie_FE_mpa": float(fe_trend),
                     "Hdiv_mpa": float(np.mean(hdiv[ok])),
                     "mb_com_DHDt_da_serie_mpa": float(np.nanmean(smb[sel]) - fe_trend
                                                      - np.nanmean(hdiv[sel])),
                     "se_pixel_mediano_mpa": float(np.median(se_mb[ok]))})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "mb_por_unidade.csv", index=False)
    pd.set_option("display.width", 240)
    print(t.round(2).to_string(index=False))

    xr.Dataset({"mb": (("y", "x"), mb.astype(np.float32)),
                "se_mb": (("y", "x"), se_mb.astype(np.float32)),
                "DHDt": (("y", "x"), dhdt.astype(np.float32)),
                "se_DHDt": (("y", "x"), se_dh.astype(np.float32)),
                "SMB": (("y", "x"), smb.astype(np.float32)),
                "H_div": (("y", "x"), hdiv.astype(np.float32)),
                "unidade": (("y", "x"), np.array(unit, dtype=str))},
               coords={"x": xs, "y": ys},
               attrs={"unidade_fisica": "m gelo eq./a; m_b > 0 = derretimento",
                      "periodo": "2019-2025", "posicao": "Lagrangiana em 2022,0"}
               ).to_netcdf(OUT_NC)

    # ---- série trimestral
    a = smb_ds["smb_accum"].values.astype(float)
    tsmb = smb_ds["time"].values.astype(float)
    srow = []
    divs = {sc: xr.open_dataset(DIV / f"div_{sc}km.nc") for sc in (1, 2, 3)}
    for u in ["Pine_Island", "Thwaites_TEIS", "Thwaites_TWGT", "Crosson", "Dotson", "Cosgrove"]:
        base = u.replace("_TEIS", "").replace("_TWGT", "")
        sel = unit == u
        ii, jj = np.nonzero(sel)
        tree = cKDTree(np.c_[smb_ds["x"].values, smb_ds["y"].values])
        near = np.unique(tree.query(np.c_[xs[jj], ys[ii]])[1])
        acc = a[:, near].mean(1)                         # acumulado médio da unidade
        dv = divs[int(best[base])]
        g = ser[ser.plataforma == u].reset_index(drop=True)
        tm, s, e = g.t_meio.to_numpy(), g.dH_lagr_m.to_numpy(), g.se_lagr_m.to_numpy()
        for k in range(len(g)):
            w = slice(max(k - 2, 0), min(k + 3, len(g)))
            tw, sw, ew = tm[w], s[w], np.maximum(e[w], 1e-3)
            W = 1 / ew ** 2
            tb = np.sum(W * tw) / W.sum()
            slope = np.sum(W * (tw - tb) * (sw - np.sum(W * sw) / W.sum())) / np.sum(W * (tw - tb) ** 2)
            se_slope = np.sqrt(1 / np.sum(W * (tw - tb) ** 2))
            yr = int(g.ano[k])
            yi = int(np.clip(yr, dv.ano.values.min(), dv.ano.values.max()))
            dmap = dv.div_anual.sel(ano=yi).values
            okk = sel & np.isfinite(dmap) & np.isfinite(H)
            hd = float(np.mean(H[okk] * dmap[okk]))
            t0, t1 = tm[k] - 0.125, tm[k] + 0.125
            smb_rate = (np.interp(t1, tsmb, acc) - np.interp(t0, tsmb, acc)) / (t1 - t0)
            srow.append({"unidade": u, "t_meio": tm[k], "ano": yr, "estacao": g.estacao[k],
                         "DHDt_mpa": slope, "se_DHDt_mpa": se_slope, "Hdiv_mpa": hd,
                         "SMB_mpa": smb_rate, "mb_mpa": smb_rate - slope - hd,
                         "se_mb_mpa": se_slope})
    s = pd.DataFrame(srow)
    s.to_csv(OUT / "mb_serie_trimestral.csv", index=False)
    print(s.groupby("unidade")[["mb_mpa", "se_mb_mpa"]].agg(["mean", "std", "median"]).round(2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
