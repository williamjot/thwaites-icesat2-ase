"""Resposta à maré com topografia Lagrangiana (para plataformas rápidas: Thwaites, Crosson).

Problema medido no ajuste euleriano (``run_tide_response_ase.py``): na Thwaites e no Crosson o
RMS por célula foi de 2,1–2,6 m e β ficou indeterminado. Entre passagens, o gelo avança centenas
de metros a quilômetros, e as ondulações da superfície atravessam a célula fixa.

Modelo conjunto:

    h_i = M_b(i)(X0_i, t_i) + a_c(i) + β_c(i) · T_i + ε_i

- ``M_b``: topografia que **anda com o gelo**. ``b`` é a célula material de 200 m na posição
  ``X0`` que a parcela ocupa em ``T_REF`` (integração da velocidade ITS_LIVE anual, RK2, passo
  ≤ 0,05 ano). Termos: constante, plano local, tendência linear e quadrática no tempo (o Dh/Dt de
  uma parcela que atravessa a quebra de declive não é linear) e ciclo anual (firn);
- ``a_c + β_c·T``: resposta à maré **fixa no espaço**. ``c`` é a célula euleriana de 500 m na
  posição observada, porque a flexão depende da distância à linha de aterramento e não da parcela;
- ``T`` = maré CATS2008 + DAC, como no ajuste euleriano.

Estimação conjunta exata: a topografia é eliminada analiticamente (complemento de Schur) e o
sistema esparso de β é resolvido direto. Ajustar a topografia antes e β depois enviesaria β para 0,
porque a tendência e o ciclo anual absorveriam o alias da maré na amostragem de 91 dias. Uma versão
anterior por backfitting não convergiu em 25 iterações na Thwaites (|Δβ| p99 = 1,9).
Referencial da topografia: material sobre gelo flutuante; fixo sobre gelo apoiado.

Erro de β: sanduíche CR1 agrupado por passagem ``(rgt, cycle)``, com a topografia projetada fora
(Z'MZ), em aproximação diagonal (ignora a covariância entre células de maré acopladas pela
mesma parcela).

Controle: na Cosgrove (lenta), o resultado deve reproduzir o euleriano.

Entrada: cache ``data/interim/tide_response/mare_<plataforma>.parquet`` (etapas 1–2 do euleriano).
Saída: ``outputs/tide_response_ase/beta_lagr_<plataforma>.parquet`` e o resumo comparativo.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "interim" / "tide_response"
VEL = ROOT / "data" / "velocity_itslive_annual.nc"
OUT = ROOT / "outputs" / "tide_response_ase"
T_REF = 2022.0
DT_MAX = 0.05
MAT_CELL = 200.0
EUL_CELL = 500.0
MIN_PASS_MAT = 8
MIN_PASS_EUL = 10
N_ITER = 25


# ---------------------------------------------------------------- velocidade e advecção
def load_velocity(x, y, pad=30e3, stride=2):
    v = xr.open_dataset(VEL)
    ys = v.y.values
    ysl = slice(y.min() - pad, y.max() + pad) if ys[0] < ys[-1] else slice(y.max() + pad,
                                                                         y.min() - pad)
    s = v.sel(x=slice(x.min() - pad, x.max() + pad), y=ysl).isel(
        x=slice(None, None, stride), y=slice(None, None, stride))
    vx, vy = s.vx.values.astype(np.float32), s.vy.values.astype(np.float32)
    years = s.time.dt.year.values
    # lacunas de um ano: média dos outros anos; lacuna permanente (oceano): vizinho mais próximo
    mx, my = np.nanmean(vx, 0), np.nanmean(vy, 0)
    for a, m in ((vx, mx), (vy, my)):
        bad = np.isnan(a)
        a[bad] = np.broadcast_to(m, a.shape)[bad]
    still = np.isnan(vx[0])
    idx = ndimage.distance_transform_edt(still, return_distances=False, return_indices=True)
    vx[:, still] = vx[:, idx[0][still], idx[1][still]]
    vy[:, still] = vy[:, idx[0][still], idx[1][still]]
    return s.x.values, s.y.values, years, vx, vy


def advect_to_ref(x, y, t, vel):
    gx, gy, years, vx, vy = vel
    dx, dy = gx[1] - gx[0], gy[1] - gy[0]
    span = T_REF - t
    n = int(np.ceil(np.abs(span).max() / DT_MAX))
    h = span / n

    def u_at(px, py, tau):
        k = np.clip(np.floor(tau).astype(int) - years[0], 0, len(years) - 1)
        ci = (px - gx[0]) / dx
        cj = (py - gy[0]) / dy
        out_x = np.empty_like(px)
        out_y = np.empty_like(py)
        for kk in np.unique(k):
            s = k == kk
            co = np.vstack([cj[s], ci[s]])
            out_x[s] = ndimage.map_coordinates(vx[kk], co, order=1, mode="nearest")
            out_y[s] = ndimage.map_coordinates(vy[kk], co, order=1, mode="nearest")
        return out_x, out_y

    px, py, tau = x.astype(float).copy(), y.astype(float).copy(), t.astype(float).copy()
    for _ in range(n):                                     # RK2 (ponto médio)
        ux, uy = u_at(px, py, tau)
        mx, my = px + 0.5 * h * ux, py + 0.5 * h * uy
        ux, uy = u_at(mx, my, tau + 0.5 * h)
        px += h * ux
        py += h * uy
        tau += h
    return px, py


# ---------------------------------------------------------------- MQ em lote por grupo
def batched_ls(X, y, g, ng):
    """Resolve um MQ por grupo; devolve coeficientes (ng, p) com NaN onde é singular."""
    p = X.shape[1]
    XtX = np.zeros((ng, p, p))
    Xty = np.zeros((ng, p))
    for a in range(p):
        Xty[:, a] = np.bincount(g, weights=X[:, a] * y, minlength=ng)
        for b in range(a, p):
            v = np.bincount(g, weights=X[:, a] * X[:, b], minlength=ng)
            XtX[:, a, b] = v
            XtX[:, b, a] = v
    beta = np.full((ng, p), np.nan)
    cond = np.linalg.cond(XtX)
    ok = np.isfinite(cond) & (cond < 1e10)
    beta[ok] = np.linalg.solve(XtX[ok], Xty[ok][..., None])[..., 0]
    return beta


def unique_pass_count(g, pid):
    key = np.unique(np.c_[g, pid], axis=0)
    return np.bincount(key[:, 0], minlength=g.max() + 1)


# ---------------------------------------------------------------- ajuste conjunto
def fit_shelf(name, vel_cache=None):
    """Solução exata (sem iteração) do modelo h = M_b + β_c·T.

    A topografia M_b de cada célula topográfica é eliminada analiticamente (complemento de Schur):
        H β = g,   H = Z'Z − Σ_b V_b' A_b⁻¹ V_b,   g = Z'h − Σ_b V_b' A_b⁻¹ X_b'h,
    com A_b = X_b'X_b e V_b = X_b'Z_b (Z tem uma coluna por célula euleriana: T na célula, 0 fora).
    H é esparsa (só se acoplam células eulerianas atravessadas pela mesma parcela).

    Referencial da topografia, por segmento: material (Lagrangiano) sobre gelo flutuante
    (dist_gl > 0) e fixo (euleriano) sobre gelo apoiado. Medido na versão totalmente Lagrangiana:
    no apoiado o RMS piorava (Thwaites, 0,8 → 1,5 m) porque ali o relevo reflete o leito e não anda
    com o gelo; seguir a parcela só somava o erro da velocidade.
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    d = pd.read_parquet(CACHE / f"mare_{name}.parquet")
    d = d[np.isfinite(d["tide_cats"])].reset_index(drop=True)
    x, y, t = d["x"].to_numpy(), d["y"].to_numpy(), d["t"].to_numpy()
    h = d["h"].to_numpy(float)
    T = d["tide_cats"].to_numpy(float) + d["dac"].to_numpy(float)
    pid = d["pass_id"].to_numpy()
    dist = d["dist_gl"].to_numpy()
    del d
    gc.collect()

    vel = load_velocity(x, y)
    X0, Y0 = advect_to_ref(x, y, t, vel)
    flo = dist > 0
    PX, PY = np.where(flo, X0, x), np.where(flo, Y0, y)        # coordenada da topografia
    disp_km = np.where(flo, np.hypot(X0 - x, Y0 - y) / 1e3, 0.0)
    del X0, Y0, vel
    gc.collect()

    def cell_index(px, py, size, tag=None):
        k = np.c_[np.floor(px / size), np.floor(py / size)].astype(np.int64)
        if tag is not None:
            k = np.c_[k, tag.astype(np.int64)]
        u, inv = np.unique(k, axis=0, return_inverse=True)
        return inv.ravel(), u

    gm, um = cell_index(PX, PY, MAT_CELL, flo)
    ge, ue = cell_index(x, y, EUL_CELL)
    keep = (unique_pass_count(gm, pid)[gm] >= MIN_PASS_MAT) & \
           (unique_pass_count(ge, pid)[ge] >= MIN_PASS_EUL)
    sel = np.nonzero(keep)[0]
    x, y, t, h, T, pid, PX, PY, disp_km, dist, flo = (
        a[sel] for a in (x, y, t, h, T, pid, PX, PY, disp_km, dist, flo))
    gm, um = cell_index(PX, PY, MAT_CELL, flo)
    ge, ue = cell_index(x, y, EUL_CELL)
    nm, ne = len(um), len(ue)

    cxm = (um[:, 0] + 0.5) * MAT_CELL
    cym = (um[:, 1] + 0.5) * MAT_CELL
    tt = t - T_REF
    Xm = np.c_[np.ones_like(tt), (PX - cxm[gm]) / 100.0, (PY - cym[gm]) / 100.0, tt, tt ** 2,
               np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)]
    p = Xm.shape[1]

    # A_b⁻¹ em lote; células topográficas singulares saem
    A = np.zeros((nm, p, p))
    for a in range(p):
        for b in range(a, p):
            v = np.bincount(gm, weights=Xm[:, a] * Xm[:, b], minlength=nm)
            A[:, a, b] = v
            A[:, b, a] = v
    cond = np.linalg.cond(A)
    okb = np.isfinite(cond) & (cond < 1e10)
    Ai = np.zeros_like(A)
    Ai[okb] = np.linalg.inv(A[okb])
    rowok = okb[gm]
    print(f"  {name}: {len(h):,} segmentos, {nm:,} células topográficas "
          f"({(~okb).sum():,} singulares), {ne:,} células de maré", flush=True)

    # pares (célula topográfica, célula de maré)
    pk = gm.astype(np.int64) * ne + ge
    upk, pinv = np.unique(pk[rowok], return_inverse=True)
    pinv = pinv.ravel()
    pb, pc = upk // ne, upk % ne
    V = np.stack([np.bincount(pinv, weights=Xm[rowok, a] * T[rowok], minlength=len(upk))
                  for a in range(p)], 1)                       # v_bc
    W = np.einsum("bij,bj->bi", Ai[pb], V)                     # A_b⁻¹ v_bc
    u = np.stack([np.bincount(gm[rowok], weights=Xm[rowok, a] * h[rowok], minlength=nm)
                  for a in range(p)], 1)                       # X_b'h
    Aiu = np.einsum("bij,bj->bi", Ai, u)

    ZtZ = np.bincount(ge[rowok], weights=T[rowok] ** 2, minlength=ne)
    Zth = np.bincount(ge[rowok], weights=T[rowok] * h[rowok], minlength=ne)
    g = Zth - np.bincount(pc, weights=np.einsum("ij,ij->i", V, Aiu[pb]), minlength=ne)

    # termo de acoplamento Σ_b v_bc'·A_b⁻¹·v_bc' — combinações de pares dentro da mesma célula b
    order = np.argsort(pb, kind="stable")
    pbs = pb[order]
    starts = np.r_[0, np.nonzero(np.diff(pbs))[0] + 1]
    counts = np.diff(np.r_[starts, len(pbs)])
    rr_, cc_, vv_ = [], [], []
    for k in np.unique(counts):
        st = starts[counts == k]
        idx = order[st[:, None] + np.arange(k)[None, :]]       # (n_bins_k, k)
        i1 = np.repeat(idx, k, axis=1).ravel()
        i2 = np.tile(idx, (1, k)).ravel()
        rr_.append(pc[i1])
        cc_.append(pc[i2])
        vv_.append(np.einsum("ij,ij->i", V[i1], W[i2]))
    Hc = sparse.coo_matrix((np.concatenate(vv_), (np.concatenate(rr_), np.concatenate(cc_))),
                           shape=(ne, ne)).tocsr()
    H = (sparse.diags(ZtZ) - Hc).tocsc()
    H = H + sparse.diags(np.full(ne, 1e-9 * max(ZtZ.max(), 1.0)))
    beta = spsolve(H, g)
    Hdiag = H.diagonal()

    # topografia e resíduo
    VB = np.zeros((nm, p))
    np.add.at(VB, pb, V * beta[pc][:, None])                   # Σ_c v_bc β_c
    bcoef = np.einsum("bij,bj->bi", Ai, u - VB)
    r = h - np.einsum("ij,ij->i", Xm, bcoef[gm]) - beta[ge] * T
    r[~rowok] = np.nan

    # erro de β: sanduíche CR1 por passagem, com só o termo da própria célula de maré
    # (aproximação diagonal: ignora a covariância entre células acopladas pela topografia)
    widx = np.searchsorted(upk, pk[rowok])
    Mz = np.full(len(h), np.nan)
    Mz[rowok] = T[rowok] - np.einsum("ij,ij->i", Xm[rowok], W[widx])
    ok = rowok & np.isfinite(r)
    gi = ge[ok]
    key, inv = np.unique(np.c_[gi, pid[ok]], axis=0, return_inverse=True)
    s1 = np.bincount(inv.ravel(), weights=Mz[ok] * r[ok])
    kc = key[:, 0]
    S11 = np.bincount(kc, weights=s1 * s1, minlength=ne)
    G = np.bincount(kc, minlength=ne)
    n = np.bincount(gi, minlength=ne)
    rr = np.bincount(gi, weights=r[ok] ** 2, minlength=ne)
    dist_med = pd.Series(dist[ok]).groupby(gi).median()
    disp_med = pd.Series(disp_km[ok]).groupby(gi).median()
    rows = []
    for c in np.nonzero((G >= MIN_PASS_EUL) & (n > 4) & (Hdiag > 0))[0]:
        var = G[c] / (G[c] - 1) * S11[c] / Hdiag[c] ** 2
        rows.append({"shelf": name, "x": (ue[c, 0] + 0.5) * EUL_CELL,
                     "y": (ue[c, 1] + 0.5) * EUL_CELL,
                     "dist_gl_km": float(dist_med.get(c, np.nan)) / 1e3,
                     "beta": float(beta[c]), "se_beta": float(np.sqrt(max(var, 0))),
                     "n_pass": int(G[c]), "n_seg": int(n[c]),
                     "rms_m": float(np.sqrt(rr[c] / n[c])),
                     "desloc_med_km": float(disp_med.get(c, np.nan))})
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / f"beta_lagr_{name}.parquet", index=False)
    return out


def compare(name, lag):
    eul = pd.read_parquet(OUT / "beta_celulas.parquet")
    eul = eul[eul.shelf == name]
    m = lag.merge(eul, on=["x", "y"], suffixes=("_lag", "_eul"))
    bins = [-15, -5, -2, -1, 0, 1, 2, 3, 5, 10, 15]
    m["faixa"] = pd.cut(m["dist_gl_km_lag"], bins)
    s = m.groupby("faixa", observed=True).agg(
        n=("beta_lag", "size"),
        beta_eul=("beta_eul", "median"), beta_lag=("beta_lag", "median"),
        se_eul=("se_beta_eul", "median"), se_lag=("se_beta_lag", "median"),
        rms_eul=("rms_m_eul", "median"), rms_lag=("rms_m_lag", "median")).round(3)
    print(f"\n== {name}: {len(m):,} células em comum; deslocamento mediano "
          f"{lag['desloc_med_km'].median():.1f} km")
    print(f"   RMS mediano {m.rms_m_eul.median():.2f} → {m.rms_m_lag.median():.2f} m; "
          f"σβ<0,25 em {(m.se_beta_eul < .25).mean():.0%} → {(m.se_beta_lag < .25).mean():.0%}")
    pd.set_option("display.width", 200)
    print(s)
    s.to_csv(OUT / f"comparacao_eul_lagr_{name}.csv")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("shelves", nargs="+")
    args = ap.parse_args()
    for name in args.shelves:
        lag = fit_shelf(name)
        compare(name, lag)
        del lag
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
