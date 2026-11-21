"""Canais basais, passo 4: aprofundamento LAGRANGIANO dos canais por emparelhamento de seções.

Motivação (passo 3): numa seção fixa, a profundidade D muda porque o fluxo traz outra parte do canal
(na PIG o gelo anda ~27 km em 2019–2025); o ruído das séries cresce com a velocidade (ρ = 0,72).
Aqui segue-se a coluna de gelo: uma observação i (centro do canal em x_i, y_i, data t_i) é
transportada pela velocidade média ITS_LIVE 2019–2025 (RK2, passo de 15 dias) até a data t_j de
cada outra observação j de canal da mesma plataforma; se a posição transportada cai a menos de
R_PAR do centro do canal em j, (i, j) é o mesmo gelo visto duas vezes, e ΔD = D_j − D_i.
Só se aceitam pares de trilhas (rgt) diferentes ou iguais, com Δt entre DT_MIN e DT_MAX.

D continua sendo uma diferença dentro do mesmo perfil (canal × quilhas vizinhas no mesmo instante):
maré, DAC e geoide se cancelam. ΔD/Δt = taxa lagrangiana de aprofundamento da depressão superficial;
em espessura hidrostática, ×9,34. Ela inclui melt diferencial (canal − quilha) e a diferença de
H·∇·u entre canal e quilha (estimada como pequena: ΔH·∇·u ~ 80 m × 0,01 a⁻¹ ≈ 1 m/a na PIG).

Erro de transporte: σ_u·Δt (ITS_LIVE v_error); pares com erro de posição esperado > R_PAR/2 saem.
Estimação por plataforma: Huber de ΔD contra Δt sem intercepto (com intercepto como teste de viés
entre trilhas) e bootstrap em blocos espaciais de 10 km pela posição do par.
Saída: outputs/canais/pares_lagrangianos.parquet, lagrangiano_plataformas.csv
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gl_migration"))
from p03_migracao import huber  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "data" / "interim" / "canais"
OUT = ROOT / "outputs" / "canais"
RNG = np.random.default_rng(20261117)
K_HID = 1027.0 / (1027.0 - 917.0)
R_PAR = float(os.environ.get("R_PAR", "400"))
SUF = os.environ.get("SUF", "")
SUF_OUT = SUF + os.environ.get("VAL", "") + ("" if R_PAR == 400 else f"_r{int(R_PAR)}")
DT_MIN, DT_MAX = 0.2, 3.0
PASSO = 15 / 365.25


def campo_vel(x, y):
    v = xr.open_dataset(ROOT / "data" / "velocity_itslive_annual.nc")
    sub = v[["vx", "vy", "v_error"]].sel(x=slice(x.min() - 40e3, x.max() + 40e3),
                                         y=slice(y.min() - 40e3, y.max() + 40e3)).mean("time").load()
    xs, ys = sub.x.values, sub.y.values   # ITS_LIVE aqui tem y crescente
    f = {k: RegularGridInterpolator((ys, xs), sub[k].values, bounds_error=False)
         for k in ("vx", "vy", "v_error")}
    return f


def transporta(f, x, y, T):
    """posições em tempos 0..T (anos) a cada PASSO; devolve tempos e (n, k, 2)."""
    nk = int(np.ceil(T / PASSO)) + 1
    tt = np.arange(nk) * PASSO
    P = np.full((len(x), nk, 2), np.nan)
    P[:, 0, 0], P[:, 0, 1] = x, y
    for k in range(1, nk):
        p = P[:, k - 1]
        v1 = np.c_[f["vx"](p[:, ::-1]), f["vy"](p[:, ::-1])]
        pm = p + 0.5 * PASSO * v1
        v2 = np.c_[f["vx"](pm[:, ::-1]), f["vy"](pm[:, ::-1])]
        P[:, k] = p + PASSO * v2
    return tt, P


def main():
    s = pd.read_parquet(IN / f"secoes{SUF}.parquet")
    e = pd.read_parquet(IN / f"canais_validados{os.environ.get('VAL', '')}.parquet")
    s = s.merge(e[["canal", "canal_basal", "largura_m", "theta"]], on="canal")
    s = s[s.canal_basal & np.isfinite(s.x) & np.isfinite(s.y)].reset_index(drop=True)
    # um valor por par de feixes e ciclo (feixes a ~90 m veem o mesmo canal)
    s["par"] = (s.beam + 1) // 2
    s = (s.groupby(["plataforma", "rgt", "par", "cycle", "canal"]).agg(t=("t", "median"), x=("x", "median"), y=("y", "median"),
                                                                       D=("D", "median"), larg=("largura_m", "first"))
         .reset_index())
    s["chave"] = s.plataforma + "_" + s.rgt.astype(str) + "_" + s.par.astype(str) + "_" + s.canal.str.split("_").str[-1]
    s = s.groupby(["plataforma", "rgt", "chave", "cycle"]).agg(t=("t", "median"), x=("x", "median"), y=("y", "median"),
                                                               D=("D", "mean"), larg=("larg", "first")).reset_index()
    pares = []
    for pl, g in s.groupby("plataforma"):
        g = g.reset_index(drop=True)
        f = campo_vel(g.x.to_numpy(), g.y.to_numpy())
        tt, P = transporta(f, g.x.to_numpy(), g.y.to_numpy(), DT_MAX)
        sig = f["v_error"](np.c_[g.y, g.x])
        t = g.t.to_numpy()
        X, Y = g.x.to_numpy(), g.y.to_numpy()
        for i in range(len(g)):
            dt = t - t[i]
            j = np.nonzero((dt >= DT_MIN) & (dt <= DT_MAX))[0]
            if len(j) == 0:
                continue
            k = np.clip(np.rint(dt[j] / PASSO).astype(int), 0, len(tt) - 1)
            px, py = P[i, k, 0], P[i, k, 1]
            dist = np.hypot(px - X[j], py - Y[j])
            erro = np.nan_to_num(sig[i], nan=50.0) * dt[j]
            ok = (dist < R_PAR) & (erro < R_PAR / 2) & np.isfinite(dist)
            for jj, dd, ee in zip(j[ok], dist[ok], erro[ok]):
                desl = np.hypot(X[jj] - X[i], Y[jj] - Y[i])
                pares.append(dict(plataforma=pl, i=g.chave[i], j=g.chave[jj], rgt_i=g.rgt[i], rgt_j=g.rgt[jj],
                                  t_i=t[i], t_j=t[jj], dt=dt[jj], D_i=g.D[i], D_j=g.D[jj], dD=g.D[jj] - g.D[i],
                                  dist=dd, erro_pos=ee, deslocamento=desl, x=0.5 * (X[i] + X[jj]), y=0.5 * (Y[i] + Y[jj])))
        print(pl, "obs", len(g), "pares", sum(p["plataforma"] == pl for p in pares), flush=True)
    p = pd.DataFrame(pares)
    OUT.mkdir(parents=True, exist_ok=True)
    p.to_parquet(OUT / f"pares_lagrangianos{SUF_OUT}.parquet", index=False)
    p["bloco"] = (p.x // 10000).astype(int).astype(str) + "_" + (p.y // 10000).astype(int).astype(str)
    p["mesma_trilha"] = p.rgt_i == p.rgt_j
    res = []
    for pl, g in p.groupby("plataforma"):
        if len(g) < 20:
            continue
        taxa = huber(g[["dt"]].to_numpy(), g.dD.to_numpy())[0]
        bi = huber(np.c_[np.ones(len(g)), g.dt], g.dD.to_numpy())
        bl = g.bloco.unique()
        reps = []
        for _ in range(300):
            q = pd.concat([g[g.bloco == b] for b in RNG.choice(bl, len(bl))])
            reps.append(huber(q[["dt"]].to_numpy(), q.dD.to_numpy())[0])
        res.append(dict(plataforma=pl, n_pares=len(g), n_blocos=len(bl), frac_outra_trilha=(~g.mesma_trilha).mean(),
                        desloc_med_km=g.deslocamento.median() / 1e3, dt_med=g.dt.median(),
                        taxa=taxa, lo=np.percentile(reps, 2.5), hi=np.percentile(reps, 97.5),
                        intercepto=bi[0], taxa_com_int=bi[1],
                        taxa_mesma_trilha=huber(g[g.mesma_trilha][["dt"]].to_numpy(), g[g.mesma_trilha].dD.to_numpy())[0]
                        if g.mesma_trilha.sum() > 10 else np.nan,
                        taxa_outra_trilha=huber(g[~g.mesma_trilha][["dt"]].to_numpy(), g[~g.mesma_trilha].dD.to_numpy())[0]
                        if (~g.mesma_trilha).sum() > 10 else np.nan,
                        dispersao=1.4826 * np.median(np.abs(g.dD - taxa * g.dt))))
    r = pd.DataFrame(res)
    for c in ("taxa", "lo", "hi"):
        r[c + "_H"] = r[c] * K_HID
    r.to_csv(OUT / f"lagrangiano_plataformas{SUF_OUT}.csv", index=False)
    with pd.option_context("display.width", 250):
        print(r.round(3).to_string())


if __name__ == "__main__":
    main()
