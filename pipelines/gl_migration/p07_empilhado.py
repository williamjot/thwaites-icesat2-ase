"""Migração da linha de aterramento — passo 7: perfis de β empilhados por plataforma.

O passo 4 (β por subconjunto de ciclos em cada perfil) deixou quase nenhum perfil válido: com ~18
ciclos por trilha em 2019–2025, metade dos ciclos não determina um perfil de β. Aqui empilhamos.

1. Por perfil (mesma origem, grade e limpeza dos passos 2 e 4): ajuste por intervalo
   r = a + b·(t − t̄) + β·(T − T̄) com todos os ciclos; resíduo com a maré preservada
   e(x, k) = r − a − b·(t_k − t̄). Coordenada alinhada x' = x − x50_ref (meia flexão média do perfil).
2. Por subconjunto S de ciclos (maré baixa/alta: abaixo/acima da mediana de T do perfil; tempo:
   t < 2022,0 / t ≥ 2022,5), em cada intervalo de x' e com todos os perfis juntos:
   e = c_{p,S}(x') + β_S(x')·(T − T̄_{p,S})  (intercepto por perfil, inclinação comum) — MQO.
3. β_S normalizado pela mediana no gelo livre (x' de +5 a +10 km); deslocamento δ entre os dois
   perfis do par por ajuste L1 (busca de ±3 km, passo 100 m) em x' de −5 a +8 km.
   Δ_maré = (δ_alta − δ_baixa)/ΔT̄; Δ_tempo = (δ_depois − δ_antes)/Δt̄ (negativo = para o interior).
4. IC de 95% por bootstrap sobre perfis (1.000 réplicas).
Filtros de perfil (os mesmos do passo 6, fixados antes): β no gelo livre 0,7–1,3; x50_ref entre −3 e
+6 km da linha do BedMachine; ≥ 12 ciclos.
Saída: outputs/gl_migration/empilhado_resumo.csv e empilhado_perfis_beta.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas as _pd


def median_filter(a, size=5, mode=None):
    """Mediana móvel centrada que ignora NaN (o median_filter do scipy corrompe vizinhos de NaN)."""
    return _pd.Series(a, dtype=float).rolling(size, center=True, min_periods=(size + 1) // 2).median().to_numpy().copy()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p02_flexao_por_ciclo import IN, B, cruzamentos, primeiro_cruzamento  # noqa: E402
from p04_subconjuntos import GRADE, limpa, perfil_beta  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "gl_migration"
RNG = np.random.default_rng(20261117)
XA = np.arange(-8000 + B / 2, 12000, B)          # grade alinhada x'
NBOOT = 1000


def residuos(R, t, T):
    dt, dT = t - t.mean(), T - T.mean()
    M = np.isfinite(R)
    Y = np.where(M, R, 0.0)
    X = np.stack([np.ones_like(dt), dt, dT])
    XtX = np.einsum("pk,qk,ik->ipq", X, X, M.astype(float))
    XtY = np.einsum("pk,ik->ip", X, Y)
    ok = (M.sum(1) >= 6) & (np.linalg.det(XtX) > 1e-9)
    E = np.full(R.shape, np.nan)
    sol = np.linalg.solve(XtX[ok], XtY[ok][..., None])[..., 0]
    E[ok] = R[ok] - sol[:, [0]] - sol[:, [1]] * dt[None, :]
    return E


def coleta(p):
    d = pd.read_parquet(IN / f"perfis_{p}.parquet")
    perfis = []
    for (rgt, par), g in d.groupby(["rgt", "par"]):
        for k, (sg, orient, lo, hi) in enumerate(cruzamentos(g)):
            gg = g[(g.s > lo) & (g.s < hi)]
            gg = gg.assign(xb=np.floor(orient * (gg.s - sg) / B) * B + B / 2)
            gg = gg[(gg.xb > -12000) & (gg.xb < 14000)]
            if len(gg) == 0 or gg.xb.min() > -4000 or gg.xb.max() < 8000:
                continue
            pb = gg.groupby(["ciclo", "xb"]).agg(r=("r", "median"), T=("T", "median"), t=("t", "median")).reset_index()
            cic = pb.groupby("ciclo").agg(T=("T", "median"), t=("t", "median"))
            if len(cic) < 12:
                continue
            R = limpa(pb.pivot(index="xb", columns="ciclo", values="r").reindex(index=GRADE, columns=cic.index).to_numpy())
            t, T = cic.t.to_numpy(), cic["T"].to_numpy()
            ref, far = perfil_beta(R, t, T)
            if ref is None or not (0.7 <= far <= 1.3):
                continue
            x50 = primeiro_cruzamento(GRADE, ref, 0.5)
            if not (-3000 <= x50 <= 6000):
                continue
            E = residuos(R, t, T)
            # realinha para x' = x − x50
            idx = np.round((GRADE - x50 - XA[0]) / B).astype(int)
            Ea = np.full((len(XA), len(t)), np.nan)
            ok = (idx >= 0) & (idx < len(XA))
            Ea[idx[ok]] = E[ok]
            perfis.append(dict(pid=f"{rgt}_{par}_{k}", E=Ea, t=t, T=T, x50=x50, far=far))
    return perfis


def beta_empilhado(perfis, sel):
    """sel(perfil) -> máscara de ciclos. β comum por intervalo com intercepto por perfil."""
    num = np.zeros(len(XA))
    den = np.zeros(len(XA))
    for pf in perfis:
        m = sel(pf)
        if m.sum() < 3:
            continue
        E = pf["E"][:, m]
        T = pf["T"][m]
        M = np.isfinite(E)
        n = M.sum(1)
        Tm = np.where(M, T[None, :], 0).sum(1) / np.maximum(n, 1)        # T̄ por intervalo (intercepto)
        Tc = np.where(M, T[None, :] - Tm[:, None], 0)
        Em = np.where(M, E, 0).sum(1) / np.maximum(n, 1)
        Ec = np.where(M, E - Em[:, None], 0)
        num += (Tc * Ec).sum(1) * (n >= 2)
        den += (Tc ** 2).sum(1) * (n >= 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(den > 0.5, num / den, np.nan)
    far = np.nanmedian(b[(XA > 5000) & (XA < 10000)])
    return median_filter(b / far, size=5, mode="nearest"), far


def desloc(b1, b0, lim=3000):
    jan = (XA > -5000) & (XA < 8000)
    melhor, q0 = np.inf, np.nan
    for k in range(-int(lim / B), int(lim / B) + 1):
        tk = np.roll(b0, k)
        if k > 0:
            tk[:k] = np.nan
        elif k < 0:
            tk[k:] = np.nan
        ok = jan & np.isfinite(b1) & np.isfinite(tk)
        if ok.sum() < 60:
            continue
        c = np.mean(np.abs(b1[ok] - tk[ok]))
        if c < melhor:
            melhor, q0 = c, k * B
    return q0


SEL = {"baixa": lambda pf: pf["T"] < np.median(pf["T"]), "alta": lambda pf: pf["T"] >= np.median(pf["T"]),
       "antes": lambda pf: pf["t"] < 2022.0, "depois": lambda pf: pf["t"] >= 2022.5}


def estimativas(perfis):
    bs = {k: beta_empilhado(perfis, f)[0] for k, f in SEL.items()}
    dT = np.mean([pf["T"][SEL["alta"](pf)].mean() - pf["T"][SEL["baixa"](pf)].mean() for pf in perfis])
    dt = np.mean([pf["t"][SEL["depois"](pf)].mean() - pf["t"][SEL["antes"](pf)].mean() for pf in perfis
                  if SEL["depois"](pf).any() and SEL["antes"](pf).any()])
    dTt = np.nanmean([pf["T"][SEL["depois"](pf)].mean() - pf["T"][SEL["antes"](pf)].mean() for pf in perfis
                      if SEL["depois"](pf).any() and SEL["antes"](pf).any()])
    return dict(mare=desloc(bs["alta"], bs["baixa"]) / dT, tempo=desloc(bs["depois"], bs["antes"]) / dt,
                dT=dT, dt=dt, dT_tempo=dTt), bs


def main(plats):
    linhas, curvas = [], []
    for p in plats:
        perfis = coleta(p)
        if len(perfis) < 8:
            print(p, "perfis", len(perfis)); continue
        est, bs = estimativas(perfis)
        for k, b in bs.items():
            curvas.append(pd.DataFrame({"plataforma": p, "subconjunto": k, "x": XA, "beta": b}))
        reps = []
        for _ in range(NBOOT):
            amostra = [perfis[i] for i in RNG.integers(0, len(perfis), len(perfis))]
            reps.append(estimativas(amostra)[0])
        reps = pd.DataFrame(reps)
        lin = dict(plataforma=p, n_perfis=len(perfis), dT_medio=est["dT"], dt_medio=est["dt"],
                   dT_entre_periodos=est["dT_tempo"],
                   mare_m_por_m=est["mare"], mare_lo=reps.mare.quantile(.025), mare_hi=reps.mare.quantile(.975),
                   tempo_m_ano=est["tempo"], tempo_lo=reps.tempo.quantile(.025), tempo_hi=reps.tempo.quantile(.975),
                   frac_reps_validas=reps.notna().all(1).mean())
        linhas.append(lin)
        print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in lin.items()})
    pd.DataFrame(linhas).to_csv(OUT / "empilhado_resumo.csv", index=False)
    pd.concat(curvas).to_parquet(OUT / "empilhado_perfis_beta.parquet", index=False)


if __name__ == "__main__":
    main(sys.argv[1:] or ["Cosgrove", "Pine_Island", "Dotson", "Crosson", "Thwaites"])
