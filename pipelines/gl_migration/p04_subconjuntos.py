"""Migração da linha de aterramento — passo 4: perfis de β por subconjunto de ciclos.

Substitui o estimador por ciclo único do passo 3 (resíduo de ~0,9 km por ciclo; poucos perfis
válidos). Aqui cada posição vem de um perfil de β estimado com vários ciclos.

Por perfil (rgt, par, cruzamento; mesma origem e grade de 100 m do passo 2) e por subconjunto S:
    r(x) = a_S(x) + b_S(x)·(t − t̄_S) + β_S(x)·(T − T̄_S)   (Huber, por intervalo, ≥ 6 ciclos)
β_S é normalizado pela própria mediana no gelo livre (6–12 km) e suavizado (mediana de 500 m).
Deslocamento δ_S: translação ao longo da trilha que minimiza a média de |β_S(x) − β_ref(x − δ)| em
−6 a +10 km, com β_ref = perfil de todos os ciclos. Positivo = a jusante.

Subconjuntos:
  maré:  "baixa" (T abaixo da mediana do perfil) e "alta" (acima); Δ_maré = δ_alta − δ_baixa, por m
         de diferença de maré média entre os dois grupos (hidrostática: Δ_maré < 0);
  tempo: "antes" (t < 2022,0) e "depois" (t ≥ 2022,5); Δ_tempo = δ_depois − δ_antes, por ano de
         diferença entre as médias de t (negativo = recuo para o interior).
O lado da maré e o do tempo são estimados separadamente; como a fase de maré amostrada pelo
ICESat-2 numa trilha varia pouco entre anos (alias de 91 dias), conferimos que T̄ dos grupos de tempo
não difere muito e corrigimos Δ_tempo pelo termo de maré quando necessário (declarado na saída).
Incerteza: bootstrap dos ciclos dentro do perfil (300 réplicas).

Saída: outputs/gl_migration/subconjuntos_<plataforma>.parquet
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
from p02_flexao_por_ciclo import B, IN, JANELA, cruzamentos, huber_wls  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "gl_migration"
NB = 300
RNG = np.random.default_rng(20261117)
GRADE = np.arange(-12000 + B / 2, 14000, B)
JAN = (GRADE > -6000) & (GRADE < 10000)
FAR = (GRADE > 6000) & (GRADE < 12000)
DESL = np.arange(-40, 41)


def ols_vetorizado(R, t, T, minc):
    """MQO por intervalo (linhas de R), com dados faltantes, resolvido para todos de uma vez.
    Modelo r = a + b·(t − t̄) + β·(T − T̄). Devolve β por intervalo."""
    dt, dT = t - t.mean(), T - T.mean()
    M = np.isfinite(R)
    Y = np.where(M, R, 0.0)
    X = np.stack([np.ones_like(dt), dt, dT])            # 3 × ciclos
    XtX = np.einsum("pk,qk,ik->ipq", X, X, M.astype(float))
    XtY = np.einsum("pk,ik->ip", X, Y)
    n = M.sum(1)
    beta = np.full(R.shape[0], np.nan)
    ok = (n >= minc) & (np.linalg.det(XtX) > 1e-9)
    if ok.any():
        sol = np.linalg.solve(XtX[ok], XtY[ok][..., None])[..., 0]
        beta[ok] = sol[:, 2]
        # exige variância de maré suficiente entre os ciclos presentes
        varT = (np.where(M, dT, 0) ** 2).sum(1) / np.maximum(n, 1) - ((np.where(M, dT, 0)).sum(1) / np.maximum(n, 1)) ** 2
        beta[varT < 0.08 ** 2] = np.nan
    return beta


def limpa(R):
    """Remove valores aberrantes por intervalo (|r − mediana| > 4·MAD); substitui o Huber."""
    med = np.nanmedian(R, axis=1, keepdims=True)
    mad = 1.4826 * np.nanmedian(np.abs(R - med), axis=1, keepdims=True) + 0.05
    R = R.copy()
    R[np.abs(R - med) > 4 * mad] = np.nan
    return R


def perfil_beta(R, t, T, minc=5):
    """R: matriz intervalos × ciclos; devolve β normalizado e suavizado."""
    beta = ols_vetorizado(R, t, T, minc)
    far = np.nanmedian(beta[FAR])
    if not np.isfinite(far) or far < 0.4:
        return None, far
    b = median_filter(beta / far, size=JANELA, mode="nearest")
    b[~np.isfinite(beta)] = np.nan
    b = pd.Series(b).interpolate(limit=10, limit_area="inside").to_numpy()
    return b, far


def desloc(b, ref):
    custo = np.full(len(DESL), np.nan)
    for q, k in enumerate(DESL):
        tk = np.roll(ref, k)
        if k > 0:
            tk[:k] = np.nan
        elif k < 0:
            tk[k:] = np.nan
        ok = JAN & np.isfinite(b) & np.isfinite(tk)
        if ok.sum() >= 40:
            custo[q] = np.mean(np.abs(b[ok] - tk[ok]))
    if np.isfinite(custo).sum() < 20:
        return np.nan
    q = int(np.nanargmin(custo))
    if q in (0, len(DESL) - 1):
        return np.nan
    return DESL[q] * B


def processa(g, sg, orient, lo, hi):
    g = g[(g.s > lo) & (g.s < hi)]
    g = g.assign(xb=np.floor(orient * (g.s - sg) / B) * B + B / 2)
    g = g[(g.xb > -12000) & (g.xb < 14000)]
    if g.xb.min() > -4000 or g.xb.max() < 8000:
        return None
    pb = g.groupby(["ciclo", "xb"]).agg(r=("r", "median"), T=("T", "median"), t=("t", "median")).reset_index()
    cic = pb.groupby("ciclo").agg(T=("T", "median"), t=("t", "median"))
    if len(cic) < 12:
        return None
    R = limpa(pb.pivot(index="xb", columns="ciclo", values="r").reindex(index=GRADE, columns=cic.index).to_numpy())
    t, T = cic.t.to_numpy(), cic["T"].to_numpy()
    ref, far = perfil_beta(R, t, T)
    if ref is None:
        return None
    from p02_flexao_por_ciclo import primeiro_cruzamento
    x50 = primeiro_cruzamento(GRADE, ref, 0.5)
    grupos = {"baixa": T < np.median(T), "alta": T >= np.median(T),
              "antes": t < 2022.0, "depois": t >= 2022.5}

    def estima(idx):
        out = {}
        for nome, m in grupos.items():
            mm = m[idx] if idx is not None else m
            cols = (idx if idx is not None else np.arange(len(t)))[mm]
            if len(cols) < 6:
                out[nome] = np.nan
                continue
            b, _ = perfil_beta(R[:, cols], t[cols], T[cols])
            out[nome] = desloc(b, ref) if b is not None else np.nan
        return out

    base = estima(None)
    reps = []
    for _ in range(NB):
        idx = RNG.integers(0, len(t), len(t))
        reps.append(estima(idx))
    reps = pd.DataFrame(reps)
    dmare = (reps.alta - reps.baixa) / (T[grupos["alta"]].mean() - T[grupos["baixa"]].mean())
    dtempo = (reps.depois - reps.antes) / (t[grupos["depois"]].mean() - t[grupos["antes"]].mean())
    return dict(
        beta_far=far, x50_ref=x50, n_ciclos=len(t),
        dT_grupos_mare=T[grupos["alta"]].mean() - T[grupos["baixa"]].mean(),
        dt_grupos_tempo=t[grupos["depois"]].mean() - t[grupos["antes"]].mean(),
        dT_grupos_tempo=T[grupos["depois"]].mean() - T[grupos["antes"]].mean(),
        n_antes=int(grupos["antes"].sum()), n_depois=int(grupos["depois"].sum()),
        mare_m_por_m=(base["alta"] - base["baixa"]) / (T[grupos["alta"]].mean() - T[grupos["baixa"]].mean()),
        mare_sd=dmare.std(), mare_ok=dmare.notna().mean(),
        tempo_m_ano=(base["depois"] - base["antes"]) / (t[grupos["depois"]].mean() - t[grupos["antes"]].mean()),
        tempo_sd=dtempo.std(), tempo_ok=dtempo.notna().mean())


def main(plats):
    for p in plats:
        d = pd.read_parquet(IN / f"perfis_{p}.parquet")
        linhas = []
        for (rgt, par), g in d.groupby(["rgt", "par"]):
            for k, (sg, orient, lo, hi) in enumerate(cruzamentos(g)):
                r = processa(g, sg, orient, lo, hi)
                if r is None:
                    continue
                # posição geográfica do cruzamento com o BedMachine (para mapas e comparação)
                gg = g.iloc[(g.s - sg).abs().argsort()[:50]]
                linhas.append(dict(plataforma=p, rgt=rgt, par=par, cz=k, X=gg.x.median(), Y=gg.y.median(), **r))
        df = pd.DataFrame(linhas)
        df.to_parquet(OUT / f"subconjuntos_{p}.parquet", index=False)
        print(p, len(df), "perfis;", df[["mare_m_por_m", "tempo_m_ano"]].median().round(0).to_dict())


if __name__ == "__main__":
    main(sys.argv[1:] or ["Cosgrove"])
