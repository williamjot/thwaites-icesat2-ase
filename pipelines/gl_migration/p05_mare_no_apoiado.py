"""Migração da linha de aterramento — passo 5: resposta à maré dentro do gelo "apoiado".

Pista registrada no passo 3 do plano basal: células a montante da linha do BedMachine com β
significativo. Pergunta: é recuo real da linha (ou intrusão de água do mar sob o gelo apoiado),
ou falso positivo?

Entrada: outputs/tide_response_ase/beta_lagr_com_dist_LA_recente.parquet (β Lagrangiano por célula de
500 m; erro CR1 agrupado por passagem) e as linhas InSAR datadas (NSIDC-0498 v2.1, recorte ASE).

Critério de célula "apoiada com resposta" (fixado antes do teste): dist_gl ≤ −1 km (a montante da
linha do BedMachine), n_pass ≥ 12, rms ≤ 1 m, β − 2·σβ > 0,3.
Testes:
 1. Coerência espacial: fração de células significativas com pelo menos 2 vizinhas (8-conectividade)
    também significativas, contra 2.000 permutações do rótulo entre as células apoiadas elegíveis da
    mesma plataforma.
 2. Posição em relação à linha InSAR mais recente (2019–2025) de cada trecho: a célula está a
    jusante dela (gelo que já flutua segundo o InSAR) ou a montante (resposta à maré em gelo que o
    InSAR dá como apoiado)? "A jusante" = a linha InSAR mais próxima (≤ 5 km) tem dist_gl do
    BedMachine menor (mais para o interior) que a célula.
Saída: outputs/gl_migration/mare_no_apoiado_celulas.parquet e mare_no_apoiado_resumo.csv
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "gl_migration"
RNG = np.random.default_rng(20261117)
CEL = 500.0


_VIZ = {}


def vizinhas_sig(df, raio=1500.0):
    """Número de células significativas num raio de 1,5 km (as células só existem ao longo das
    trilhas, então a vizinhança imediata da grade quase nunca existe)."""
    chave = (len(df), float(df.x.iloc[0]), float(df.y.iloc[0]))
    if chave not in _VIZ:
        t = cKDTree(np.c_[df.x, df.y])
        _VIZ[chave] = [np.array([j for j in v if j != i]) for i, v in enumerate(t.query_ball_point(np.c_[df.x, df.y], raio))]
    viz = _VIZ[chave]
    s = df.sig.to_numpy()
    return np.array([s[v].sum() if len(v) else 0 for v in viz])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(ROOT / "outputs" / "tide_response_ase" / "beta_lagr_com_dist_LA_recente.parquet")
    eleg = d[(d.dist_gl_km <= -1) & (d.n_pass >= 12) & (d.rms_m <= 1.0) & np.isfinite(d.se_beta)].copy()
    eleg["sig"] = (eleg.beta - 2 * eleg.se_beta) > 0.3
    # linhas InSAR 2019–2025: vértices com o ano
    gl = gpd.read_file(ROOT / "data" / "grounding" / "processed" / "InSAR_GL_ASE_v02.1.gpkg")
    gl = gl[gl.Year.astype(int) >= 2019].explode(index_parts=False)
    vx, vy, vyear = [], [], []
    for geom, yr in zip(gl.geometry, gl.Year.astype(int)):
        xs, ys = np.asarray(geom.coords.xy)
        vx += list(xs); vy += list(ys); vyear += [yr] * len(xs)
    vx, vy, vyear = map(np.asarray, (vx, vy, vyear))
    # dist_gl do BedMachine nos vértices InSAR, pelo vizinho mais próximo na grade de células
    tall = cKDTree(np.c_[d.x, d.y])
    dd, ii = tall.query(np.c_[vx, vy])
    vdist = np.where(dd < 750, d.dist_gl_km.to_numpy()[ii], np.nan)
    tv = cKDTree(np.c_[vx, vy])
    linhas = []
    for p, g in eleg.groupby("shelf"):
        g = g.copy()
        g["nviz"] = vizinhas_sig(g)
        sig = g.sig.to_numpy()
        obs = np.mean(g.nviz[sig] >= 1) if sig.any() else np.nan
        perm = []
        for _ in range(2000):
            gp = g.assign(sig=RNG.permutation(sig))
            nv = vizinhas_sig(gp) if _ < 1000 else None
            if nv is None:
                break
            perm.append(np.mean(nv[gp.sig.to_numpy()] >= 1) if sig.any() else np.nan)
        perm = np.array(perm)
        # posição relativa à linha InSAR recente
        dv, iv = tv.query(np.c_[g.x, g.y], k=20, distance_upper_bound=5000)
        jus, mont, sem = 0, 0, 0
        estado = []
        for k in range(len(g)):
            ok = np.isfinite(dv[k])
            if not ok.any():
                estado.append("sem_InSAR"); continue
            idx = iv[k][ok]
            dl = np.nanmin(vdist[idx]) if np.isfinite(vdist[idx]).any() else np.nan
            if not np.isfinite(dl):
                estado.append("sem_InSAR")
            elif dl < g.dist_gl_km.iloc[k]:
                estado.append("jusante_InSAR")
            else:
                estado.append("montante_InSAR")
        g["estado"] = estado
        linhas.append(dict(plataforma=p, elegiveis=len(g), significativas=int(sig.sum()),
                           frac_sig=sig.mean(), coerencia_obs=obs,
                           coerencia_perm_p95=np.nanpercentile(perm, 95) if len(perm) else np.nan,
                           p_coerencia=(np.sum(perm >= obs) + 1) / (len(perm) + 1) if len(perm) else np.nan,
                           sig_jusante_InSAR=int(((g.estado == "jusante_InSAR") & g.sig).sum()),
                           sig_montante_InSAR=int(((g.estado == "montante_InSAR") & g.sig).sum()),
                           sig_sem_InSAR=int(((g.estado == "sem_InSAR") & g.sig).sum()),
                           dist_mediana_sig_km=float(g.loc[g.sig, "dist_gl_km"].median())))
        eleg.loc[g.index, "estado"] = g.estado
        eleg.loc[g.index, "nviz"] = g.nviz
        print(linhas[-1])
    eleg.to_parquet(OUT / "mare_no_apoiado_celulas.parquet", index=False)
    pd.DataFrame(linhas).to_csv(OUT / "mare_no_apoiado_resumo.csv", index=False)


if __name__ == "__main__":
    main()
