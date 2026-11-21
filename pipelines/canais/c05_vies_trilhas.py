"""Canais basais, passo 5: taxa lagrangiana com desvio próprio de cada seção (viés entre trilhas).

Problema (passo 4): na PIG, pares da mesma trilha dão +0,26 m/a e pares de trilhas diferentes −0,13.
Cada seção (trilha × par de feixes × canal) pode medir D com um viés próprio: ângulo de corte,
quilhas a distâncias diferentes, canais vizinhos dentro da janela. Modelo:
    ΔD_ij = o_{s(j)} − o_{s(i)} + r·Δt_ij + ε
o_s: desvio da seção s (média zero por componente conexa — só diferenças são identificáveis);
r: taxa lagrangiana comum. r é identificado pelos pares da mesma seção e pelos ciclos fechados da
rede; com o desvio por seção, um viés entre trilhas não entra em r.
Estimação: Huber (IRLS); IC por bootstrap em blocos de 10 km (300). Por plataforma e por classe de
velocidade (< 1 km/a; >= 1 km/a), pois o passo 4 mostrou que o tronco rápido da PIG não empareia bem.
Uso: python c05_vies_trilhas.py [sufixo dos pares, ex. _sint05]
Saída: outputs/canais/lagrangiano_efeito_secao{suf}.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "canais"
RNG = np.random.default_rng(20261117)
K_HID = 1027.0 / (1027.0 - 917.0)
SUF = sys.argv[1] if len(sys.argv) > 1 else ""
NB = 300


def huber_irls(X, y, k=1.345, it=30):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(it):
        r = y - X @ b
        s = 1.4826 * np.median(np.abs(r - np.median(r))) or 1e-6
        u = np.abs(r) / (k * s)
        w = np.where(u <= 1, 1.0, 1.0 / np.maximum(u, 1e-12))
        sw = np.sqrt(w)
        bn = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        if np.allclose(bn, b, atol=1e-7):
            break
        b = bn
    return b


def taxa_fe(g):
    sec = pd.Index(sorted(set(g.i) | set(g.j)))
    ii, jj = sec.get_indexer(g.i), sec.get_indexer(g.j)
    n = len(g)
    X = np.zeros((n, len(sec) + 1))
    X[np.arange(n), jj] += 1.0
    X[np.arange(n), ii] -= 1.0
    X[:, -1] = g.dt.to_numpy()
    # restrição de soma zero dos desvios (resolve a indeterminação); lstsq dá a solução de norma mínima
    return huber_irls(X, g.dD.to_numpy())[-1]


def main():
    p = pd.read_parquet(OUT / f"pares_lagrangianos{SUF}.parquet")
    p["vel"] = p.deslocamento / p.dt
    p["classe"] = np.where(p.vel < 1000, "lento <1 km/a", "rápido >=1 km/a")
    p["bloco"] = (p.x // 10000).astype(int).astype(str) + "_" + (p.y // 10000).astype(int).astype(str)
    p["mesma_secao"] = p.i == p.j
    linhas = []
    for (pl, cl), g in p.groupby(["plataforma", "classe"]):
        if len(g) < 50:
            continue
        r = taxa_fe(g)
        same = g[g.mesma_secao]
        r_same = huber_irls(same[["dt"]].to_numpy(), same.dD.to_numpy())[0] if len(same) > 20 else np.nan
        r_pool = huber_irls(g[["dt"]].to_numpy(), g.dD.to_numpy())[0]
        bl = g.bloco.unique()
        reps = []
        for _ in range(NB):
            q = pd.concat([g[g.bloco == b].assign(i=lambda d, k=k: d.i + f"#{k}", j=lambda d, k=k: d.j + f"#{k}")
                           for k, b in enumerate(RNG.choice(bl, len(bl)))])
            try:
                reps.append(taxa_fe(q))
            except Exception:
                pass
        linhas.append(dict(plataforma=pl, classe=cl, n_pares=len(g), n_secoes=len(set(g.i) | set(g.j)),
                           n_mesma_secao=len(same), n_blocos=len(bl), taxa_fe=r,
                           lo=np.percentile(reps, 2.5), hi=np.percentile(reps, 97.5),
                           taxa_mesma_secao=r_same, taxa_agrupada=r_pool))
    d = pd.DataFrame(linhas)
    for c in ("taxa_fe", "lo", "hi"):
        d[c + "_H"] = d[c] * K_HID
    d.to_csv(OUT / f"lagrangiano_efeito_secao{SUF}.csv", index=False)
    with pd.option_context("display.width", 250):
        print(d.round(3).to_string())


if __name__ == "__main__":
    main()
