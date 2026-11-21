"""InSAR + ICESat-2, passo A: dh/dt euleriano junto à linha de aterramento (células de 500 m).

O produto de dh/dt do gelo apoiado exclui 5 km junto à linha; aqui é preciso exatamente essa faixa.
Entrada: data/interim/tide_response/mare_<plataforma>.parquet (ATL06 r007, 2019–2025, quatro
estações, ±15 km da linha do BedMachine). Faixa usada: dist_gl de −8 a +2 km.
Modelo por célula, sobre r = h − REMA (32 m):
    r = a + g_x·(x − x̄) + g_y·(y − ȳ) + b·(t − t̄) + c·T + ε
b = dh/dt (m/ano); T = maré CATS2008 + DAC, incluída porque a flexão começa a montante da linha;
g_x, g_y absorvem a declividade residual entre trilhas. Erro de b: sanduíche CR1 agrupado por
passagem (rgt, ciclo), porque segmentos de uma passagem compartilham erro. Exige ≥ 10 passagens e
≥ 3 anos de cobertura.
Limitações: euleriano (ondulações advectadas entram como ruído); firn não descontado (o sinal junto
à linha é dinâmico, de metros por ano, e o firn modelado é de centímetros por ano); sem GIA.
Saída: outputs/gl_migration/dhdt_junto_linha.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p01_transectos import amostra_rema  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "interim" / "tide_response"
OUT = ROOT / "outputs" / "gl_migration"
CEL = 500.0
PLAT = ["Pine_Island", "Thwaites", "Crosson", "Dotson", "Cosgrove"]


def ajuste(g):
    x = g.x.to_numpy() - g.x.mean()
    y = g.y.to_numpy() - g.y.mean()
    t = g.t.to_numpy()
    X = np.c_[np.ones(len(g)), x, y, t - t.mean(), g["T"].to_numpy()]
    r = g.r.to_numpy()
    XtX = X.T @ X
    if np.linalg.cond(XtX) > 1e10:
        return None
    inv = np.linalg.inv(XtX)
    beta = inv @ X.T @ r
    e = r - X @ beta
    # CR1 agrupado por passagem
    meat = np.zeros((5, 5))
    for _, idx in pd.Series(np.arange(len(g))).groupby(g.pass_id.to_numpy()):
        s = X[idx.values].T @ e[idx.values]
        meat += np.outer(s, s)
    G, n, k = g.pass_id.nunique(), len(g), 5
    V = inv @ meat @ inv * G / (G - 1) * (n - 1) / (n - k)
    return beta[3], np.sqrt(V[3, 3]), beta[4], float(np.sqrt(np.mean(e ** 2)))


def main():
    linhas = []
    for p in PLAT:
        d = pd.read_parquet(SRC / f"mare_{p}.parquet")
        d = d[(d.dist_gl >= -8000) & (d.dist_gl <= 2000)].copy()
        d["r"] = d.h - amostra_rema(d.x.to_numpy(), d.y.to_numpy())
        d["T"] = d.tide_cats + d.dac
        d = d[np.isfinite(d.r) & np.isfinite(d["T"])]
        d["ix"] = np.floor(d.x / CEL).astype(int)
        d["iy"] = np.floor(d.y / CEL).astype(int)
        for (ix, iy), g in d.groupby(["ix", "iy"]):
            if g.pass_id.nunique() < 10 or g.t.max() - g.t.min() < 3:
                continue
            # remove aberrantes grosseiros (> 5 MAD da mediana da célula)
            med = g.r.median()
            mad = 1.4826 * (g.r - med).abs().median() + 0.1
            g = g[(g.r - med).abs() < 5 * mad]
            if g.pass_id.nunique() < 10:
                continue
            a = ajuste(g)
            if a is None:
                continue
            linhas.append(dict(plataforma=p, x=(ix + 0.5) * CEL, y=(iy + 0.5) * CEL,
                               dist_gl_km=g.dist_gl.median() / 1e3, dhdt=a[0], se=a[1], beta=a[2],
                               rms=a[3], n_pass=g.pass_id.nunique(), n_seg=len(g)))
        print(p, sum(1 for l in linhas if l["plataforma"] == p))
    df = pd.DataFrame(linhas)
    df.to_parquet(OUT / "dhdt_junto_linha.parquet", index=False)
    print(df.groupby("plataforma")[["dhdt", "se", "rms"]].median().round(2))


if __name__ == "__main__":
    main()
