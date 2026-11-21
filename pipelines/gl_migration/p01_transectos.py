"""Migração da linha de aterramento (ASE, ICESat-2) — passo 1: perfis repetidos por par de feixes.

Entrada: data/interim/tide_response/mare_<plataforma>.parquet (segmentos ATL06 r007 a ±15 km da linha
do BedMachine; colunas x, y, h, t, pass_id = rgt*1000 + ciclo, dac, dist_gl, tide_cats).
Os segmentos não trazem o identificador do feixe; os três pares de feixes de cada trilha de
referência (RGT) são separados pela coordenada transversal (pares a ~3,3 km, feixes do mesmo par a
~90 m e desvio de repetição < ~100 m): agrupamos por lacunas > 1 km. O par é tratado como um perfil.

Para cada perfil (rgt, par): coordenada ao longo da trilha s (eixo principal dos pontos), orientada
para s crescer do gelo apoiado para o flutuante (dist_gl cresce com s); r = h − REMA (mosaico 32 m)
remove a topografia estática; maré T = tide_cats + dac (deslocamento da superfície do mar).

Saída: data/interim/gl_migration/perfis_<plataforma>.parquet com
    rgt, par, ciclo, t, s, c (transversal), r, T, dist_gl
Nada aqui é resultado; é a organização dos dados para os passos seguintes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "interim" / "tide_response"
OUT = ROOT / "data" / "interim" / "gl_migration"
REMA = ROOT / "data" / "REMA_ase_32m.tif"
GAP_PAR = 1000.0       # m: lacuna transversal que separa pares de feixes
MIN_PTS = 2000         # pontos mínimos por perfil


def amostra_rema(x, y):
    """Vizinho mais próximo no mosaico de 32 m, lendo só a janela que cobre os pontos."""
    from rasterio.windows import from_bounds
    with rasterio.open(REMA) as r:
        w = from_bounds(x.min() - 64, y.min() - 64, x.max() + 64, y.max() + 64, r.transform).round_offsets().round_lengths()
        a = r.read(1, window=w).astype(np.float32)
        tr = r.window_transform(w)
        nod = r.nodata
    col = np.floor((x - tr.c) / tr.a).astype(int)
    row = np.floor((y - tr.f) / tr.e).astype(int)
    ok = (row >= 0) & (row < a.shape[0]) & (col >= 0) & (col < a.shape[1])
    z = np.full(len(x), np.nan)
    z[ok] = a[row[ok], col[ok]]
    z[z == nod] = np.nan
    return z


def perfis(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for rgt, g in df.groupby("rgt"):
        xy = g[["x", "y"]].to_numpy()
        c0 = xy.mean(0)
        u, sv, vt = np.linalg.svd(xy - c0, full_matrices=False)
        ax, tr = vt[0], vt[1]
        s = (xy - c0) @ ax
        c = (xy - c0) @ tr
        ordem = np.argsort(c)
        cs = c[ordem]
        cortes = np.where(np.diff(cs) > GAP_PAR)[0]
        rotulo = np.zeros(len(c), int)
        lim = np.r_[-np.inf, (cs[cortes] + cs[cortes + 1]) / 2, np.inf]
        rotulo = np.searchsorted(lim, c) - 1
        for p in np.unique(rotulo):
            m = rotulo == p
            if m.sum() < MIN_PTS:
                continue
            gg = g.loc[m.nonzero()[0] + 0 if False else g.index[m]]
            sp = s[m]
            # orientação: s cresce do apoiado (dist_gl < 0) para o flutuante
            if np.corrcoef(sp, gg["dist_gl"].to_numpy())[0, 1] < 0:
                sp = -sp
            out.append(pd.DataFrame({
                "rgt": rgt, "par": int(p), "ciclo": (gg["pass_id"] % 1000).to_numpy(),
                "t": gg["t"].to_numpy(), "s": sp, "c": c[m] - np.median(c[m]),
                "h": gg["h"].to_numpy(), "x": gg["x"].to_numpy(), "y": gg["y"].to_numpy(),
                "T": (gg["tide_cats"] + gg["dac"]).to_numpy(), "dist_gl": gg["dist_gl"].to_numpy()}))
    return pd.concat(out, ignore_index=True)


def main(plataformas):
    OUT.mkdir(parents=True, exist_ok=True)
    for p in plataformas:
        df = pd.read_parquet(SRC / f"mare_{p}.parquet")
        df["rgt"] = df["pass_id"] // 1000
        pf = perfis(df)
        pf["r"] = pf["h"] - amostra_rema(pf["x"].to_numpy(), pf["y"].to_numpy())
        pf = pf[np.isfinite(pf["r"]) & np.isfinite(pf["T"])]
        pf.to_parquet(OUT / f"perfis_{p}.parquet", index=False)
        n = pf.groupby(["rgt", "par"]).agg(n=("r", "size"), ciclos=("ciclo", "nunique"))
        print(p, "perfis", len(n), "pontos", len(pf), "ciclos/perfil mediana", n["ciclos"].median())


if __name__ == "__main__":
    main(sys.argv[1:] or ["Pine_Island", "Thwaites", "Crosson", "Dotson", "Cosgrove"])
