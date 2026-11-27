"""Mélange e fraturas da Thwaites, passo 11: a geleira respondeu à evacuação do mélange de 2024?

Teste com velocidade: compostos ANUAIS do ITS_LIVE (v2, atualização de setembro de 2025; 120 m,
EPSG:3031), lidos direto do S3 público em Zarr. Três blocos de 100 km cobrem o tronco oeste, a
língua (TWGT), o mélange e a TEIS.

Desenho
  1. Linha central do tronco: para cada coluna x (passo 5 km, faixa de ±2,5 km) toma-se o y de
     velocidade máxima da mediana 2019–2023 em y ∈ [−505, −440] km; a velocidade do ano é a mediana
     numa janela ±3 km em torno desse ponto.
  2. Referência = média 2019–2023 (antes do evento de fev–mar 2024). Variabilidade natural = desvio
     padrão interanual 2016–2023 em cada ponto da linha.
  3. Anomalias de 2024 e 2025 em % e em unidades de desvio padrão (z), por ponto e por trecho:
     língua/mélange (x < −1560 km), transição (−1560 a −1535 km), tronco (−1535 a −1500 km) e
     montante (−1495 a −1435 km).
  4. TEIS (controle): mediana numa caixa fixa da plataforma oriental, por ano.

Cuidados
  - O rótulo de tempo do composto (AAAA-01-01) é tratado como o ano civil AAAA; o de 2026 tem
    poucos pares e é descartado.
  - A linha de aterramento NÃO é traçada aqui (sem BedMachine/MEaSUREs neste passo): os trechos são
    definidos por posição ao longo do fluxo. Sobreponha a linha de aterramento antes de publicar.
  - Velocidades no mélange e na língua após 2024 misturam gelo da língua e blocos soltos; o sinal
    ali não é interpretável como aceleração da geleira.

Saída: outputs/melange/velocidade_linha_central.csv, velocidade_trechos.csv, velocidade_teis.csv
       e a cópia resumida notebooks/dados_resumo/melange_resposta_velocidade_itslive.csv.
Uso:   python pipelines/melange/m11_resposta_velocidade_itslive.py [--cache DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "melange"
RESUMO = ROOT / "notebooks" / "dados_resumo" / "melange_resposta_velocidade_itslive.csv"

BASE_URL = ("https://its-live-data.s3.amazonaws.com/composites/annual/v2-updated-september2025/"
            "S70W100/ITS_LIVE_velocity_EPSG3031_120m_X{x}_Y{y}.zarr")
BLOCOS = {"NW": (-1550000, -450000), "NE": (-1450000, -450000), "SW": (-1550000, -550000)}
ANOS = list(range(2016, 2026))
REF = [2019, 2020, 2021, 2022, 2023]
VAR_NAT = list(range(2016, 2024))
RES = 120.0
TRECHOS = {"lingua_melange": (-1600, -1560.1), "transicao": (-1560, -1535.1),
           "tronco": (-1535, -1500), "montante": (-1495, -1435)}
CAIXA_TEIS = dict(x=(-1595e3, -1565e3), y=(-432e3, -410e3))


def carrega_bloco(nome, x, y, cache: Path | None):
    if cache is not None and (cache / f"{nome}.npz").exists():
        d = np.load(cache / f"{nome}.npz")
        return {k: d[k] for k in d.files}
    import xarray as xr
    ds = xr.open_dataset(BASE_URL.format(x=x, y=y), engine="zarr", chunks=None)
    sub = ds[["v", "v_error"]].sel(time=slice(f"{ANOS[0]}-01-01", f"{ANOS[-1]}-12-31")).load()
    d = dict(x=sub.x.values, y=sub.y.values,
             ano=pd.DatetimeIndex(sub.time.values).year.to_numpy(),
             v=sub.v.values.astype("float32"), v_error=sub.v_error.values.astype("float32"))
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache / f"{nome}.npz", **d)
    return d


def mosaico(blocos):
    X = np.arange(min(b["x"].min() for b in blocos.values()), max(b["x"].max() for b in blocos.values()) + 1, RES)
    Y = np.arange(max(b["y"].max() for b in blocos.values()), min(b["y"].min() for b in blocos.values()) - 1, -RES)
    V = {}
    for ano in ANOS:
        out = np.full((Y.size, X.size), np.nan)
        for b in blocos.values():
            a = b["v"][list(b["ano"]).index(ano)].astype(float)
            ix = np.rint((b["x"] - X[0]) / RES).astype(int)
            iy = np.rint((Y[0] - b["y"]) / RES).astype(int)
            sub = out[np.ix_(iy, ix)]
            m = np.isnan(sub)
            sub[m] = a[m]
            out[np.ix_(iy, ix)] = sub
        V[ano] = out
    return X, Y, V


def linha_central(X, Y, V):
    ref = np.nanmedian(np.stack([V[a] for a in REF]), axis=0)
    iyw = np.where((Y <= -440e3) & (Y >= -505e3))[0]
    linhas = []
    for xc in np.arange(-1595e3, -1430e3, 5e3):
        ix = np.where(np.abs(X - xc) < 2500)[0]
        with np.errstate(all="ignore"):
            prof = np.nanmean(ref[np.ix_(iyw, ix)], axis=1)
        if np.all(np.isnan(prof)):
            continue
        yc = Y[iyw][np.nanargmax(prof)]
        iy = np.where(np.abs(Y - yc) < 3000)[0]
        r = dict(x_km=xc / 1e3, y_km=yc / 1e3)
        for a in ANOS:
            r[str(a)] = float(np.nanmedian(V[a][np.ix_(iy, ix)]))
        linhas.append(r)
    df = pd.DataFrame(linhas)
    df["v_ref"] = df[[str(a) for a in REF]].mean(axis=1)
    df["dp_interanual_pct"] = df[[str(a) for a in VAR_NAT]].std(axis=1) / df.v_ref * 100
    for a in (2024, 2025):
        df[f"dif_{a}_pct"] = (df[str(a)] / df.v_ref - 1) * 100
        df[f"z_{a}"] = df[f"dif_{a}_pct"] / df.dp_interanual_pct
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=OUT / "itslive_cache")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    blocos = {n: carrega_bloco(n, x, y, args.cache) for n, (x, y) in BLOCOS.items()}
    X, Y, V = mosaico(blocos)
    df = linha_central(X, Y, V)
    df.to_csv(OUT / "velocidade_linha_central.csv", index=False)

    tr = []
    for nome, (a, b) in TRECHOS.items():
        s = df[(df.x_km >= a) & (df.x_km <= b)]
        tr.append(dict(trecho=nome, x_ini_km=a, x_fim_km=b, pontos=len(s),
                       v_ref_m_ano=s.v_ref.mean(), dp_interanual_pct=s.dp_interanual_pct.mean(),
                       dif_2024_pct=s.dif_2024_pct.mean(), dif_2025_pct=s.dif_2025_pct.mean(),
                       max_abs_z_2025=s.z_2025.abs().max()))
    tr = pd.DataFrame(tr)
    tr.to_csv(OUT / "velocidade_trechos.csv", index=False)

    ix = (X >= CAIXA_TEIS["x"][0]) & (X <= CAIXA_TEIS["x"][1])
    iy = (Y >= CAIXA_TEIS["y"][0]) & (Y <= CAIXA_TEIS["y"][1])
    teis = pd.DataFrame(dict(ano=ANOS, v_mediana_m_ano=[float(np.nanmedian(V[a][np.ix_(iy, ix)])) for a in ANOS]))
    teis.to_csv(OUT / "velocidade_teis.csv", index=False)

    resumo = pd.concat([tr.assign(tipo="trecho_tronco_oeste"),
                        teis.assign(tipo="teis_caixa")], ignore_index=True)
    resumo.to_csv(RESUMO, index=False)
    pd.set_option("display.width", 200)
    print(tr.round(2).to_string(index=False))
    print(teis.round(0).to_string(index=False))


if __name__ == "__main__":
    main()
