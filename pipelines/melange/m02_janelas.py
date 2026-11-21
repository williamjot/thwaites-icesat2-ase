"""Mélange e fraturas da Thwaites, passo 2: janelas de 1 km ao longo das trilhas e nível da água.

Entrada: data/interim/melange/thwaites_atl06.parquet (passo 1).
1. Nível da água por campanha (estação × ano): moda do hf nos segmentos sobre o oceano do BedMachine
   v4.1 entre −5 e +3 m (histograma de 0,1 m). hfw = hf − nível: altura acima da água local
   (absorve a topografia dinâmica média e o erro de geoide de grande escala).
2. Janela = (grânulo, feixe, floor(x_atc / 1000 m)). ATL06 tem um segmento a cada 20 m, então cabem
   50 por janela. Por janela:
     n            segmentos com hf (qualquer qualidade)
     f_ret        n / 50 (fração com retorno; falta de retorno = nuvem, fenda íngreme ou sombra)
     f_grosso     fração com hfw ≥ LIM_GROSSO (gelo de plataforma ou bloco de iceberg)
     f_agua       fração com hfw < LIM_AGUA (água aberta ou gelo marinho fino)
     hfw_med      mediana do hfw
     rug          desvio robusto (1,4826·MAD) do hfw dos segmentos grossos, depois de tirar a reta
                  da janela (rugosidade de curto comprimento de onda; fraturas e blocos)
     trans        número de transições grosso ↔ não grosso ao longo da janela (fragmentação)
     x, y, t      posição média e tempo (ano juliano desde 2018), est, ano da campanha
     bm           classe do BedMachine no centro (0 oceano, 2 apoiado, 3 flutuante)
LIM_GROSSO e LIM_AGUA saem do vale e do pico do histograma do hfw sobre o oceano (impressos aqui);
o passo 3 testa a sensibilidade a eles.
Saída: data/interim/melange/janelas_1km.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "data" / "interim" / "melange"
BM = ROOT / "data" / "NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc"


def bm_classe(x, y):
    m = xr.open_dataset(BM)["mask"].sel(x=slice(-1660e3, -1460e3), y=slice(-360e3, -560e3))
    return m.sel(x=xr.DataArray(x, dims="p"), y=xr.DataArray(y, dims="p"), method="nearest").values


def campanha(t, est):
    # DJF de dez/ano conta para o ano seguinte; t é juliano (diferença < 12 h, irrelevante aqui)
    ano = np.floor(t).astype(int)
    frac = t - ano
    ano = np.where((est == 1) & (frac > 0.8), ano + 1, ano)
    return ano


def main():
    d = pd.read_parquet(DIR / "thwaites_atl06.parquet")
    d["bm"] = bm_classe(d.x.to_numpy(), d.y.to_numpy()).astype(np.int8)
    d["ano"] = campanha(d.t.to_numpy(), d.est.to_numpy())

    # 1. nível da água por campanha
    oc = d[(d.bm == 0) & (d.hf > -5) & (d.hf < 3)]
    niv = {}
    for (a, e), g in oc.groupby(["ano", "est"]):
        if len(g) < 5000:
            continue
        h, b = np.histogram(g.hf, bins=np.arange(-5, 3.01, 0.1))
        k = np.argmax(h)
        niv[(a, e)] = 0.5 * (b[k] + b[k + 1])
    nv = pd.Series(niv, name="nivel").rename_axis(["ano", "est"]).reset_index()
    print("nível da água por campanha (m acima do geoide): mediana", nv.nivel.median().round(2),
          "| p05–p95", nv.nivel.quantile([0.05, 0.95]).round(2).tolist(), "| n", len(nv))
    d = d.merge(nv, on=["ano", "est"], how="left")
    d["nivel"] = d.nivel.fillna(nv.nivel.median())
    d["hfw"] = d.hf - d.nivel

    # limiares a partir do histograma sobre o oceano
    # sem clip: valores fora do intervalo ficam de fora (um clip empilha tudo no último bin)
    h, b = np.histogram(d.loc[d.bm == 0, "hfw"], bins=np.arange(-5, 80.5, 1.0))
    c = 0.5 * (b[1:] + b[:-1])
    sel = (c > 20) & (c < 75)
    pico_alto = c[sel][np.argmax(h[sel])]
    vale = c[(c > 3) & (c < pico_alto)][np.argmin(h[(c > 3) & (c < pico_alto)])]
    LIM_GROSSO = float(vale)
    LIM_AGUA = 2.0      # água + gelo marinho: ~90% do pico baixo fica abaixo (impresso abaixo)
    print(f"pico de gelo grosso sobre o oceano: {pico_alto:.1f} m; vale: {vale:.1f} m -> LIM_GROSSO = {LIM_GROSSO}")
    baixo = d.loc[(d.bm == 0) & (d.hfw < LIM_GROSSO), "hfw"]
    print(f"fração do pico baixo abaixo de {LIM_AGUA} m: {(baixo < LIM_AGUA).mean():.3f}")

    # 2. janelas de 1 km
    d["j"] = np.floor(d.xatc / 1000.0).astype(np.int64)
    d["grosso"] = d.hfw >= LIM_GROSSO
    d["agua"] = d.hfw < LIM_AGUA
    d = d.sort_values(["gran", "beam", "xatc"])
    chave = ["gran", "beam", "j"]
    g = d.groupby(chave, sort=False)
    jan = g.agg(n=("hfw", "size"), f_grosso=("grosso", "mean"), f_agua=("agua", "mean"),
                hfw_med=("hfw", "median"), x=("x", "mean"), y=("y", "mean"), t=("t", "mean"),
                est=("est", "first"), ano=("ano", "first"), rgt=("rgt", "first"), cycle=("cycle", "first"),
                f_qs0=("qs", lambda s: (s == 0).mean()))
    # transições grosso <-> não grosso
    mud = d.grosso.astype(np.int8).diff().abs()
    novo = (d[chave] != d[chave].shift()).any(axis=1)
    mud[novo] = 0
    jan["trans"] = mud.groupby([d[k] for k in chave], sort=False).sum().to_numpy()

    # rugosidade: resíduo da reta nos segmentos grossos
    gr = d[d.grosso]

    def rug(s):
        if len(s) < 8:
            return np.nan
        xa = s.xatc.to_numpy(); hh = s.hfw.to_numpy()
        p = np.polyfit(xa - xa.mean(), hh, 1)
        r = hh - np.polyval(p, xa - xa.mean())
        return 1.4826 * np.median(np.abs(r - np.median(r)))

    rg = gr.groupby(chave, sort=False)[["xatc", "hfw"]].apply(rug).rename("rug")
    jan = jan.join(rg)
    jan["f_ret"] = (jan.n / 50.0).clip(upper=1.0)
    jan = jan.reset_index()
    jan["bm"] = bm_classe(jan.x.to_numpy(), jan.y.to_numpy()).astype(np.int8)
    jan.attrs = {}
    jan.to_parquet(DIR / "janelas_1km.parquet", index=False)
    pd.Series({"LIM_GROSSO": LIM_GROSSO, "LIM_AGUA": LIM_AGUA}).to_json(DIR / "limiares.json")
    nv.to_csv(DIR / "nivel_agua_campanha.csv", index=False)
    print(f"{len(jan):,} janelas; por classe BedMachine:", jan.groupby("bm").size().to_dict())


if __name__ == "__main__":
    main()
