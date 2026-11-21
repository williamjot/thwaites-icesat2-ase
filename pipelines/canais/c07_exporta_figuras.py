"""Canais basais, passo 7: exporta para CSV o que as figuras em R dos canais precisam (sem recalcular resultados).

Saídas em manuscrito/figuras/dados/canais/:
  pig_rema_hp.csv     REMA 32 m (data/REMA_ase_32m.tif) em blocos 4×4 (128 m), passa-alta só no domínio hidrostático
                      (menos gaussiana σ = 2 km) — só para mostrar a forma dos canais na superfície; ‡ exibição
  pig_mascara.csv     máscara hidrostática 500 m (classe, plataforma) na janela da PIG
  pig_vel.csv         ITS_LIVE v2 anual, média 2019–2025 de vx, vy, em blocos de 1 km
  pig_secoes.csv      seções de canal da PIG (passo 2b), agregadas por chave; critério estrito e relaxado
  taxas_plataformas.csv  taxa lagrangiana com desvio por seção (passo 5) nas variantes R400, R200, R800, relaxada,
                      e a recuperação do teste sintético (+0,5 m/a injetado)
  lowery_resumo.csv   contraste canal − quilha no produto de Lowery et al. (2025) com IC bootstrap em blocos (passo 6)
  lowery_secoes.csv   comparação por seção (passo 6)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.windows import from_bounds
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "manuscrito" / "figuras" / "dados" / "canais"
CAN = ROOT / "outputs" / "canais"
RNG = np.random.default_rng(20261117)
K_HID = 1027.0 / (1027.0 - 917.0)
X0, X1, Y0, Y1 = -1680e3, -1555e3, -350e3, -250e3   # janela da PIG (EPSG:3031)


def chave(e):
    par = (e.beam + 1) // 2
    return e.plataforma + "_" + e.rgt.astype(str) + "_" + par.astype(str) + "_" + (e.b0 // 10).astype(str)


def hidrostatico(xs, ys):
    """Classe da máscara hidrostática (500 m; 1 = hidrostático, o domínio de detecção dos canais, passo 1)
    no ponto mais próximo de cada (x, y)."""
    m = xr.open_dataset(ROOT / "data" / "interim" / "hydrostatic_mask_500m.nc")["classe"]
    m = m.sel(x=slice(X0 - 1e3, X1 + 1e3), y=slice(Y1 + 1e3, Y0 - 1e3))
    return m.sel(x=xr.DataArray(xs, dims="x"), y=xr.DataArray(ys, dims="y"), method="nearest").values


def rema():
    with rasterio.open(ROOT / "data" / "REMA_ase_32m.tif") as r:
        w = from_bounds(X0, Y0, X1, Y1, r.transform)
        z = r.read(1, window=w).astype("float64")
        tr = r.window_transform(w)
        nod = r.nodata
    z[(z == nod) | ~np.isfinite(z)] = np.nan
    f = 4
    ny, nx = (z.shape[0] // f) * f, (z.shape[1] // f) * f
    zb = np.nanmean(z[:ny, :nx].reshape(ny // f, f, nx // f, f), axis=(1, 3))
    xs = tr.c + (np.arange(nx // f) + 0.5) * 32.0 * f
    ys = tr.f - (np.arange(ny // f) + 0.5) * 32.0 * f
    # convolução normalizada só sobre o domínio hidrostático (sem zona de flexão): o degrau na linha de
    # aterramento, a depressão da flexão e a frente não entram no passa-alta
    ok = np.isfinite(zb) & (hidrostatico(xs, ys) == 1)
    sig = 2000.0 / (32.0 * f)
    num = gaussian_filter(np.where(ok, zb, 0.0), sig)
    den = gaussian_filter(ok.astype(float), sig)
    hp = np.where(ok, zb - num / np.maximum(den, 1e-6), np.nan)
    X, Y = np.meshgrid(xs, ys)
    d = pd.DataFrame(dict(x_km=X.ravel() / 1e3, y_km=Y.ravel() / 1e3, hp=hp.ravel()))
    return d[np.isfinite(d.hp)].round(3)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rema().to_csv(OUT / "pig_rema_hp.csv", index=False)

    m = xr.open_dataset(ROOT / "data" / "interim" / "hydrostatic_mask_500m.nc")[["classe", "plataforma"]]
    m = m.sel(x=slice(X0, X1), y=slice(Y1, Y0)).to_dataframe().reset_index()
    m = m.assign(x_km=m.x / 1e3, y_km=m.y / 1e3)[["x_km", "y_km", "classe", "plataforma"]]
    bm = xr.open_dataset(ROOT / "data" / "NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc")["mask"]
    bm = bm.sel(x=slice(X0, X1), y=slice(Y1, Y0)).to_dataframe().reset_index()
    bm = bm.assign(x_km=bm.x / 1e3, y_km=bm.y / 1e3)[["x_km", "y_km", "mask"]].rename(columns={"mask": "bedmachine"})
    m = m.merge(bm, on=["x_km", "y_km"], how="outer")
    m.to_csv(OUT / "pig_mascara.csv", index=False)
    print("classes na janela:", m.classe.value_counts().to_dict(), m.bedmachine.value_counts().to_dict())

    v = xr.open_dataset(ROOT / "data" / "velocity_itslive_annual.nc")[["vx", "vy"]]
    v = v.sel(x=slice(X0, X1), y=slice(Y0, Y1)).mean("time")
    v = v.coarsen(x=8, y=8, boundary="trim").mean().to_dataframe().reset_index()
    v = v.assign(x_km=v.x / 1e3, y_km=v.y / 1e3, vel=np.hypot(v.vx, v.vy))
    bmv = xr.open_dataset(ROOT / "data" / "NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc")["mask"]
    v["bm"] = bmv.sel(x=xr.DataArray(v.x.to_numpy(), dims="p"), y=xr.DataArray(v.y.to_numpy(), dims="p"),
                      method="nearest").values
    # isolinhas sobre o oceano são cobertas no R por uma camada de oceano desenhada por cima
    v[["x_km", "y_km", "vx", "vy", "vel"]].dropna().round(2).to_csv(OUT / "pig_vel.csv", index=False)

    t = pd.read_csv(CAN / "tendencias_secoes.csv")[["chave", "dDdt", "dDdt_lo", "dDdt_hi", "D_med"]]
    linhas = []
    for crit, arq in (("estrito", "canais_validados.parquet"), ("relaxado", "canais_validados_relax.parquet")):
        e = pd.read_parquet(ROOT / "data" / "interim" / "canais" / arq)
        e = e[(e.plataforma == "Pine_Island") & e.canal_basal & np.isfinite(e.x)].copy()
        e["chave"] = chave(e)
        g = e.groupby("chave").agg(x=("x", "mean"), y=("y", "mean"), vel=("vel", "mean"), prom=("prom", "median"),
                                   larg=("largura_m", "median"), theta=("theta", "mean"),
                                   n_ciclos=("n_ciclos", "max")).reset_index()
        g["criterio"] = crit
        linhas.append(g)
    s = pd.concat(linhas)
    s = s.sort_values("criterio").drop_duplicates("chave", keep="first")     # estrito vence (vem antes)
    s = s.merge(t, on="chave", how="left")
    s = s.assign(x_km=s.x / 1e3, y_km=s.y / 1e3, classe=np.where(s.vel < 1000, "lento", "rápido"))
    s.drop(columns=["x", "y"]).round(4).to_csv(OUT / "pig_secoes.csv", index=False)
    print("seções PIG:", s.groupby(["criterio", "classe"]).size().to_dict())

    var = {"": "R400 (central)", "_r200": "R200", "_r800": "R800", "_relax": "critério relaxado"}
    tx = []
    for suf, rot in var.items():
        d = pd.read_csv(CAN / f"lagrangiano_efeito_secao{suf}.csv")
        tx.append(d.assign(variante=rot))
    tx = pd.concat(tx)
    base = tx[tx.variante == "R400 (central)"].set_index(["plataforma", "classe"]).taxa_fe
    si = pd.read_csv(CAN / "lagrangiano_efeito_secao_sint05.csv").set_index(["plataforma", "classe"]).taxa_fe
    rec = ((si - base) / 0.5).rename("recuperacao_sint").reset_index()
    tx = tx.merge(rec, on=["plataforma", "classe"], how="left")
    tx["classe"] = tx.classe.str.split(" ").str[0]
    tx.round(4).to_csv(OUT / "taxas_plataformas.csv", index=False)

    d = pd.read_csv(CAN / "comparacao_lowery.csv")
    d["bloco"] = (d.x // 10000).astype(int).astype(str) + "_" + (d.y // 10000).astype(int).astype(str)
    res = []
    for cl, g in d.groupby("classe"):
        g = g[np.isfinite(g.dm_lowery)]
        bl = g.bloco.unique()
        reps = [pd.concat([g[g.bloco == b] for b in RNG.choice(bl, len(bl))]).dm_lowery.median() for _ in range(2000)]
        res.append(dict(classe=cl, fonte="Lowery et al. (2025), CryoSat-2", n=len(g), n_blocos=len(bl),
                        dm=g.dm_lowery.median(), lo=np.percentile(reps, 2.5), hi=np.percentile(reps, 97.5),
                        melt_centro=g.m_c.median(), melt_quilhas=np.nanmedian(0.5 * (g.m_q1 + g.m_q2))))
    fe = tx[(tx.variante == "R400 (central)") & (tx.plataforma == "Pine_Island")]
    for _, r in fe.iterrows():
        res.append(dict(classe=r.classe, fonte="ICESat-2 (este trabalho)", n=r.n_secoes, n_blocos=r.n_blocos,
                        dm=r.taxa_fe_H, lo=r.lo_H, hi=r.hi_H))
    pd.DataFrame(res).round(3).to_csv(OUT / "lowery_resumo.csv", index=False)
    print(pd.DataFrame(res).round(2).to_string())
    d.drop(columns=["bloco"]).round(4).to_csv(OUT / "lowery_secoes.csv", index=False)


if __name__ == "__main__":
    main()
