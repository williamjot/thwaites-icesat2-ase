"""Canais basais, passo 6: comparação com o derretimento basal de Lowery et al. (2025, CryoSat-2).

Produto: PIG_Basal_Melt.nc (UK PDC, doi:10.5285/d093b7b9-f590-48dd-8206-74d239cdbf30, OGL-UK-3.0):
melt lagrangiano mensal 2012-07..2021-12, 250 m, EPSG:3031 (meridiano 0, paralelo −71), cada mapa
com ~3 anos de dados. Período comum com o ICESat-2: mapas de 2019-07 a 2021-12.

Para cada seção de canal validada da PIG (passo 2b), contraste canal − quilha no melt de Lowery:
  m_c = melt no centro do canal; m_q = média do melt nos dois pontos a ±1,5·W_perp do centro,
  na direção perpendicular ao fluxo (W_perp = largura ao longo da trilha × sen θ).
Nosso equivalente: Δm_IS2 ≈ (ρw/(ρw−ρi))·dD/dt = 9,34·dD/dt (aprofundamento da depressão =
canal derretendo mais que as quilhas; despreza a diferença de H·∇·u, ~1 m/a).
Comparação por classe de velocidade; a taxa ICESat-2 por seção é a de seção fixa (passo 3), válida
só em gelo lento (< 1 km/a). Saída: outputs/canais/comparacao_lowery.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
RNG = np.random.default_rng(20261117)
K_HID = 1027.0 / (1027.0 - 917.0)


def main():
    L = xr.open_dataset(ROOT / "data" / "external_validation" / "lowery2025" / "PIG_Basal_Melt.nc").Basal_Melt
    L = L.sel(Date=slice("2019-07-01", "2021-12-01")).mean("Date")
    e = pd.read_parquet(ROOT / "data" / "interim" / "canais" / "canais_validados.parquet")
    e = e[(e.plataforma == "Pine_Island") & e.canal_basal].copy()
    t = pd.read_csv(ROOT / "outputs" / "canais" / "tendencias_secoes.csv")
    t = t[t.plataforma == "Pine_Island"]
    e["par"] = (e.beam + 1) // 2
    e["chave"] = e.plataforma + "_" + e.rgt.astype(str) + "_" + e.par.astype(str) + "_" + (e.b0 // 10).astype(str)
    e = e.groupby("chave").agg(x=("x", "mean"), y=("y", "mean"), vx=("vx", "mean"), vy=("vy", "mean"),
                               vel=("vel", "mean"), larg=("largura_m", "mean"), theta=("theta", "mean")).reset_index()
    sp = np.hypot(e.vx, e.vy)
    px, py = -e.vy / sp, e.vx / sp
    wp = e.larg * np.sin(np.radians(e.theta))

    def amostra(x, y):
        return L.interp(x=xr.DataArray(x.to_numpy(), dims="p"), y=xr.DataArray(y.to_numpy(), dims="p")).values

    e["m_c"] = amostra(e.x, e.y)
    e["m_q1"] = amostra(e.x + 1.5 * wp * px, e.y + 1.5 * wp * py)
    e["m_q2"] = amostra(e.x - 1.5 * wp * px, e.y - 1.5 * wp * py)
    e["dm_lowery"] = e.m_c - 0.5 * (e.m_q1 + e.m_q2)
    d = e.merge(t[["chave", "dDdt", "dDdt_lo", "dDdt_hi", "D_med"]], on="chave", how="left")
    d["dm_is2"] = K_HID * d.dDdt
    d["classe"] = np.where(d.vel < 1000, "lento", "rápido")
    d.to_csv(ROOT / "outputs" / "canais" / "comparacao_lowery.csv", index=False)
    d["bloco"] = (d.x // 10000).astype(int).astype(str) + "_" + (d.y // 10000).astype(int).astype(str)
    for cl, g in d.groupby("classe"):
        g = g[np.isfinite(g.dm_lowery)]
        bl = g.bloco.unique()
        reps = [pd.concat([g[g.bloco == b] for b in RNG.choice(bl, len(bl))]).dm_lowery.median() for _ in range(2000)]
        both = g[np.isfinite(g.dm_is2)]
        r = np.corrcoef(both.dm_is2, both.dm_lowery)[0, 1] if len(both) > 3 else np.nan
        print(f"{cl}: n={len(g)} blocos={len(bl)}  Δm Lowery (canal−quilha) mediana {g.dm_lowery.median():.2f} m/a "
              f"[{np.percentile(reps, 2.5):.2f}, {np.percentile(reps, 97.5):.2f}]; melt centro {g.m_c.median():.1f}; "
              f"quilhas {np.nanmedian(0.5 * (g.m_q1 + g.m_q2)):.1f}")
        print(f"   com taxa IS2: n={len(both)}  Δm IS2 mediana {both.dm_is2.median():.2f}; Lowery nas mesmas {both.dm_lowery.median():.2f}; r={r:.2f}")


if __name__ == "__main__":
    main()
