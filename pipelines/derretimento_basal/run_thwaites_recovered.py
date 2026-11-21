"""Thwaites com a faixa de 1,5–4 km junto à linha recuperada por correção (em vez de excluída).

Correção medida célula a célula por ``run_gz_recovery_ase.py`` (maré × β na zona de flexão e
espessura × fator do radar MCoRDS): Δ = m_b(corrigido) − m_b(sem correção) nas mesmas células.
Δ é suavizado (kernel gaussiano de 1,5 km) e somado ao m_b contínuo do domínio união
(``componentes_uniao_500m.nc``: SMB − DH/Dt − H·∇·u na escala de 2 km). Δ = 0 além de 8 km
(onde o fator do radar já é 1).

Domínio: o da Thwaites no produto **sem** o corte de 4 km do radar (substituído pela correção);
ficam a zona de flexão (d\\* = 1,5 km), as linhas de 2019–2025 (1,5 km) e a faixa da frente.
Incerteza da correção: |Δ| inteiro por pixel (o perfil do radar vem de 13–14 voos de 2009–2018,
e a parte de 4–6 km é dominada pela variação de 0,94 → 0,98 do perfil).

Teste de aceitação: m_b a ≤ 3 km do furo (Davis et al. 2023) entre 2,0 e 5,4 m/a (± σ); e o
centro da TEIS (Wild et al. 2024) sem mudança no RMS.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
CELLS = ROOT / "outputs" / "gz_recovery" / "celulas_Thwaites.parquet"
COMP = ROOT / "data" / "interim" / "basal_melt" / "componentes_uniao_500m.nc"
OUT = ROOT / "outputs" / "gz_recovery"
RHO_I, PIX = 917.0, 0.25e6
SIG = 1500.0


def main() -> int:
    r = pd.read_parquet(CELLS)
    p = r.pivot_table(index=["x", "y"], columns="versao", values=["mb", "dist_gl_km"]).dropna()
    p.columns = [f"{a}_{b}" for a, b in p.columns]
    p = p.reset_index()
    p["delta"] = p.mb_corrigido - p.mb_sem_correcao
    c = xr.open_dataset(COMP)
    XX, YY = np.meshgrid(c.x.values, c.y.values)
    u = c.unidade.values
    mb_raw = c.SMB.values - c.DHDt.values - c.H_gemb.values * c.div_2km.values
    dist = c.dist_LA_BedMachine_km.values
    th = np.isin(u, ["Thwaites_TEIS", "Thwaites_TWGT"]) & np.isfinite(mb_raw)
    tree = cKDTree(np.c_[p.x, p.y])
    delta = np.zeros(u.shape)
    ii, jj = np.nonzero(th & (dist < 8.0))
    nb = tree.query_ball_point(np.c_[XX[ii, jj], YY[ii, jj]], 3 * SIG)
    for n, idx in enumerate(nb):
        if idx:
            idx = np.asarray(idx)
            w = np.exp(-0.5 * ((p.x.values[idx] - XX[ii[n], jj[n]]) ** 2
                               + (p.y.values[idx] - YY[ii[n], jj[n]]) ** 2) / SIG ** 2)
            delta[ii[n], jj[n]] = np.sum(w * p.delta.values[idx]) / w.sum()
    mb_cor = mb_raw + delta
    front_ok = (c.frente_dist_km.values > 1.5) & ~(c.perda_gelo_frac.values > 0.10)
    base = th & front_ok & (c.classe.values == 1) & (c.dist_LA_recente_km.values >= 1.5)
    old = base & (dist >= 4.0)                  # domínio do produto (corte do radar a 4 km)
    new = base                                  # recuperado: d* (1,5 km) em diante
    rows = []
    for un in ["Thwaites_TEIS", "Thwaites_TWGT"]:
        for lab, dom, mm in (("antes (corte 4 km, sem correção)", old, mb_raw),
                             ("recuperado (corrigido)", new, mb_cor)):
            s = dom & (u == un)
            rows.append({"unidade": un, "versao": lab, "area_km2": s.sum() * 0.25,
                         "mb_mpa": float(np.mean(mm[s])), "mb_mediana": float(np.median(mm[s])),
                         "sigma_correcao_mpa": float(np.mean(np.abs(delta[s]))),
                         "Gt": float(np.mean(mm[s]) * s.sum() * PIX * RHO_I / 1e12)})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "thwaites_recuperada.csv", index=False)
    tf = Transformer.from_crs(4326, 3031, always_xy=True)
    bx, by = tf.transform(-104.825, -75.207)
    dd = np.hypot(XX - bx, YY - by) / 1e3
    for R in (2, 3):
        s = new & (dd <= R)
        print(f"furo ≤ {R} km: {s.sum()} pixels no domínio recuperado; m_b mediano "
              f"{np.median(mb_cor[s]) if s.any() else np.nan:.2f} m/a "
              f"(sem correção {np.median(mb_raw[s]) if s.any() else np.nan:.2f})")
    xr.Dataset({"mb_corrigido": (("y", "x"), np.where(new, mb_cor, np.nan).astype(np.float32)),
                "delta_correcao": (("y", "x"), np.where(th, delta, np.nan).astype(np.float32))},
               coords={"x": c.x.values, "y": c.y.values}).to_netcdf(
        ROOT / "data" / "interim" / "basal_melt" / "thwaites_recuperada_500m.nc")
    pd.set_option("display.width", 200)
    print(t.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
