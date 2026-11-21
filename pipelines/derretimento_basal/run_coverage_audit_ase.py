"""Auditoria de cobertura ATL06 sobre as plataformas da ASE, por estação e ano.

Pergunta: em que plataformas e com que resolução temporal (trimestral, semestral, anual) há
dado suficiente para um produto de derretimento basal? Nenhum parâmetro do produto é escolhido
aqui; isto só mede.

Entrada: ``data/ase_v2/{djf,mam,jja,son}/*.parquet`` (contrato v2) e
``data/interim/ase_shelf_mask_240m.nc`` (``pipelines/build_shelf_mask_ase.py``).
Filtro: ``atl06_quality_summary == 0`` e altura finita abaixo do fill (3,4e38 não é NaN).
Ano da estação: DJF recebe o ano de janeiro (dez/2018 entra em DJF 2019).

Métricas por plataforma × estação × ano:
- ``n_seg``: segmentos válidos;
- ``n_passes``: passagens independentes ``(rgt, cycle)``; a unidade independente é a passagem,
  não o segmento (ver a nota sobre σ formal otimista);
- ``frac_cel_2km``: fração das células de 2 km da plataforma com ≥1 segmento.

Saída: ``outputs/coverage_audit_ase/cobertura.csv`` e ``tabela_frac_cel_2km.csv``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "data" / "ase_v2"
MASK = ROOT / "data" / "interim" / "ase_shelf_mask_240m.nc"
OUT = ROOT / "outputs" / "coverage_audit_ase"
COLS = ["x", "y", "h_elv", "atl06_quality_summary", "rgt", "cycle"]
CELL = 2000.0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    m = xr.open_dataset(MASK)["shelf"]
    legend = dict(p.split("=") for p in m.attrs["legenda"].split("; "))
    names = {int(k): v for k, v in legend.items() if int(k) > 0}
    xs, ys, sh = m["x"].values, m["y"].values, m.values
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]

    # células de 2 km de cada plataforma (denominador da fração de cobertura)
    XX, YY = np.meshgrid(xs, ys)
    cells_total = {}
    for k in names:
        sel = sh == k
        cells_total[k] = len(set(zip((XX[sel] // CELL).astype(int),
                                     (YY[sel] // CELL).astype(int))))

    rows, cellsets = [], {}
    for season_dir in sorted(p for p in STORE.iterdir() if p.is_dir()):
        season = season_dir.name
        for f in sorted(season_dir.glob("*.parquet")):
            month = int(f.name.split("_")[1][4:6])
            year = int(f.name.split("_")[1][:4]) + (1 if month == 12 else 0)
            d = pd.read_parquet(f, columns=COLS)
            d = d[(d["atl06_quality_summary"] == 0) & np.isfinite(d["h_elv"])
                  & (d["h_elv"] < 1e30)]
            if d.empty:
                continue
            j = np.rint((d["x"].to_numpy() - xs[0]) / dx).astype(int)
            i = np.rint((d["y"].to_numpy() - ys[0]) / dy).astype(int)
            ok = (i >= 0) & (i < sh.shape[0]) & (j >= 0) & (j < sh.shape[1])
            d, i, j = d[ok], i[ok], j[ok]
            k = sh[i, j]
            d = d.assign(shelf=k)[k > 0]
            if d.empty:
                continue
            for kk, g in d.groupby("shelf"):
                rows.append({"shelf": names[int(kk)], "season": season, "year": year,
                             "n_seg": len(g),
                             "passes": set(zip(g["rgt"], g["cycle"]))})
                key = (int(kk), season, year)
                cellsets.setdefault(key, set()).update(
                    zip((g["x"] // CELL).astype(int), (g["y"] // CELL).astype(int)))
        print(f"{season}: lido", flush=True)

    df = pd.DataFrame(rows)
    agg = (df.groupby(["shelf", "season", "year"])
             .agg(n_seg=("n_seg", "sum"),
                  n_passes=("passes", lambda s: len(set().union(*s))))
             .reset_index())
    inv = {v: k for k, v in names.items()}
    agg["frac_cel_2km"] = [
        len(cellsets[(inv[r.shelf], r.season, r.year)]) / cells_total[inv[r.shelf]]
        for r in agg.itertuples()]
    agg.to_csv(OUT / "cobertura.csv", index=False)
    tab = agg.pivot_table(index=["shelf", "season"], columns="year", values="frac_cel_2km")
    tab.round(2).to_csv(OUT / "tabela_frac_cel_2km.csv")
    pd.set_option("display.width", 200)
    print(tab.round(2))
    print(agg.pivot_table(index=["shelf", "season"], columns="year", values="n_passes"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
