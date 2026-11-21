"""Termo de espalhamento H·∇·u por célula e por plataforma (fim do passo 6 do plano basal).

H: mediana de H por pixel de 500 m da máscara, 2019–2025, depois do filtro de presença de gelo
(o mesmo de ``run_thickness_ase.ice_presence_filter``), com o firn GEMB (principal) e o IMAU
(alternativa). Pixel sem dado próprio: média gaussiana normalizada (σ = 1 km) dos vizinhos,
e só se a vizinhança tiver peso suficiente. Isso não é extrapolação para fora do domínio.
∇·u: ``data/interim/divergence/div_<σ>km.nc`` (``run_divergence_ase.py``).

Escala σ por plataforma: o mínimo da validação cruzada (``outputs/divergence_ase/escala_otima.csv``:
1 km para Crosson, Dotson e Cosgrove; 2 km para a Thwaites; 3 km para a PIG). As outras escalas
(1, 2, 3, 5 km) saem como sensibilidade, porque o mínimo de CV é raso.

Unidades: Thwaites dividida em TEIS (≤ 1.000 m/a) e TWGT (> 1.000 m/a), como na série de espessura.

Saída: ``data/interim/flux_term/H_div_500m.nc`` (mapas) e
``outputs/flux_term_ase/H_div_por_unidade.csv`` (médias por área, média 2019–2025 e por ano).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipelines" / "derretimento_basal"))
from run_thickness_ase import ice_presence_filter, speed_split  # noqa: E402

MASK = ROOT / "data" / "interim" / "hydrostatic_mask_500m.nc"
THK = ROOT / "data" / "interim" / "thickness"
DIV = ROOT / "data" / "interim" / "divergence"
OUT_NC = ROOT / "data" / "interim" / "flux_term" / "H_div_500m.nc"
OUT = ROOT / "outputs" / "flux_term_ase"
NAMES = {1: "Pine_Island", 2: "Thwaites", 3: "Crosson", 4: "Dotson", 5: "Cosgrove"}
SCALES = (1, 2, 3, 5)


def grid_median(x, y, v, xs, ys):
    j = np.floor((x - xs[0]) / (xs[1] - xs[0]) + 0.5).astype(np.int64)
    i = np.floor((y - ys[0]) / (ys[1] - ys[0]) + 0.5).astype(np.int64)
    ok = (i >= 0) & (i < len(ys)) & (j >= 0) & (j < len(xs)) & np.isfinite(v)
    key = i[ok] * len(xs) + j[ok]
    med = pd.Series(v[ok]).groupby(key).median()
    g = np.full(len(ys) * len(xs), np.nan)
    g[med.index.to_numpy()] = med.to_numpy()
    return g.reshape(len(ys), len(xs))


def fill_gaps(g, dom, sigma_px=2.0):
    w = np.isfinite(g).astype(float)
    num = ndimage.gaussian_filter(np.nan_to_num(g), sigma_px)
    den = ndimage.gaussian_filter(w, sigma_px)
    out = g.copy()
    hole = dom & ~np.isfinite(g) & (den > 0.2)
    out[hole] = num[hole] / den[hole]
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_NC.parent.mkdir(parents=True, exist_ok=True)
    m = xr.open_dataset(MASK)
    xs, ys = m.x.values, m.y.values
    cls, plat = m.classe.values, m.plataforma.values
    dom_all = cls == 1
    H = {"gemb": np.full(cls.shape, np.nan), "imau": np.full(cls.shape, np.nan)}
    fast = np.zeros(cls.shape, bool)
    for k, name in NAMES.items():
        d = pd.read_parquet(THK / f"espessura_{name}.parquet",
                            columns=["x", "y", "h_sea", "H_gemb", "H_imau", "classe"])
        d = d[(d["classe"] == 1) & np.isfinite(d["H_gemb"])]
        keep = ice_presence_filter(d["x"].to_numpy(), d["y"].to_numpy(), d["h_sea"].to_numpy())
        d = d[keep]
        for f in ("gemb", "imau"):
            g = grid_median(d["x"].to_numpy(), d["y"].to_numpy(), d[f"H_{f}"].to_numpy(), xs, ys)
            sel = plat == k
            H[f][sel] = g[sel]
        if name == "Thwaites":
            ii, jj = np.nonzero(dom_all & (plat == k))
            fs = speed_split(xs[jj], ys[ii])
            fast[ii[fs], jj[fs]] = True
    cov = {f: float(np.isfinite(H[f][dom_all]).mean()) for f in H}
    for f in H:
        H[f] = fill_gaps(H[f], dom_all)
    print(f"cobertura de H própria no domínio: {cov}; depois de preencher: "
          f"{ {f: round(float(np.isfinite(H[f][dom_all]).mean()), 3) for f in H} }", flush=True)

    unit = np.full(cls.shape, "", dtype=object)
    for k, name in NAMES.items():
        sel = dom_all & (plat == k)
        unit[sel] = name
    unit[dom_all & (plat == 2) & fast] = "Thwaites_TWGT"
    unit[dom_all & (plat == 2) & ~fast] = "Thwaites_TEIS"

    best = pd.read_csv(ROOT / "outputs" / "divergence_ase" / "escala_otima.csv")
    best = dict(zip(best.plataforma, best.sigma_km))
    rows, layers = [], {}
    for sc in SCALES:
        ds = xr.open_dataset(DIV / f"div_{sc}km.nc")
        dm = ds.div_media.values
        da = ds.div_anual.values
        years = ds.ano.values
        for f in H:
            layers[f"H_div_{f}_{sc}km"] = (H[f] * dm).astype(np.float32)
        for u in sorted(set(unit[dom_all])):
            sel = unit == u
            ok = sel & np.isfinite(H["gemb"]) & np.isfinite(dm)
            base = u.split("_TEIS")[0].split("_TWGT")[0]
            row = {"unidade": u, "sigma_km": sc, "otima": sc == best.get(base),
                   "area_km2": sel.sum() * 0.25, "frac_area_com_valor": ok.sum() / sel.sum(),
                   "H_medio_m": float(np.mean(H["gemb"][ok])),
                   "div_media_1pa": float(np.mean(dm[ok])),
                   "H_div_gemb_mpa": float(np.mean((H["gemb"] * dm)[ok])),
                   "H_div_imau_mpa": float(np.nanmean((H["imau"] * dm)[ok]))}
            for k, yv in enumerate(years):
                okk = ok & np.isfinite(da[k])
                row[f"H_div_{yv}"] = float(np.mean((H["gemb"] * da[k])[okk]))
            rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "H_div_por_unidade.csv", index=False)
    xr.Dataset({k: (("y", "x"), v) for k, v in layers.items()}
               | {"H_gemb": (("y", "x"), H["gemb"].astype(np.float32)),
                  "H_imau": (("y", "x"), H["imau"].astype(np.float32))},
               coords={"x": xs, "y": ys}).to_netcdf(OUT_NC)
    pd.set_option("display.width", 220)
    cols = ["unidade", "sigma_km", "otima", "area_km2", "frac_area_com_valor", "H_medio_m",
            "div_media_1pa", "H_div_gemb_mpa", "H_div_imau_mpa"]
    print(t[cols].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
