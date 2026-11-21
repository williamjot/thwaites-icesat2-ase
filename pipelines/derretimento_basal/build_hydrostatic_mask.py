"""Máscara hidrostática das plataformas da ASE, construída a partir da resposta medida à maré.

Substitui o buffer fixo de 2 km de Meng et al. 2025 por regras tiradas dos dados
(``outputs/tide_response_ase/beta_lagr_todas.parquet``, de ``run_tide_response_lagrangian.py``).

1. **Patamar de flutuação P** por plataforma: mediana de β nas células entre 5 e 15 km a jusante
   da linha de aterramento do BedMachine. O patamar não é 1: na Dotson e na PIG ele fica em
   1,05–1,09, provavelmente porque o CATS subestima a maré local. Testar contra 1 exato
   reprovaria gelo livre.
2. **Distância de corte d\*** por plataforma: início da primeira faixa de 250 m em que a mediana
   de β alcança f·P e se mantém no km seguinte (faixas com <10 células são neutras). Variantes
   f = 0,8 / 0,9 / 0,95 e uma por sobreposição de IC (bootstrap por blocos espaciais de 5 km).
   **Padrão: f = 0,9, escolha declarada e não ótimo demonstrado** — o passo do derretimento basal
   deve medir a sensibilidade às variantes. A variante por IC não é o padrão porque é permissiva
   onde a incerteza é grande (falta de evidência de flexão viraria "hidrostático").
3. **Exclusão célula a célula além de d\***: célula com σβ < 0,25 e β + 2σβ < P_inf (limite
   inferior do IC de P), mantida só se tiver ao menos uma vizinha de 500 m (8-conexa) também
   reprovada. Uma reprovação isolada é tratada como falso positivo do teste a 2σ.

Domínio: gelo flutuante do BedMachine v4.1 ∩ máscara das plataformas (`ase_shelf_mask_240m.nc`,
NSIDC-0794 em mar/2021). Conservador por construção: gelo que o BedMachine dá como apoiado fica
fora mesmo onde a linha pode ter recuado, e células sem β medido além de d\* entram (β é
medido só até 15 km da linha; mais longe, a flexão por maré não é esperada).

Classes: 0 fora do domínio; 1 hidrostático; 2 zona de flexão (dist < d\*); 3 excluído por β
(além de d\*).

Saída: ``data/interim/hydrostatic_mask_500m.nc`` e ``outputs/hydrostatic_mask/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
BETA = ROOT / "outputs" / "tide_response_ase" / "beta_lagr_todas.parquet"
SHELF_MASK = ROOT / "data" / "interim" / "ase_shelf_mask_240m.nc"
BEDMACHINE = ROOT / "data" / "NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc"
OUT_NC = ROOT / "data" / "interim" / "hydrostatic_mask_500m.nc"
OUT = ROOT / "outputs" / "hydrostatic_mask"
BLOCK = 5000.0
NBOOT = 1000
BIN = 0.25
RNG = np.random.default_rng(20261115)
FRACS = (0.8, 0.9, 0.95)
VARIANTS = ("f080", "f090", "f095", "ic")
DEFAULT = "f090"


def block_boot_median(g, nboot=NBOOT):
    """IC 95% da mediana de β por bootstrap de blocos espaciais de 5 km."""
    blk = (np.floor(g["x"] / BLOCK).astype(np.int64) * 100000
           + np.floor(g["y"] / BLOCK).astype(np.int64)).to_numpy()
    ub, inv = np.unique(blk, return_inverse=True)
    vals = g["beta"].to_numpy()
    groups = [vals[inv == k] for k in range(len(ub))]
    if len(groups) < 3:
        return np.nan, np.nan
    meds = np.empty(nboot)
    for b in range(nboot):
        pick = RNG.integers(0, len(groups), len(groups))
        meds[b] = np.median(np.concatenate([groups[k] for k in pick]))
    return np.percentile(meds, [2.5, 97.5])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(BETA)
    d = d[np.isfinite(d["beta"]) & np.isfinite(d["dist_gl_km"])]

    # 1–2: patamar e distância de corte
    params = {}
    for s, g in d.groupby("shelf"):
        plat = g[(g.dist_gl_km >= 5) & (g.dist_gl_km <= 15)]
        P = float(plat["beta"].median())
        P_lo, P_hi = block_boot_median(plat)
        edges = np.arange(0, 6.0 + 1e-9, BIN)[:-1]
        med, ok_ic = [], []
        for a in edges:
            gb = g[(g.dist_gl_km >= a) & (g.dist_gl_km < a + BIN)]
            if len(gb) < 10:          # faixa sem dado suficiente: neutra
                med.append(np.nan)
                ok_ic.append(True)
                continue
            med.append(float(gb["beta"].median()))
            lo, hi = block_boot_median(gb, nboot=300)
            ok_ic.append(bool(np.isfinite(hi) and hi >= P_lo))
        med = np.array(med)

        def first_stable(passes, span=4):
            """início da 1ª faixa que passa e cujas `span` faixas seguintes (1 km) também
            passam; faixas neutras (NaN) não reprovam."""
            for i in range(len(passes)):
                if passes[i] and all(passes[i:i + span]):
                    return float(edges[i])
            return np.nan

        dstar = {}
        for f in FRACS:
            dstar[f"f{int(f * 100):03d}"] = first_stable(
                [bool(np.isnan(m) or m >= f * P) for m in med])
        # variante por sobreposição de IC: guardada como sensibilidade, NÃO padrão. Medido em
        # 2026-11-15: é permissiva onde a incerteza é grande (Thwaites 0,75 km, embora a mediana
        # só chegue a 0,9 em 1,4 km); ausência de evidência de flexão vira "hidrostático".
        dstar["ic"] = first_stable(ok_ic, span=len(ok_ic))
        for k, v in dstar.items():
            if not np.isfinite(v):
                # sem isto, dist < NaN é sempre falso e a plataforma inteira vira "hidrostática"
                raise RuntimeError(f"{s}: d* ({k}) indeterminado")
        params[s] = {"P": round(P, 3), "P_ic95": [round(P_lo, 3), round(P_hi, 3)],
                     "d_star_km": dstar, "n_celulas_patamar": int(len(plat))}
        print(f"{s:12s} patamar {P:.3f} [{P_lo:.3f}, {P_hi:.3f}]  d* = "
              + ", ".join(f"{k} {v:.2f}" for k, v in dstar.items()) + " km", flush=True)

    # grade de 500 m do BedMachine sobre a área das plataformas
    sm = xr.open_dataset(SHELF_MASK)["shelf"]
    legend = {v: int(k) for k, v in (p.split("=") for p in sm.attrs["legenda"].split("; "))}
    x0, x1, y0, y1 = float(sm.x.min()), float(sm.x.max()), float(sm.y.min()), float(sm.y.max())
    bm = xr.open_dataset(BEDMACHINE)["mask"].sel(x=slice(x0, x1), y=slice(y1, y0))
    xs, ys, mk = bm.x.values, bm.y.values, bm.values
    flo = mk == 3
    grd = np.isin(mk, [1, 2, 4])
    dist_km = ndimage.distance_transform_edt(flo) * 0.5          # distância ao não-flutuante
    dist_km[~flo] = np.nan
    shelf_lab = sm.interp(x=xr.DataArray(xs, dims="x"), y=xr.DataArray(ys, dims="y"),
                          method="nearest").values.astype(np.int16)

    # β por célula de 500 m na grade (centros das células de β caem dentro de pixels BedMachine)
    # As células de β (centro em (k+0,5)·500 m) estão defasadas meio pixel dos centros do
    # BedMachine (múltiplos de 500 m): a razão cai exatamente em ,5. np.rint arredonda ,5 para o
    # par e fundia colunas vizinhas (medido: nenhuma célula reprovada tinha vizinha). floor
    # desloca todas igualmente.
    jj = np.floor((d["x"].to_numpy() - xs[0]) / (xs[1] - xs[0])).astype(int)
    ii = np.floor((d["y"].to_numpy() - ys[0]) / (ys[1] - ys[0])).astype(int)
    inside = (ii >= 0) & (ii < len(ys)) & (jj >= 0) & (jj < len(xs))
    beta = np.full(mk.shape, np.nan, np.float32)
    se = np.full(mk.shape, np.nan, np.float32)
    beta[ii[inside], jj[inside]] = d["beta"].to_numpy()[inside]
    se[ii[inside], jj[inside]] = d["se_beta"].to_numpy()[inside]

    layers, rows = {}, []
    for var in VARIANTS:
        cls = np.zeros(mk.shape, np.int8)
        for s, prm in params.items():
            k = legend[s]
            dom = flo & (shelf_lab == k)
            dstar = prm["d_star_km"][var]
            P_lo = prm["P_ic95"][0]
            flex = dom & (dist_km < dstar)
            cand = dom & ~flex & (se < 0.25) & (beta + 2 * se < P_lo)
            nb = ndimage.convolve(cand.astype(np.int8), np.ones((3, 3), np.int8), mode="constant")
            excl = cand & (nb >= 2)                              # ela + ≥1 vizinha
            hyd = dom & ~flex & ~excl
            cls[hyd], cls[flex], cls[excl] = 1, 2, 3
            a = 0.25
            meng = dom & (dist_km >= 2.0)
            rows.append({"variante": var, "plataforma": s, "d_star_km": dstar,
                         "patamar": prm["P"], "area_flutuante_km2": dom.sum() * a,
                         "hidrostatica_km2": hyd.sum() * a, "zona_flexao_km2": flex.sum() * a,
                         "excluida_beta_km2": excl.sum() * a,
                         "reprovadas_isoladas": int((cand & ~excl).sum()),
                         "buffer2km_Meng_km2": meng.sum() * a,
                         "diferenca_vs_Meng_km2": (hyd.sum() - meng.sum()) * a})
        layers[var] = cls

    tab = pd.DataFrame(rows).round(2)
    tab.to_csv(OUT / "areas_por_plataforma.csv", index=False)
    (OUT / "parametros.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(tab.to_string(index=False))

    data = {"classe": (("y", "x"), layers[DEFAULT])}
    data.update({f"classe_{v}": (("y", "x"), layers[v]) for v in VARIANTS if v != DEFAULT})
    data.update({"dist_gl_km": (("y", "x"), dist_km.astype(np.float32)),
                 "beta": (("y", "x"), beta), "se_beta": (("y", "x"), se),
                 "plataforma": (("y", "x"), shelf_lab)})
    ds = xr.Dataset(data, coords={"x": xs, "y": ys})
    leg = ("0=fora; 1=hidrostatico; 2=zona de flexao (dist<d*); 3=excluido por beta alem de d*")
    for v in [k for k in ds.data_vars if k.startswith("classe")]:
        ds[v].attrs = {"legenda": leg, "crs": "EPSG:3031",
                       "variante": DEFAULT if v == "classe" else v[7:]}
    ds["classe"].attrs["parametros"] = json.dumps(params)
    ds.attrs["nota"] = (f"classe = variante padrão {DEFAULT} (d* onde a mediana de β alcança "
                        "0,9·patamar). Padrão declarado, não ótimo demonstrado: a escolha deve "
                        "ser testada pela sensibilidade do derretimento basal às variantes.")
    ds.to_netcdf(OUT_NC)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
