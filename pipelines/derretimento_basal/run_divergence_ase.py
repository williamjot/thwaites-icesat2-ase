"""Divergência do fluxo ∇·u das plataformas da ASE (passo 6 do plano basal) e escolha da escala.

No referencial Lagrangiano, o balanço de massa é DH/Dt = SMB − m_b − H ∇·u. O termo de
transporte u·∇H já está dentro de DH/Dt (série Lagrangiana do passo 5), então aqui entra
**H ∇·u**, não ∇·(H u). Usar ∇·(H u) junto com um DH/Dt Lagrangiano somaria u·∇H duas vezes.

Velocidade: ITS_LIVE anual 2019–2025 (``data/velocity_itslive_annual.nc``), subamostrada a 240 m.
Derivada: filtro gaussiano de ordem 1 com convolução normalizada (lacunas não viram zero).

Escolha da escala σ (não fixada por convenção):
- **ruído(σ)**: divide os anos em pares (2020, 2022, 2024) e ímpares (2019, 2021, 2023, 2025);
  a divergência média de cada metade é calculada, e o ruído de uma média de 7 anos é estimado
  por DP(par − ímpar)/2 (os erros dos mosaicos anuais são em boa parte independentes entre anos;
  a mudança real da divergência entre as metades entra no estimador, que é, portanto, um teto);
- **validação cruzada (critério de escolha)**: CV(σ) = RMS(div_σ[pares] − div_σ0[ímpares]),
  média com o sentido inverso, σ0 = 0,5 km. Como o ruído dos ímpares é independente dos pares,
  CV² = (erro de div_σ contra o campo real a 0,5 km)² + constante; o mínimo em σ é a escala
  ótima no sentido do erro quadrático. A mudança real entre as duas metades (aceleração) entra
  como termo quase constante em σ.
- Uma primeira versão estimava o viés por RMS(div_σ − div_σ/2) e foi descartada: nas escalas
  pequenas essa diferença é quase só ruído, e o "erro total" caía monotonicamente até 8 km.

Saída: ``outputs/divergence_ase/escala_*.csv`` e, para as escalas finais,
``data/interim/divergence/div_<σ>km.nc`` (campos anuais e média, na grade de 500 m da máscara).
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
VEL = ROOT / "data" / "velocity_itslive_annual.nc"
MASK = ROOT / "data" / "interim" / "hydrostatic_mask_500m.nc"
OUT = ROOT / "outputs" / "divergence_ase"
CACHE = ROOT / "data" / "interim" / "divergence"
SCALES_KM = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0)
STRIDE = 2
NAMES = {1: "Pine_Island", 2: "Thwaites", 3: "Crosson", 4: "Dotson", 5: "Cosgrove"}


def load_vel():
    m = xr.open_dataset(MASK)
    sel = m.classe.values > 0
    ii, jj = np.nonzero(sel)
    pad = 30e3
    x0, x1 = m.x.values[jj].min() - pad, m.x.values[jj].max() + pad
    y0, y1 = m.y.values[ii].min() - pad, m.y.values[ii].max() + pad
    v = xr.open_dataset(VEL)
    ys = v.y.values
    ysl = slice(y0, y1) if ys[0] < ys[-1] else slice(y1, y0)
    s = v.sel(x=slice(x0, x1), y=ysl).isel(x=slice(None, None, STRIDE), y=slice(None, None, STRIDE))
    return (s.x.values, s.y.values, s.time.dt.year.values,
            s.vx.values.astype(np.float32), s.vy.values.astype(np.float32))


def deriv(a, sigma_px, axis):
    """Derivada gaussiana de ordem 1 com convolução normalizada (NaN = lacuna)."""
    w = np.isfinite(a).astype(np.float32)
    f = np.where(w > 0, a, 0.0).astype(np.float32)
    order = [0, 0]
    order[axis] = 1
    num_d = ndimage.gaussian_filter(f, sigma_px, order=order, mode="nearest")
    num = ndimage.gaussian_filter(f, sigma_px, mode="nearest")
    den = ndimage.gaussian_filter(w, sigma_px, mode="nearest")
    den_d = ndimage.gaussian_filter(w, sigma_px, order=order, mode="nearest")
    # derivada de num/den pela regra do quociente
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (num_d * den - num * den_d) / den ** 2
    out[den < 0.5] = np.nan              # menos da metade do peso com dado: sem estimativa
    return out


def divergence(vx, vy, xs, ys, sigma_m):
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    sp = sigma_m / abs(dx)
    return deriv(vx, sp, axis=1) / dx + deriv(vy, sp, axis=0) / dy     # 1/ano


def to_mask_grid(field, xs, ys, mx, my):
    da = xr.DataArray(field, coords={"y": ys, "x": xs}, dims=("y", "x"))
    return da.interp(x=xr.DataArray(mx, dims="x"), y=xr.DataArray(my, dims="y")).values


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--save", nargs="*", type=float, default=[],
                    help="escalas (km) para gravar os campos")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    xs, ys, years, vx, vy = load_vel()
    m = xr.open_dataset(MASK)
    mx, my = m.x.values, m.y.values
    cls, plat = m.classe.values, m.plataforma.values
    even = np.isin(years, [2020, 2022, 2024])
    rows = []
    scales = sorted(set(SCALES_KM) | set(args.save))
    s0 = scales[0]
    ref_even = ref_odd = None
    for skm in scales:
        divs = np.stack([divergence(vx[k], vy[k], xs, ys, skm * 1e3) for k in range(len(years))])
        mean = np.nanmean(divs, 0)
        g_mean = to_mask_grid(mean, xs, ys, mx, my)
        g_even = to_mask_grid(np.nanmean(divs[even], 0), xs, ys, mx, my)
        g_odd = to_mask_grid(np.nanmean(divs[~even], 0), xs, ys, mx, my)
        if skm == s0:
            ref_even, ref_odd = g_even, g_odd
        for k, name in NAMES.items():
            dom = ((cls == 1) & (plat == k) & np.isfinite(g_even) & np.isfinite(g_odd)
                   & np.isfinite(ref_even) & np.isfinite(ref_odd))
            cv = np.sqrt(0.5 * (np.mean((g_even[dom] - ref_odd[dom]) ** 2)
                                + np.mean((g_odd[dom] - ref_even[dom]) ** 2)))
            rows.append({"plataforma": name, "sigma_km": skm,
                         "sinal_rms_1pa": float(np.sqrt(np.mean(g_mean[dom] ** 2))),
                         "ruido_media7a_1pa": float(np.std(g_even[dom] - g_odd[dom]) / 2),
                         "cv_1pa": float(cv),
                         "media_dominio_1pa": float(np.mean(g_mean[dom]))})
        if skm in args.save:
            ds = xr.Dataset({"div_media": (("y", "x"), g_mean.astype(np.float32)),
                             "div_anual": (("ano", "y", "x"), np.stack(
                                 [to_mask_grid(divs[k], xs, ys, mx, my)
                                  for k in range(len(years))]).astype(np.float32))},
                            coords={"x": mx, "y": my, "ano": years})
            ds.attrs = {"sigma_km": skm, "unidade": "1/ano", "fonte": "ITS_LIVE anual 2019-2025"}
            ds.to_netcdf(CACHE / f"div_{skm:g}km.nc")
        print(f"σ = {skm:g} km pronto", flush=True)
        del divs
        gc.collect()
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "escala_diagnostico.csv", index=False)
    best = t.loc[t.groupby("plataforma")["cv_1pa"].idxmin(), ["plataforma", "sigma_km"]]
    best.to_csv(OUT / "escala_otima.csv", index=False)
    pd.set_option("display.width", 200)
    print(t.round(5).to_string(index=False))
    print("escala ótima por CV:", dict(zip(best.plataforma, best.sigma_km)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
