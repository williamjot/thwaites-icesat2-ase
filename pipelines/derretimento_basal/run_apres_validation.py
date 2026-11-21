"""Validação do derretimento basal com ApRES (ITGC MELT, TEIS G1–G4, 2020).

Para cada ponto, com os arquivos baixados a cada ~5 dias (``fetch_validation_radar.py``):
- perfil complexo médio por arquivo (``thwaites.validation.apres``);
- fundo: eco mais forte entre 300 e 800 m no primeiro arquivo;
- por par consecutivo: derretimento = deslocamento do fundo além do previsto pela deformação
  vertical (reta robusta das camadas internas, fase desenrolada). Pares com < 20 camadas
  coerentes saem (um par com 20 camadas deu 24 m/a contra ~1,5–2 m/a dos vizinhos);
- taxa anual: Σ(derretimento dos pares válidos) / Σ(Δt dos pares válidos); IC 95% por bootstrap
  dos pares.

Comparação com o produto: m_b (mapa contínuo) nos pixels hidrostáticos dentro da caixa dos
pontos e num raio de 3 km. A posição de cada ponto não está nos metadados (GPS desligado); a
espessura medida pelo ApRES ordena os pontos ao longo do fluxo.

Saída: ``outputs/validation/apres_*.csv``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from thwaites.validation.apres import Config, file_profile, find_bed, melt_between  # noqa: E402

SRC = ROOT / "data" / "validation" / "apres_teis"
MB = ROOT / "data" / "interim" / "basal_melt" / "mb_continuo_500m.nc"
MASK = ROOT / "data" / "interim" / "hydrostatic_mask_500m.nc"
OUT = ROOT / "outputs" / "validation"
BBOX = (-104.8247, -75.21998, -104.7924, -75.19011)
MIN_LAYERS = 20
RNG = np.random.default_rng(1)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    pairs, sites = [], []
    for site in sorted(p.name for p in SRC.iterdir() if p.is_dir()):
        files = sorted((SRC / site).glob("*.nc"))
        if len(files) < 3:
            continue
        prof = [file_profile(f, cfg) for f in files]
        R = prof[0][0]
        zb = find_bed(R, prof[0][1], 300.0, 800.0)
        for (R1, s1, t1), (R2, s2, t2) in zip(prof[:-1], prof[1:]):
            dt = (t2 - t1) / np.timedelta64(1, "D") / 365.25
            m, b, n = melt_between(R, s1, s2, cfg, zb)
            pairs.append({"ponto": site, "t0": str(t1), "t1": str(t2), "dt_a": dt,
                          "melt_m": m, "deform_por_m": b, "n_camadas": n,
                          "melt_mpa": m / dt if np.isfinite(m) else np.nan,
                          "deform_1pa": b / dt if np.isfinite(b) else np.nan})
        d = pd.DataFrame([p for p in pairs if p["ponto"] == site])
        ok = d[(d.n_camadas >= MIN_LAYERS) & np.isfinite(d.melt_m)]
        rate = ok.melt_m.sum() / ok.dt_a.sum()
        boot = []
        for _ in range(2000):
            k = RNG.integers(0, len(ok), len(ok))
            boot.append(ok.melt_m.to_numpy()[k].sum() / ok.dt_a.to_numpy()[k].sum())
        sites.append({"ponto": site, "espessura_m": zb, "n_pares": len(d), "n_validos": len(ok),
                      "periodo_dias": ok.dt_a.sum() * 365.25, "melt_mpa": rate,
                      "ic95_lo": np.percentile(boot, 2.5), "ic95_hi": np.percentile(boot, 97.5),
                      "melt_mediana_pares_mpa": ok.melt_mpa.median(),
                      "deform_mediana_1pa": ok.deform_1pa.median()})
        print(sites[-1], flush=True)
    pd.DataFrame(pairs).to_csv(OUT / "apres_pares.csv", index=False)
    s = pd.DataFrame(sites)
    s.to_csv(OUT / "apres_pontos.csv", index=False)

    # produto perto dos pontos
    tf = Transformer.from_crs(4326, 3031, always_xy=True)
    mb = xr.open_dataset(MB)
    mk = xr.open_dataset(MASK)
    lon = np.linspace(BBOX[0], BBOX[2], 9)
    lat = np.linspace(BBOX[1], BBOX[3], 9)
    LO, LA = np.meshgrid(lon, lat)
    X, Y = tf.transform(LO.ravel(), LA.ravel())
    cx, cy = np.mean(X), np.mean(Y)
    XX, YY = np.meshgrid(mb.x.values, mb.y.values)
    rr = np.hypot(XX - cx, YY - cy)
    rows = []
    for r_km in (1.5, 3.0, 5.0):
        sel = (rr <= r_km * 1e3) & np.isfinite(mb.mb.values)
        if sel.any():
            rows.append({"raio_km": r_km, "n_pixels": int(sel.sum()),
                         "mb_mediano_mpa": float(np.median(mb.mb.values[sel])),
                         "mb_p25": float(np.percentile(mb.mb.values[sel], 25)),
                         "mb_p75": float(np.percentile(mb.mb.values[sel], 75)),
                         "se_mediano_mpa": float(np.nanmedian(mb.se_mb.values[sel])),
                         "dist_gl_mediana_km": float(np.nanmedian(mk.dist_gl_km.values[sel]))})
    p = pd.DataFrame(rows)
    p.to_csv(OUT / "produto_perto_apres.csv", index=False)
    pd.set_option("display.width", 220)
    print(s.round(3).to_string(index=False))
    print(p.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
