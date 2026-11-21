"""FASE 2H - Cobertura das referências verticais locais sobre a ROI do ASE.

Não calcula freeboard. Apenas confirma, para o geoide EIGEN-6C4 (distribuído
dentro do BedMachine v4.1) e para o DTU15MDT local: cobertura, resolução, CRS,
datum declarado e ausências — inclusive separadamente sobre gelo flutuante, que
é onde a referência de nível do mar precisa existir.

Saída:
    data/processed/vertical_reference_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from thwaites import load_config  # noqa: E402
from thwaites.legacy_roi import (  # noqa: E402
    BM_FLOATING_ICE,
    BM_GROUNDED_ICE,
    BM_OCEAN,
    BedMachineSampler,
    ROIBox,
)

BEDMACHINE = "NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc"
DTU15MDT = Path("vertical_reference") / "dtu15mdt_ase.nc"


def _audit_geoid(roi: ROIBox, data_dir: Path) -> dict:
    sampler = BedMachineSampler(data_dir / BEDMACHINE, roi,
                                variables=("mask", "geoid"), pad_m=0.0)
    geoid = sampler.fields["geoid"]
    mask = sampler.fields["mask"]
    finite = np.isfinite(geoid)
    out = {
        "produto": "EIGEN-6C4 (campo `geoid` do BedMachine Antarctica v4.1)",
        "arquivo": BEDMACHINE,
        "datum": "altura do geoide EIGEN-6C4 acima do elipsoide WGS84, em metros",
        "crs": "EPSG:3031",
        "resolucao_m": abs(float(sampler.dx)),
        "janela_roi_celulas": [int(v) for v in geoid.shape],
        "fracao_finita": float(finite.mean()),
        "min_m": float(np.nanmin(geoid)), "max_m": float(np.nanmax(geoid)),
        "fracao_finita_gelo_flutuante": float(
            finite[mask == BM_FLOATING_ICE].mean()),
        "fracao_finita_gelo_aterrado": float(
            finite[mask == BM_GROUNDED_ICE].mean()),
        "fracao_finita_oceano": float(finite[mask == BM_OCEAN].mean()),
        "celulas_por_classe": {
            "oceano": int((mask == BM_OCEAN).sum()),
            "gelo_aterrado": int((mask == BM_GROUNDED_ICE).sum()),
            "gelo_flutuante": int((mask == BM_FLOATING_ICE).sum()),
        },
    }
    return out


def _audit_mdt(roi: ROIBox, data_dir: Path) -> dict:
    import xarray as xr

    path = data_dir / DTU15MDT
    with xr.open_dataset(path) as ds:
        mdt = ds["mdt"].values
        lon = ds["lon"].values
        lat = ds["lat"].values
        attrs = {k: str(v) for k, v in ds.attrs.items()}
    lon2, lat2 = np.meshgrid(lon, lat)
    in_roi = ((lon2 >= roi.lon_min) & (lon2 <= roi.lon_max)
              & (lat2 >= roi.lat_min) & (lat2 <= roi.lat_max))
    contem_roi = bool(lon.min() <= roi.lon_min and lon.max() >= roi.lon_max
                      and lat.min() <= roi.lat_min and lat.max() >= roi.lat_max)
    return {
        "produto": "DTU15MDT 2 arc-min",
        "arquivo": str(DTU15MDT).replace("\\", "/"),
        "datum": attrs.get("vertical_definition", ""),
        "geoide_de_referencia": attrs.get("geoid_reference", ""),
        "crs": "EPSG:4326 (grade lon/lat; x/y EPSG:3031 anexos no arquivo)",
        "resolucao_graus": float(abs(np.diff(lon)[0])),
        "grade": [int(v) for v in mdt.shape],
        "extensao_lon": [float(lon.min()), float(lon.max())],
        "extensao_lat": [float(lat.min()), float(lat.max())],
        "contem_a_roi": contem_roi,
        "celulas_na_roi": int(in_roi.sum()),
        "fracao_finita_na_roi": float(np.isfinite(mdt[in_roi]).mean()),
        "min_m": float(np.nanmin(mdt)), "max_m": float(np.nanmax(mdt)),
        "ressalva": ("cobertura de VALORES, não de observações: o MDT é um "
                     "produto de altimetria de satélite sobre oceano aberto e, "
                     "sob plataforma de gelo, os valores vêm de "
                     "extrapolação/preenchimento do modelo, não de medida "
                     "direta. A incerteza sob plataforma não é a mesma do mar "
                     "aberto e precisa ser tratada como termo próprio."),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data" / "processed" / "vertical_reference_audit.json")
    args = ap.parse_args()

    cfg = load_config()
    roi = ROIBox.from_config(cfg)
    result = {
        "roi": {"lon_min": roi.lon_min, "lon_max": roi.lon_max,
                "lat_min": roi.lat_min, "lat_max": roi.lat_max},
        "eigen_6c4": _audit_geoid(roi, args.data_dir),
        "dtu15mdt": _audit_mdt(roi, args.data_dir),
        "geoid_legacy_parquet": {
            "jja": "INUTILIZÁVEL — coluna 100% nula nos 4.681 arquivos",
            "djf": "presente, mas NÃO usar como fonte científica principal",
            "decisao": ("as duas janelas usam a mesma fonte externa "
                        "(EIGEN-6C4 do BedMachine), nunca o campo do Parquet"),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
