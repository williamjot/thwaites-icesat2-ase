"""Recorta o GEMB continental (FAC e SMB, 10 km) à janela das cinco plataformas do basal v2.

O recorte antigo (`gemb_*_ase_2018_2025.nc`, 614 células) só cobre Thwaites–Crosson–Dotson.
Este cobre a janela de todo o flutuante da máscara da Fase 1b, com 50 km de margem.

Fonte: Zenodo 10.5281/zenodo.19266798 (GEMB v1.0 ERA5, registro v1.5), CC-BY-4.0.
Os continentais (2 × 1,1 GB) devem estar em ``data/forcing_download_tmp/`` com o MD5 conferido.
Depois do recorte eles são APAGADOS (``--keep`` para manter), como manda a regra do projeto de
não guardar o bruto pesado.

Construção igual à do recorte antigo: total = anomalia (dFAC / accum_SMB) + campo centrado.
Só ficam células finitas em todas as épocas. Nada é extrapolado.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TMP = ROOT / "data" / "forcing_download_tmp"
OUT = ROOT / "data" / "basal_forcing"
ITEMS = {
    "fac": ("GEMB_Antarctica_FAC_1979_2025_mesh_10km_v1_5.nc", "1aff50efd7bf0d67a8da228f565cd962",
            "dFAC", "centered_FAC", "gemb_fac_roi_2018_2025.nc", "m", "firn air content (total)"),
    "smb_accum": ("GEMB_Antarctica_SMB_1979_2025_mesh_10km_v1_5.nc",
                  "74fef5ecbba4feee08dc5c22be66a8cd", "accum_SMB", "centered_SMB",
                  "gemb_smb_accum_roi_2018_2025.nc", "m ice equivalent",
                  "accumulated surface mass balance (total, m ice eq.)"),
}
T_MIN = 2018.0
PAD_M = 50_000.0


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--keep", action="store_true", help="não apagar os continentais")
    args = ap.parse_args()
    z = np.load(ROOT / "outputs/basal_v2/fase1b/mascara.npz")
    ii, jj = np.nonzero(z["cls"] > 0)
    x0, x1 = z["x"][jj].min() - PAD_M, z["x"][jj].max() + PAD_M
    y0, y1 = z["y"][ii].min() - PAD_M, z["y"][ii].max() + PAD_M
    sources = []
    try:
        for var, (fname, digest, anom, centred, out_name, units, long_name) in ITEMS.items():
            src = TMP / fname
            sources.append(src)
            got = md5(src)
            if got != digest:
                raise ValueError(f"MD5 de {fname} não confere: {got}")
            with nc.Dataset(src) as d:
                gx, gy, t = (np.asarray(d[k][:], float) for k in ("x", "y", "time"))
                jx = np.flatnonzero((gx >= x0) & (gx <= x1))
                iy = np.flatnonzero((gy >= y0) & (gy <= y1))
                it = np.flatnonzero(t >= T_MIN)
                sl = (slice(it[0], it[-1] + 1), slice(iy[0], iy[-1] + 1),
                      slice(jx[0], jx[-1] + 1))
                a = np.ma.filled(np.ma.asarray(d[anom][sl], dtype=float), np.nan)
                c = np.ma.filled(np.ma.asarray(d[centred][sl[1:]], dtype=float), np.nan)
            total = a + c[None]
            xx, yy = np.meshgrid(gx[jx], gy[iy])
            ok = np.isfinite(total).all(axis=0)
            vals = total[:, ok]
            with nc.Dataset(OUT / out_name, "w") as o:
                o.createDimension("time", len(it))
                o.createDimension("point", int(ok.sum()))
                for name, data, dims, un in (("time", t[it], ("time",), "decimal year"),
                                             ("x", xx[ok], ("point",), "m"),
                                             ("y", yy[ok], ("point",), "m")):
                    v = o.createVariable(name, "f8", dims)
                    v[:] = data
                    v.units = un
                v = o.createVariable(var, "f8", ("time", "point"), zlib=True)
                v[:] = vals
                v.units, v.long_name = units, long_name
                o.source_file, o.source_md5 = fname, digest
                o.source_doi = "10.5281/zenodo.19266798"
                o.source_version = "GEMB v1.0 ERA5, record v1.5 (2026-03-27)"
                o.crs = "EPSG:3031"
                o.construction = f"{anom} + {centred}"
                o.extrapolation_policy = "forbidden"
                o.window = f"x {x0:.0f}..{x1:.0f}, y {y0:.0f}..{y1:.0f}; t >= {T_MIN}"
            print(f"{out_name}: {int(ok.sum())} células de {ok.size} na janela, "
                  f"{len(it)} épocas ({t[it][0]:.3f}–{t[it][-1]:.3f})")
    finally:
        if not args.keep:
            for s in sources:
                s.unlink(missing_ok=True)
                print(f"apagado: {s.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
