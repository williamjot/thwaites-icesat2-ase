"""Recuperar a faixa junto à linha de aterramento corrigindo a espessura, em vez de excluí-la.

Na faixa de 0–15 km a jusante da linha de aterramento do BedMachine, a espessura hidrostática
erra por dois motivos medidos neste projeto:

1. **Maré.** Na zona de flexão a superfície responde só a uma fração β da maré
   (``run_tide_response_lagrangian.py``). Subtrair a maré cheia deixa (1 − β)·T de ruído na
   altura. Correção: subtrair β_n·(CATS + DAC), com β_n = clip(β/P, 0, 1) (P = patamar da
   plataforma). β por célula de 500 m quando σβ < 0,3; senão, a mediana de β/P por faixa de
   250 m de distância à linha.
2. **Flexão/apoio.** Mesmo com a maré certa, a flutuação superestima H perto da linha. O radar
   MCoRDS normalizado por voo mediu a razão H_radar/H_hidro em função da distância
   (``perfil_razao_normalizado_por_voo.csv``; Thwaites 0,91 a 0–0,5 km, 0,94–0,95 a 2–4 km).
   Correção: H × c(d), com c interpolado entre os centros das faixas e c = 1 além de 12 km.
   **Limite:** o radar é de 2009–2018 e a distância é à linha do BedMachine.

Calcula-se a cadeia inteira duas vezes (sem correção e com correção), com o mesmo método do
produto: altura → H (firn GEMB) → filtro de presença de gelo → tendência por célula material de
500 m (Lagrangiana, ITS_LIVE) → m_b = SMB − DH/Dt − H·∇·u (escala ótima da plataforma).
Nada é interpolado: os valores são por célula com dado.

Teste: ApRES na zona de aterramento da TEIS em 2020 (Davis et al. 2023: médias de 2,0–5,4 m/a;
G1–G4 processados aqui: 2,0–4,1) e os 40 pontos de Wild et al. 2024 no centro da TEIS.

Uso: ``python pipelines/run_gz_recovery_ase.py Thwaites [outras plataformas]``.
Saída: ``outputs/gz_recovery/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipelines" / "derretimento_basal"))
from run_basal_melt_ase import cell_trends, smb_rate_field  # noqa: E402
from run_flux_term_ase import grid_median  # noqa: E402
from run_thickness_ase import FirnField, GEMB, GRID_SPACING, ice_presence_filter, speed_split  # noqa: E402
from run_tide_response_lagrangian import advect_to_ref, load_velocity  # noqa: E402

TIDE = ROOT / "data" / "interim" / "tide_response"
BETA = ROOT / "outputs" / "tide_response_ase" / "beta_lagr_todas.parquet"
RADAR = ROOT / "outputs" / "mcords_hydro_test" / "perfil_razao_normalizado_por_voo.csv"
MASK = ROOT / "data" / "interim" / "hydrostatic_mask_500m.nc"
BED = ROOT / "data" / "NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc"
MDT = ROOT / "data" / "vertical_reference" / "dtu15mdt_ase.nc"
DIV = ROOT / "data" / "interim" / "divergence"
OUT = ROOT / "outputs" / "gz_recovery"
FACTOR = 1027.0 / 110.0
OPT_SCALE = {"Pine_Island": 3, "Thwaites": 2, "Crosson": 1, "Dotson": 1, "Cosgrove": 1}
BAND_KM = 15.0
# cortes do radar com tolerância de 3% (run_factorial_v2_ase.radar_cut)
RADAR_CUT = {"Pine_Island": 2.0, "Thwaites": 4.0, "Crosson": 0.0, "Dotson": 0.5, "Cosgrove": 1.5}


def beta_n(seg, shelf, P, dstar):
    b = pd.read_parquet(BETA)
    b = b[b.shelf == shelf]
    key_b = np.floor(b.x / 500).astype(np.int64) * 10_000_000 + np.floor(b.y / 500).astype(np.int64)
    good = b.se_beta < 0.3
    lut = dict(zip(key_b[good], np.clip(b.beta[good] / P, 0, 1)))
    prof = b.groupby(pd.cut(b.dist_gl_km, np.arange(0, BAND_KM + 0.25, 0.25)),
                     observed=True).beta.median() / P
    mids = np.array([i.mid for i in prof.index])
    fallback = np.clip(np.interp(seg.dist_gl.to_numpy() / 1e3, mids, prof.to_numpy(),
                                 left=prof.iloc[0], right=1.0), 0, 1)
    key = (np.floor(seg.x / 500).astype(np.int64) * 10_000_000
           + np.floor(seg.y / 500).astype(np.int64)).to_numpy()
    out = np.array([lut.get(k, np.nan) for k in key])
    out = np.where(np.isfinite(out), out, fallback)
    # Além de d* (zona hidrostática medida), β de célula é ruído em torno de 1: aplicá-lo pioraria
    # o centro da TEIS (medido: Wild +0,71 m/a contra −0,13). Lá vale a maré cheia, como no produto.
    return np.where(seg.dist_gl.to_numpy() / 1e3 >= dstar, 1.0, out)


def radar_factor(seg, shelf):
    p = pd.read_csv(RADAR)
    p = p[p.plataforma == shelf].copy()
    lo = p.dist_gl_km.str.extract(r"\(([-\d.]+),").astype(float)[0]
    hi = p.dist_gl_km.str.extract(r",\s*([-\d.]+)\]").astype(float)[0]
    p = p.assign(mid=(lo + hi) / 2)
    p = p[p.mid > 0].sort_values("mid")
    d = seg.dist_gl.to_numpy() / 1e3
    c = np.interp(d, p.mid.to_numpy(), p.razao_norm_med.to_numpy(), left=p.razao_norm_med.iloc[0], right=1.0)
    # Segue o perfil medido até a primeira faixa que fica a menos de 1% de 1 e usa 1 depois dela.
    # Duas versões anteriores falharam: (a) perfil até 12 km aplicava o ruído de ±2% além do viés
    # e criava afinamento falso no centro da TEIS; (b) rampa fixa de 2 km após o corte comprimia a
    # subida real (0,94 → 0,996 em ~4 km) e baixava m_b em ~2,6 m/a a 4–6 km.
    vals, mids = p.razao_norm_med.to_numpy(), p.mid.to_numpy()
    near1 = np.nonzero(np.abs(vals - 1) <= 0.01)[0]
    end = mids[near1[0]] if len(near1) else mids[-1]
    c = np.where(d <= end, c, 1.0)
    upper = p.dist_gl_km.str.extract(r",\s*([-\d.]+)\]").astype(float)[0].to_numpy()
    cut_end = upper[near1[0]] if len(near1) else upper[-1]
    ramp = np.clip((cut_end - d) / max(cut_end - end, 1e-6), 0, 1)
    return np.where(d <= end, c, 1.0 + ramp * (vals[near1[0]] - 1.0) if len(near1) else 1.0)


def run_chain(seg, H, xs, ys, div, tag):
    keep = ice_presence_filter(seg.x.to_numpy(), seg.y.to_numpy(), seg.h_sea_corr.to_numpy()
                               if tag == "corrigido" else seg.h_sea_raw.to_numpy())
    s = seg[keep].reset_index(drop=True)
    Hk = H[keep]
    ct = cell_trends(s.X0.to_numpy(), s.Y0.to_numpy(), s.t.to_numpy(), Hk, s.pass_id.to_numpy())
    Hg = grid_median(s.x.to_numpy(), s.y.to_numpy(), Hk, xs, ys)
    return ct, Hg


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    m = xr.open_dataset(MASK)
    xs, ys = m.x.values, m.y.values
    params = json.loads(m.classe.attrs["parametros"])
    bed = xr.open_dataset(BED)
    mdt = xr.open_dataset(MDT)
    tf = Transformer.from_crs(3031, 4326, always_xy=True)
    firn = FirnField(GEMB, GRID_SPACING["gemb"])
    for shelf in sys.argv[1:] or ["Thwaites"]:
        seg = pd.read_parquet(TIDE / f"mare_{shelf}.parquet")
        seg = seg[(seg.dist_gl > 0) & (seg.dist_gl <= BAND_KM * 1e3) & np.isfinite(seg.tide_cats)]
        seg = seg.reset_index(drop=True)
        X, Y = xr.DataArray(seg.x.to_numpy()), xr.DataArray(seg.y.to_numpy())
        lon, lat = tf.transform(seg.x.to_numpy(), seg.y.to_numpy())
        href = bed.geoid.interp(x=X, y=Y).values + np.nan_to_num(
            mdt.mdt.interp(lat=xr.DataArray(lat), lon=xr.DataArray(lon)).values)
        P = params[shelf]["P"]
        bn = beta_n(seg, shelf, P, params[shelf]["d_star_km"]["f090"])
        c = radar_factor(seg, shelf)
        fac = np.empty(len(seg))
        for a in range(0, len(seg), 500_000):
            sl = slice(a, a + 500_000)
            fac[sl] = firn(seg.x.to_numpy()[sl], seg.y.to_numpy()[sl], seg.t.to_numpy()[sl])[0]
        T = seg.tide_cats.to_numpy(float) + seg.dac.to_numpy(float)
        seg["h_sea_raw"] = seg.h.to_numpy() - T - href
        seg["h_sea_corr"] = seg.h.to_numpy() - bn * T - href
        H_raw = FACTOR * (seg.h_sea_raw.to_numpy() - fac)
        H_cor = FACTOR * (seg.h_sea_corr.to_numpy() - fac) * c
        ok = np.isfinite(H_raw) & np.isfinite(H_cor)
        seg, H_raw, H_cor = seg[ok].reset_index(drop=True), H_raw[ok], H_cor[ok]
        vel = load_velocity(seg.x.to_numpy(), seg.y.to_numpy())
        X0, Y0 = advect_to_ref(seg.x.to_numpy(), seg.y.to_numpy(), seg.t.to_numpy(), vel)
        del vel
        seg["X0"], seg["Y0"] = X0, Y0
        div = xr.open_dataset(DIV / f"div_{OPT_SCALE[shelf]}km.nc").div_media.values
        smb, _, _ = smb_rate_field(xs, ys, np.ones(div.shape, bool))
        rows = []
        for tag, H in (("sem_correcao", H_raw), ("corrigido", H_cor)):
            ct, Hg = run_chain(seg, H, xs, ys, div, tag)
            j = np.floor((ct.x.to_numpy() - 250 - xs[0]) / (xs[1] - xs[0]) + 0.5).astype(int)
            i = np.floor((ct.y.to_numpy() - 250 - ys[0]) / (ys[1] - ys[0]) + 0.5).astype(int)
            okp = (i >= 0) & (i < len(ys)) & (j >= 0) & (j < len(xs))
            ct = ct[okp].assign(i=i[okp], j=j[okp])
            ct["H"] = Hg[ct.i, ct.j]
            ct["divu"] = div[ct.i, ct.j]
            ct["SMB"] = smb[ct.i, ct.j]
            ct["dist_gl_km"] = m.dist_gl_km.values[ct.i, ct.j]
            ct["mb"] = ct["SMB"] - ct["dHdt"] - ct["H"] * ct["divu"]
            ct["versao"] = tag
            rows.append(ct)
        r = pd.concat(rows, ignore_index=True)
        r = r[np.isfinite(r.mb) & np.isfinite(r.dist_gl_km)]
        if shelf == "Thwaites":
            fast = speed_split(r.x.to_numpy(), r.y.to_numpy())
            r["unidade"] = np.where(fast, "Thwaites_TWGT", "Thwaites_TEIS")
        else:
            r["unidade"] = shelf
        r.to_parquet(OUT / f"celulas_{shelf}.parquet", index=False)
        bins = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 11, 15]
        t = r.groupby(["unidade", "versao", pd.cut(r.dist_gl_km, bins)], observed=True).agg(
            n=("mb", "size"), mb_mediana=("mb", "median"), DHDt_mediana=("dHdt", "median"),
            se_mediano=("se", "median")).reset_index()
        t.to_csv(OUT / f"perfil_{shelf}.csv", index=False)
        pd.set_option("display.width", 200, "display.max_rows", 200)
        print(t.pivot_table(index=["unidade", "dist_gl_km"], columns="versao",
                            values="mb_mediana", observed=True).round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
