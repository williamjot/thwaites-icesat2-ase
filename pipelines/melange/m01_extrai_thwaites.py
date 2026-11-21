"""Mélange e fraturas da Thwaites, passo 1: segmentos ATL06 r007 da região da Thwaites, 2019–2025.

Fonte: data/ase_v2/{jja,djf,son,mam}/ATL06_*.parquet (já extraídos; nada é baixado).
Caixa EPSG:3031: x −1650..−1470 km, y −550..−370 km — língua oeste (tronco e mélange) até o mar
aberto, plataforma leste (TEIS), margem de cisalhamento e ponto de ancoragem.

Não filtra por atl06_quality_summary: sobre mélange e água o flag de qualidade rejeita justamente os
retornos que interessam. O flag vai como coluna (qs) e cada passo decide o que usar.
Valores de preenchimento do ATL06 são finitos (3,4e38; 127 em int8): viram NaN aqui.

Altura acima do geoide, corrigida de maré e de pressão (o h_li não as aplica sobre gelo flutuante,
e tide_ocean não inclui tide_equilibrium):
    hf = h_li − geoid − tide_ocean − tide_equilibrium − dac
Não remove a topografia dinâmica média do oceano: o nível da água é calibrado empiricamente
(passo 2) a partir dos próprios retornos de água.
t_year é ano juliano desde 2018 (delta_time / 365,25 d), não ano de calendário.
Saída: data/interim/melange/thwaites_atl06.parquet
"""
from __future__ import annotations

import glob
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "interim" / "melange"
X0, X1, Y0, Y1 = -1650e3, -1470e3, -550e3, -370e3
COLS = ["x", "y", "t_year", "delta_time", "h_elv", "geoid", "tide_ocean", "tide_equilibrium", "dac",
        "atl06_quality_summary", "h_robust_sprd", "dh_fit_dx", "segment_id", "x_atc", "rgt", "cycle", "beam",
        "n_fit_photons", "snr_significance", "cloud_flg_asr", "bsnow_conf"]
EST = {"jja": 0, "djf": 1, "son": 2, "mam": 3}


def _fill(a, lim=1e10):
    a = a.astype("float64")
    a[~np.isfinite(a) | (np.abs(a) > lim)] = np.nan
    return a


def le(args):
    f, est = args
    d = pd.read_parquet(f, columns=COLS)
    d = d[(d.x > X0) & (d.x < X1) & (d.y > Y0) & (d.y < Y1)]
    if d.empty:
        return None
    h = _fill(d.h_elv.to_numpy(), 1e4)
    corr = [_fill(d[c].to_numpy(), 1e3) for c in ("geoid", "tide_ocean", "tide_equilibrium", "dac")]
    hf = h - corr[0] - np.nan_to_num(corr[1]) - np.nan_to_num(corr[2]) - np.nan_to_num(corr[3])
    out = pd.DataFrame({
        "x": d.x.to_numpy(np.float32), "y": d.y.to_numpy(np.float32), "t": d.t_year.to_numpy(),
        "dt": d.delta_time.to_numpy(), "hf": hf.astype(np.float32),
        "mare_ok": np.isfinite(corr[1]) & np.isfinite(corr[3]),
        "qs": d.atl06_quality_summary.to_numpy(np.int8),
        "sprd": _fill(d.h_robust_sprd.to_numpy(), 1e4).astype(np.float32),
        "dhdx": _fill(d.dh_fit_dx.to_numpy(), 1e4).astype(np.float32),
        "seg": d.segment_id.to_numpy(np.int64), "xatc": d.x_atc.to_numpy(),
        "rgt": d.rgt.to_numpy(np.int16), "cycle": d.cycle.to_numpy(np.int16), "beam": d.beam.to_numpy(np.int8),
        "nfot": d.n_fit_photons.to_numpy(np.int32), "nuvem": d.cloud_flg_asr.to_numpy(np.int8),
        "est": np.int8(est), "gran": Path(f).stem,
    })
    return out[np.isfinite(out.hf)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tarefas = [(f, v) for k, v in EST.items() for f in sorted(glob.glob(str(ROOT / "data" / "ase_v2" / k / "*.parquet")))]
    with ProcessPoolExecutor(8) as ex:
        partes = [p for p in ex.map(le, tarefas, chunksize=16) if p is not None]
    d = pd.concat(partes, ignore_index=True)
    d.to_parquet(OUT / "thwaites_atl06.parquet", index=False)
    print(f"{len(tarefas)} grânulos lidos; {len(partes)} com dados na caixa; {len(d):,} segmentos")
    print(d.groupby("est").size().rename({0: "JJA", 1: "DJF", 2: "SON", 3: "MAM"}).to_dict())
    print("fração com maré/DAC presentes:", round(d.mare_ok.mean(), 3), "| qs==0:", round((d.qs == 0).mean(), 3))
    print("t:", d.t.min().round(3), "–", d.t.max().round(3))


if __name__ == "__main__":
    main()
