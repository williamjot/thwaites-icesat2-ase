"""Espessura hidrostática e sua série temporal por plataforma (passos 4 e 5 do plano basal).

Passo 4: espessura por segmento, só dentro da máscara hidrostática medida
(``data/interim/hydrostatic_mask_500m.nc``):

    h_mar = h_elv − (CATS + DAC) − (geoide EIGEN-6C4 do ATL06 + DTU15MDT)
    H     = (h_mar − FAC) · ρw / (ρw − ρi),       ρw = 1027, ρi = 917 kg/m³

- Maré: CATS2008-v2023, a mesma usada para medir β. O CATS já contém mf e mm, por isso o
  ``tide_equilibrium`` do ATL06 **não** é somado (contaria o longo período duas vezes). O
  patamar de β medido (0,95–1,07) indica erro de amplitude de até ~7%, ou seja, ~2–3 cm de
  ruído em h e ~0,2–0,3 m em H; isso é declarado, não corrigido.
- FAC (ar no firn): GEMB v1.5 (principal, 2018–2025) e IMAU-FDM v1.2A (alternativa, até
  2023,98). Interpolação linear no espaço (triangulação dos pontos do modelo) e no tempo.
  Segmento mais distante do ponto válido mais próximo do que um espaçamento da grade do modelo
  (GEMB 10 km, IMAU 27 km) é descartado: a política dos recortes é não extrapolar. Um limite
  fixo de 10 km cortava 58% dos segmentos no IMAU (mediana da distância ao ponto: 11 km).
- Sistema de maré permanente (tide-free × mean-tide) entre o geoide e o MDT não foi
  harmonizado; é um deslocamento quase constante que afeta o nível de H, não a sua variação.

Passo 5: série de espessura por plataforma e estação-ano, por efeitos fixos:

    H_i = a_b + gx_b·dx + gy_b·dy + s_k(i)

``b`` = célula de 500 m; ``k`` = estação-ano (28: DJF, MAM, JJA e SON de 2019 a 2025; DJF
recebe o ano de janeiro). A topografia da célula é eliminada exatamente e as 28 incógnitas s_k
saem de um sistema 28×28; a cobertura que muda de estação para estação não entra como sinal.
Duas versões:
- **Lagrangiana** (principal): célula na posição material em 2022,0 (ITS_LIVE anual, RK2), ou
  seja, s_k é a variação média de espessura de parcelas: o DH/Dt do balanço de massa;
- **Euleriana**: célula fixa. Serve de verificação cruzada.
Erro de s_k: jackknife por grupos de passagens (20 grupos aleatórios de ``(rgt, cycle)``); a
passagem é a unidade independente.

Limitação declarada: s_k é a média ponderada pela informação de cada célula, não a média por
área. Se a cobertura se correlacionar com a taxa de afinamento (por exemplo, mais dado longe da
linha de aterramento), há viés. A média por área de tendências por célula sai como verificação.

Etapas com cache em ``data/interim/thickness/``: A extração, B maré CATS, C espessura, D séries.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr
from scipy.spatial import Delaunay, cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipelines" / "derretimento_basal"))
from thwaites import load_config  # noqa: E402
from thwaites.corrections.cats_tide import CatsTidePredictor  # noqa: E402
from run_tide_response_lagrangian import load_velocity, advect_to_ref  # noqa: E402

STORE = ROOT / "data" / "ase_v2"
MASK = ROOT / "data" / "interim" / "hydrostatic_mask_500m.nc"
MDT = ROOT / "data" / "vertical_reference" / "dtu15mdt_ase.nc"
GEMB = ROOT / "data" / "basal_forcing" / "gemb_fac_roi_2018_2025.nc"
IMAU = ROOT / "data" / "basal_forcing" / "imau_fdm_fac_ase_2018_2023.nc"
CACHE = ROOT / "data" / "interim" / "thickness"
OUT = ROOT / "outputs" / "thickness_ase"
RHO_W, RHO_I = 1027.0, 917.0
FACTOR = RHO_W / (RHO_W - RHO_I)
CELL = 500.0
GRID_SPACING = {"gemb": 10_000.0, "imau": 27_000.0}
N_JACK = 20
SHELVES = {1: "Pine_Island", 2: "Thwaites", 3: "Crosson", 4: "Dotson", 5: "Cosgrove"}
VARIANTS = ["classe", "classe_f080", "classe_f095", "classe_ic"]
COLS = ["x", "y", "lon", "lat", "h_elv", "t_year", "rgt", "cycle", "dac", "geoid",
        "atl06_quality_summary"]
SEASON_OF_MONTH = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
SEASONS = ["DJF", "MAM", "JJA", "SON"]


# ------------------------------------------------------------------ A: extração
def stage_a():
    m = xr.open_dataset(MASK)
    xs, ys = m.x.values, m.y.values
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    cls = {v: m[v].values for v in VARIANTS}
    any_hyd = np.zeros(cls["classe"].shape, bool)
    for v in VARIANTS:
        any_hyd |= cls[v] == 1
    shelf = m.plataforma.values
    mdt = xr.open_dataset(MDT)
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / "segmentos.parquet"
    writer, n_tot = None, 0
    files = sorted(STORE.glob("*/*.parquet"))
    for n, f in enumerate(files, 1):
        d = pd.read_parquet(f, columns=COLS)
        d = d[(d["atl06_quality_summary"] == 0) & np.isfinite(d["h_elv"]) & (d["h_elv"] < 1e30)
              & np.isfinite(d["dac"]) & (np.abs(d["dac"]) < 1e30)
              & np.isfinite(d["geoid"]) & (np.abs(d["geoid"]) < 1e30)]
        if len(d):
            # ponto → pixel cujo centro é xs[j]: o pixel cobre xs[j] ± 250 m, daí o +0,5
            j = np.floor((d["x"].to_numpy() - xs[0]) / dx + 0.5).astype(int)
            i = np.floor((d["y"].to_numpy() - ys[0]) / dy + 0.5).astype(int)
            ok = (i >= 0) & (i < len(ys)) & (j >= 0) & (j < len(xs))
            d, i, j = d[ok], i[ok], j[ok]
            keep = any_hyd[i, j]
            d, i, j = d[keep], i[keep], j[keep]
        if len(d):
            mm = mdt["mdt"].interp(lat=xr.DataArray(d["lat"].to_numpy()),
                                   lon=xr.DataArray(d["lon"].to_numpy())).values
            ok = np.isfinite(mm)
            d, i, j, mm = d[ok], i[ok], j[ok], mm[ok]
            cols = {
                "x": d["x"].to_numpy(), "y": d["y"].to_numpy(),
                "lon": d["lon"].to_numpy(), "lat": d["lat"].to_numpy(),
                "h": d["h_elv"].to_numpy(),
                "h_ref": (d["geoid"].to_numpy() + mm).astype(np.float32),
                "t": d["t_year"].to_numpy(),
                "pass_id": (d["rgt"].to_numpy().astype(np.int32) * 1000
                            + d["cycle"].to_numpy().astype(np.int32)),
                "dac": d["dac"].to_numpy().astype(np.float32),
                "granule_month": np.full(len(d), int(f.name.split("_")[1][4:6]), np.int8),
                "granule_year": np.full(len(d), int(f.name.split("_")[1][:4]), np.int16),
                "shelf": shelf[i, j].astype(np.int8)}
            for v in VARIANTS:
                cols[v] = cls[v][i, j].astype(np.int8)
            t = pa.table(cols)
            if t.num_rows:
                if writer is None:
                    writer = pq.ParquetWriter(out, t.schema, compression="zstd")
                writer.write_table(t)
                n_tot += t.num_rows
        if n % 500 == 0:
            print(f"  A: {n}/{len(files)} arquivos, {n_tot:,} segmentos", flush=True)
    if writer is not None:
        writer.close()
    print(f"A: {n_tot:,} segmentos no flutuante hidrostático (união das variantes)", flush=True)


# ------------------------------------------------------------------ B: maré
def stage_b(cfg):
    with CatsTidePredictor(cfg) as pred:
        for k, name in SHELVES.items():
            out = CACHE / f"mare_{name}.parquet"
            if out.is_file():
                continue
            d = pq.read_table(CACHE / "segmentos.parquet",
                              filters=[("shelf", "=", k)]).to_pandas()
            tide = np.empty(len(d), np.float32)
            step = int(cfg.cats.chunk_size)
            for a in range(0, len(d), step):
                tide[a:a + step] = pred.predict(d["lon"].to_numpy()[a:a + step],
                                                d["lat"].to_numpy()[a:a + step],
                                                d["t"].to_numpy()[a:a + step])
            d["tide_cats"] = tide
            d.to_parquet(out, index=False, compression="zstd")
            print(f"B: maré {name}: {len(d):,} segmentos", flush=True)
            del d, tide
            gc.collect()


# ------------------------------------------------------------------ C: espessura
class FirnField:
    """FAC de um modelo em pontos dispersos: linear no espaço (Delaunay) e no tempo."""

    def __init__(self, path, max_dist):
        self.max_dist = max_dist
        d = xr.open_dataset(path)
        a = np.ma.filled(np.ma.masked_invalid(d["fac"].values.astype(float)), np.nan)
        ok = np.isfinite(a).all(axis=0)
        self.xy = np.c_[d["x"].values[ok], d["y"].values[ok]]
        self.fac = a[:, ok]
        self.time = d["time"].values.astype(float)
        self.tri = Delaunay(self.xy)
        self.tree = cKDTree(self.xy)

    def __call__(self, x, y, t):
        p = np.c_[x, y]
        out = np.full(len(x), np.nan)
        near, _ = self.tree.query(p)
        s = self.tri.find_simplex(p)
        inside = (s >= 0) & (near <= self.max_dist) & (t >= self.time[0]) & (t <= self.time[-1])
        if not inside.any():
            return out, near
        T = self.tri.transform[s[inside]]
        b = np.einsum("ijk,ik->ij", T[:, :2], p[inside] - T[:, 2])
        w = np.c_[b, 1 - b.sum(1)]
        verts = self.tri.simplices[s[inside]]
        k1 = np.clip(np.searchsorted(self.time, t[inside]), 1, len(self.time) - 1)
        a = (t[inside] - self.time[k1 - 1]) / (self.time[k1] - self.time[k1 - 1])
        # indexação direta (época, vértice): nunca materializar fac[k1] inteiro (n × npts)
        v0 = self.fac[(k1 - 1)[:, None], verts]
        v1 = self.fac[k1[:, None], verts]
        out[inside] = (((1 - a)[:, None] * v0 + a[:, None] * v1) * w).sum(1)
        return out, near


def stage_c():
    gemb = FirnField(GEMB, GRID_SPACING["gemb"])
    imau = FirnField(IMAU, GRID_SPACING["imau"])
    for name in SHELVES.values():
        out = CACHE / f"espessura_{name}.parquet"
        if out.is_file():
            continue
        d = pd.read_parquet(CACHE / f"mare_{name}.parquet")
        h_sea = d["h"] - d["tide_cats"].astype(float) - d["dac"].astype(float) \
            - d["h_ref"].astype(float)
        fg, near = np.empty(len(d)), np.empty(len(d))
        fi = np.empty(len(d))
        step = 500_000
        for a in range(0, len(d), step):
            sl = slice(a, a + step)
            fg[sl], near[sl] = gemb(d["x"].to_numpy()[sl], d["y"].to_numpy()[sl],
                                    d["t"].to_numpy()[sl])
            fi[sl], _ = imau(d["x"].to_numpy()[sl], d["y"].to_numpy()[sl], d["t"].to_numpy()[sl])
        d["h_sea"] = h_sea.astype(np.float32)
        d["fac_gemb"] = fg.astype(np.float32)
        d["fac_imau"] = fi.astype(np.float32)
        d["H_gemb"] = ((h_sea - fg) * FACTOR).astype(np.float32)
        d["H_imau"] = ((h_sea - fi) * FACTOR).astype(np.float32)
        d = d.drop(columns=["lon", "lat"])
        d.to_parquet(out, index=False, compression="zstd")
        print(f"C: {name}: {len(d):,} segmentos; H_gemb finito em "
              f"{np.isfinite(d['H_gemb']).mean():.1%}, mediana {np.nanmedian(d['H_gemb']):.0f} m; "
              f"H_imau finito em {np.isfinite(d['H_imau']).mean():.1%}", flush=True)
        del d
        gc.collect()


# ------------------------------------------------------------------ D: séries
def season_year(month, year):
    s = np.vectorize(SEASON_OF_MONTH.get)(month)
    yy = year + (month == 12)
    return s, yy


def fe_series(px, py, H, k, nk, w=None):
    """Efeitos fixos: H = a_b + plano_b + s_k. Elimina (a, gx, gy) por célula e resolve s."""
    key = np.c_[np.floor(px / CELL), np.floor(py / CELL)].astype(np.int64)
    _, gb = np.unique(key, axis=0, return_inverse=True)
    gb = gb.ravel()
    nb = gb.max() + 1
    cx = (np.floor(px / CELL) + 0.5) * CELL
    cy = (np.floor(py / CELL) + 0.5) * CELL
    X = np.c_[np.ones_like(H), (px - cx) / 100.0, (py - cy) / 100.0]
    p = 3
    A = np.zeros((nb, p, p))
    for a in range(p):
        for b in range(a, p):
            v = np.bincount(gb, weights=X[:, a] * X[:, b], minlength=nb)
            A[:, a, b] = A[:, b, a] = v
    cond = np.linalg.cond(A)
    okb = np.isfinite(cond) & (cond < 1e8)
    Ai = np.zeros_like(A)
    Ai[okb] = np.linalg.inv(A[okb])
    r = okb[gb]
    # X_b'D_b (nb, p, nk) e X_b'H
    XD = np.zeros((nb, p, nk))
    for a in range(p):
        XD[:, a, :] = np.bincount(gb[r] * nk + k[r], weights=X[r, a],
                                  minlength=nb * nk).reshape(nb, nk)
    DD = np.bincount(k[r], minlength=nk).astype(float)
    XH = np.stack([np.bincount(gb[r], weights=X[r, a] * H[r], minlength=nb) for a in range(p)], 1)
    DH = np.bincount(k[r], weights=H[r], minlength=nk)
    AiXD = np.einsum("bij,bjk->bik", Ai, XD)
    M = np.diag(DD) - np.einsum("bik,bil->kl", XD, AiXD)
    g = DH - np.einsum("bik,bi->k", AiXD, XH)
    # s é definido a menos de uma constante (absorvida por a_b): fixa média zero
    M1 = np.vstack([np.c_[M, np.ones(nk)], np.r_[np.ones(nk), 0.0]])
    sol = np.linalg.lstsq(M1, np.r_[g, 0.0], rcond=None)[0][:nk]
    sol[DD == 0] = np.nan
    return sol, DD


def ice_presence_filter(x, y, h_sea):
    """Remove segmentos que não são gelo de plataforma (água aberta pós-desprendimento, mélange).

    Critério relativo: altura sobre o mar < 50% da mediana da célula FIXA de 500 m. O maior
    ∂H/∂t euleriano medido é ~8 m/a (≲ 1 m/a de altura), ou seja, ≲ 15% da altura em 7 anos;
    uma queda de 50% não é afinamento. Não há vale na distribuição que justifique um corte
    absoluto (medido: cauda contínua de 0 a 30 m na Thwaites, na PIG e no Crosson). Também sai
    a célula cuja mediana é < 5 m (célula dominada por água ou mélange).
    """
    key = np.floor(x / CELL).astype(np.int64) * 10_000_000 + np.floor(y / CELL).astype(np.int64)
    med = pd.Series(h_sea).groupby(key).transform("median").to_numpy()
    return (h_sea >= 0.5 * med) & (med >= 5.0)


def speed_split(x, y, thr=1000.0):
    """Thwaites: TEIS (lenta) × TWGT/tronco (rápida). Corte de 1.000 m/a no vale da distribuição
    de velocidade média de 2019–2021 (modo lento em 600–800 m/a; cauda rápida até 3,8 km/a).
    Área rápida: 1.132 km² com corte de 1.000 e 1.365 km² com 800 m/a."""
    v = xr.open_dataset(ROOT / "data" / "velocity_itslive_annual.nc")
    pad = 20e3
    ys = v.y.values
    ysl = slice(y.min() - pad, y.max() + pad) if ys[0] < ys[-1] else slice(y.max() + pad,
                                                                         y.min() - pad)
    s = v.sel(x=slice(x.min() - pad, x.max() + pad), y=ysl).isel(time=[0, 1, 2]).mean("time")
    sp = np.hypot(s.vx, s.vy).interp(x=xr.DataArray(x), y=xr.DataArray(y)).values
    return np.where(np.isfinite(sp), sp > thr, False)


def stage_d(variant="classe", jack=True):
    rows = []
    units = []
    for k_shelf, name in SHELVES.items():
        d = pd.read_parquet(CACHE / f"espessura_{name}.parquet",
                            columns=["x", "y", "t", "pass_id", "granule_month", "granule_year",
                                     "h_sea", "H_gemb", "H_imau", variant])
        d = d[(d[variant] == 1) & np.isfinite(d["H_gemb"])].reset_index(drop=True)
        keep = ice_presence_filter(d["x"].to_numpy(), d["y"].to_numpy(), d["h_sea"].to_numpy())
        print(f"D: {name}: filtro de presença de gelo removeu {1 - keep.mean():.1%}", flush=True)
        d = d[keep].reset_index(drop=True)
        if name == "Thwaites":
            fast = speed_split(d["x"].to_numpy(), d["y"].to_numpy())
            units += [("Thwaites_TEIS", d[~fast].reset_index(drop=True)),
                      ("Thwaites_TWGT", d[fast].reset_index(drop=True))]
        else:
            units.append((name, d))
    for name, d in units:
        s_idx, yy = season_year(d["granule_month"].to_numpy(), d["granule_year"].to_numpy())
        k = ((yy - 2019) * 4 + s_idx).astype(int)
        good = (k >= 0) & (k < 28)
        d, k = d[good].reset_index(drop=True), k[good]
        x, y, t = d["x"].to_numpy(), d["y"].to_numpy(), d["t"].to_numpy()
        vel = load_velocity(x, y)
        X0, Y0 = advect_to_ref(x, y, t, vel)
        del vel
        gc.collect()
        H = d["H_gemb"].to_numpy(float)
        pid = d["pass_id"].to_numpy()
        res = {}
        for frame, (px, py) in {"lagr": (X0, Y0), "eul": (x, y)}.items():
            s, n = fe_series(px, py, H, k, 28)
            if jack:
                upid = np.unique(pid)
                grp = np.random.default_rng(7).integers(0, N_JACK, len(upid))
                gp = grp[np.searchsorted(upid, pid)]
                jk = np.array([fe_series(px[gp != g], py[gp != g], H[gp != g], k[gp != g], 28)[0]
                               for g in range(N_JACK)])
                se = np.sqrt((N_JACK - 1) / N_JACK
                             * np.nansum((jk - np.nanmean(jk, 0)) ** 2, 0))
            else:
                se = np.full(28, np.nan)
            res[frame] = (s, se, n)
        okI = np.isfinite(d["H_imau"].to_numpy())
        sI, _ = fe_series(X0[okI], Y0[okI], d["H_imau"].to_numpy(float)[okI], k[okI], 28)
        for kk in range(28):
            rows.append({"plataforma": name, "variante": variant,
                         "ano": 2019 + kk // 4, "estacao": SEASONS[kk % 4],
                         "t_meio": 2019 + kk // 4 + [0.0, 0.25, 0.5, 0.75][kk % 4] + 0.04,
                         "dH_lagr_m": res["lagr"][0][kk], "se_lagr_m": res["lagr"][1][kk],
                         "dH_eul_m": res["eul"][0][kk], "se_eul_m": res["eul"][1][kk],
                         "dH_lagr_imau_m": sI[kk], "n_seg": int(res["lagr"][2][kk])})
        print(f"D: {name}: {len(H):,} segmentos, H mediano {np.median(H):.0f} m", flush=True)
        del d, X0, Y0, H
        gc.collect()
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / f"serie_espessura_{variant}.csv", index=False)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", default="ABCD")
    ap.add_argument("--variant", default="classe")
    ap.add_argument("--no-jack", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    if "A" in args.stage and not (CACHE / "segmentos.parquet").is_file():
        stage_a()
    if "B" in args.stage:
        stage_b(cfg)
    if "C" in args.stage:
        stage_c()
    if "D" in args.stage:
        out = stage_d(args.variant, jack=not args.no_jack)
        pd.set_option("display.width", 200, "display.max_rows", 200)
        print(out.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
