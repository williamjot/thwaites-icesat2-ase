# -*- coding: utf-8 -*-
"""Metodologia da serie TEIS aplicada a uma celula de GELO ATERRADO.

Replica pipelines/figures/run_teis_hi_seasonal_timeseries.py passo a passo,
mudando apenas o que a fisica do gelo aterrado exige ou o usuario escolheu:

  igual a TEIS
    * dados: recortes legacy_ase (ATL06 v007), granulos com join QC verificado
    * filtro de faixa: -500 < h_elv < 4000, s_elv > 0, feixes 1-6
    * quality_mask sobre o sidecar qc_flags (linhas por row_idx)
    * FAC: IMAU-FDM FDMv12A, IDW 4 vizinhos + linear no tempo, por observacao,
      sem extrapolacao; sigma de representatividade entre vizinhos
    * janela: intercepto por janela + plano dx/dy, Huber IRLS, pesos
      balanceados por bloco (RGT, ciclo), sigma = max(SE cluster-robusta,
      RMSE/sqrt(N_blocos)) (+) sigma_rep do FAC em quadratura
    * tendencia: Theil-Sen + Monte Carlo condicional a sigma; teste sazonal
  diferente, por fisica
    * bm_mask == grounded (e nao floating); sem freeboard, sem geoide/MDT
    * mare e DAC NAO subtraidas (contrato vertical: so sobre flutuante)
    * sem relacao hidrostatica: grandeza = h_ice = h - FAC (amplificacao 1)
    * sem controle de frente de calving (celula a 23 km da linha de aterramento)
  diferente, por escolha do usuario
    * superficie de referencia REMA: ajuste sobre h - REMA - FAC
    * janelas JJA 2019 ... JJA 2023 (todas cobertas pelo IMAU-FDM)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(r"D:/WILLIAM/ARTIGO PIBIC ANTARTICA")
sys.path.insert(0, str(ROOT))

from thwaites import load_config                                     # noqa: E402
from thwaites.atl06_granule_id import (                              # noqa: E402
    t_year_to_calendar_decimal_year, t_year_to_datetime)
from thwaites.glaciology.basal_forcing import (                      # noqa: E402
    PointForcingCube, sample_at_times)
from thwaites.legacy_fit import block_balanced_weights               # noqa: E402
from thwaites.legacy_qc_join import QC_JOIN_VERIFIED                 # noqa: E402
from thwaites.qc.atl06_flags import quality_mask                     # noqa: E402
from thwaites.corrections.slope import (                             # noqa: E402
    resolve_rema_path, sample_rema_bilinear)
from thwaites.corrections.gia import GIAField                        # noqa: E402

SCRATCH = Path(__file__).resolve().parent
# Definidos em main(); os valores abaixo preservam o comportamento historico
# do script quando ele roda sem argumentos.
OUT = ROOT / "outputs" / "grounded_H_timeseries" / "nucleo_metodo_TEIS"
RAW_PARQUET = ROOT / "data" / "processed" / "grounded_legacy_cells_raw.parquet"
PREV_SERIE = (ROOT / "outputs" / "grounded_H_timeseries"
              / "nucleo_afinamento_thwaites" / "serie_inverno_verao.csv")


class _Thick:
    """Utilitários que vinham de run_legacy_thickness.py (arquivado em 2026-10-18)."""

    FDM_FAC = Path("basal_forcing") / "imau_fdm_fac_ase_2018_2023.nc"
    QC_NEEDED = ["atl06_quality_summary", "h_robust_sprd", "msw_flag", "dem_h"]

    @staticmethod
    def _read_parquet_rows(path: Path, columns: list[str], positions: np.ndarray) -> pd.DataFrame:
        """Lê posições globais do sidecar QC preservando a ordem, por row-group."""
        import pyarrow.parquet as pq

        positions = np.asarray(positions, dtype="int64")
        if positions.size == 0:
            return pd.DataFrame(columns=columns)
        pf = pq.ParquetFile(path)
        tamanhos = np.array([pf.metadata.row_group(i).num_rows
                             for i in range(pf.metadata.num_row_groups)], dtype="int64")
        inicios = np.r_[0, np.cumsum(tamanhos)[:-1]]
        grupos = np.searchsorted(np.cumsum(tamanhos), positions, side="right")
        partes = {int(g): pf.read_row_group(int(g), columns=columns).to_pandas() for g in np.unique(grupos)}
        linhas = [partes[int(g)].iloc[int(pos - inicios[int(g)])] for pos, g in zip(positions, grupos)]
        return pd.DataFrame(linhas).reset_index(drop=True)


THICK = _Thick

CELL_NAME = "nucleo"
CX, CY = -1462500.0, -642500.0
HALF = 2500.0


def build_windows(ultimo_jja: int):
    """JJA 2019, DJF 2019/2020, JJA 2020, ... ate JJA <ultimo_jja>."""
    janelas = []
    for ano in range(2019, ultimo_jja + 1):
        janelas.append(f"JJA {ano}")
        if ano < ultimo_jja:
            janelas.append(f"DJF {ano}/{ano + 1}")
    return tuple(janelas)


# Definidos em main() pelos argumentos de linha de comando.
# COM_FIRN=False e obrigatorio para ir alem de JJA 2023: o recorte do IMAU-FDM
# termina em 2023,98 e a extrapolacao e proibida no produto.
TARGET_WINDOWS = build_windows(2025)
WINDOW_ORDER = {w: i for i, w in enumerate(TARGET_WINDOWS)}
COM_FIRN = False
ACEITA_JOIN_NAO_VERIFICADO = False
MIN_OBS_WINDOW = 30
BOOTSTRAP_DRAWS = 20_000
BM_GROUNDED = 2
RHO_I = 917.0
AREA_M2 = 25.0e6
GT_PER_M = RHO_I * AREA_M2 / 1e12


def window_from_t_year(values):
    """Copia literal de run_teis_hi_seasonal_timeseries.window_from_t_year."""
    labels = []
    for value in np.asarray(values, dtype=float):
        date = t_year_to_datetime(float(value))
        if date.month in (6, 7, 8):
            labels.append(f"JJA {date.year}")
        elif date.month == 12:
            labels.append(f"DJF {date.year}/{date.year + 1}")
        elif date.month in (1, 2):
            labels.append(f"DJF {date.year - 1}/{date.year}")
        else:
            labels.append("OUTSIDE")
    return np.asarray(labels, dtype=object)


# ------------------------------------------------------------------ dados
def load_observations(cfg):
    raw = pd.read_parquet(RAW_PARQUET)
    audit = {"linhas_na_celula": int((raw.cell == CELL_NAME).sum())}
    raw = raw[raw.cell == CELL_NAME].copy()
    raw["window"] = window_from_t_year(raw.t_year.to_numpy())
    raw = raw[raw.window.isin(TARGET_WINDOWS)].copy()
    audit["linhas_nas_janelas"] = int(len(raw))
    audit["bm_mask_contagem"] = {str(k): int(v) for k, v in raw.bm_mask.value_counts().items()}
    raw = raw[raw.bm_mask == BM_GROUNDED].copy()
    audit["linhas_aterradas"] = int(len(raw))

    stats = pd.read_csv(ROOT / "data" / "processed" / "legacy_granule_qc_stats.csv",
                        usecols=["source_file", "join_verdict"])
    verified = set(stats.loc[stats.join_verdict == QC_JOIN_VERIFIED, "source_file"])
    n0 = len(raw)
    if ACEITA_JOIN_NAO_VERIFICADO:
        # O veredito do pareamento e fail-closed: reprova tambem quando o teste
        # por linha nao e unanime nos seis feixes, mesmo com todos os testes
        # estruturais aprovados. Aceitar esses granulos e uma decisao do
        # usuario, e fica registrada grânulo a grânulo no relatorio - nao se
        # apaga a diferenca entre "verificado" e "admitido sem verificacao".
        usados = sorted(set(raw.source_file.unique()) - verified)
        audit["granulos_join_nao_verificado_aceitos"] = usados
        audit["linhas_de_join_nao_verificado"] = int(
            (~raw.source_file.isin(verified)).sum())
        audit["rej_granulo_sem_join"] = 0
    else:
        raw = raw[raw.source_file.isin(verified)].copy()
        audit["rej_granulo_sem_join"] = int(n0 - len(raw))

    h = raw.h_elv.to_numpy("float64")
    s = raw.s_elv.to_numpy("float64")
    t = raw.t_year.to_numpy("float64")
    full = (np.isfinite(h) & (h > -500.0) & (h < 4000.0) & np.isfinite(s)
            & (s > 0.0) & np.isfinite(t) & np.isin(raw.beam.to_numpy(), np.arange(1, 7)))
    audit["rej_full"] = int((~full).sum())
    audit["s_elv_acima_1e30_apos_full"] = int((full & (s > 1e30)).sum())
    raw = raw[full].copy()

    keep_all = np.zeros(len(raw), bool)
    raw = raw.reset_index(drop=True)
    for (season, gran), idx in raw.groupby(["season_file", "source_file"]).groups.items():
        idx = np.asarray(idx)
        qc_path = ROOT / "data" / season.lower() / "qc_flags" / gran
        if not qc_path.is_file():
            continue
        pos = raw.loc[idx, "row_idx"].to_numpy("int64")
        qc = THICK._read_parquet_rows(qc_path, THICK.QC_NEEDED, pos)
        keep_q, _ = quality_mask(qc, cfg, h_li=raw.loc[idx, "h_elv"].to_numpy("float64"))
        keep_all[idx] = np.asarray(keep_q, bool)
    audit["rej_qc"] = int((~keep_all).sum())
    obs = raw[keep_all].copy()

    # superficie aterrada: h_elv sem mare/DAC; referencia REMA
    rema = sample_rema_bilinear(obs.x.to_numpy(), obs.y.to_numpy(), resolve_rema_path(cfg))
    obs["rema"] = rema
    obs["h_res"] = obs.h_elv.to_numpy("float64") - rema
    ok = np.isfinite(obs.h_res) & (np.abs(obs.h_res) <= float(cfg.slope.max_slope_ref_m))
    audit["rej_rema"] = int((~ok).sum())
    obs = obs[ok].copy()

    cube = PointForcingCube.from_native_subset(ROOT / "data" / THICK.FDM_FAC, "fac")
    cal = t_year_to_calendar_decimal_year(obs.t_year.to_numpy())
    fac = sample_at_times(cube, obs.x.to_numpy(), obs.y.to_numpy(), cal)
    obs["fac"] = fac["value"]
    obs["fac_sigma_rep"] = fac["sigma_rep"]
    obs["fac_nearest_m"] = fac["nearest_distance_m"]
    audit["obs_sem_suporte_de_FAC"] = int((~np.isfinite(obs.fac)).sum())
    if COM_FIRN:
        ok = np.isfinite(obs.fac)
        audit["rej_fac_sem_suporte"] = int((~ok).sum())
        obs = obs[ok].copy()
        obs["h_ice_res"] = obs.h_res - obs.fac
    else:
        # Sem desconto de firn: a grandeza e h - REMA. O FAC continua amostrado
        # e gravado, mas so como diagnostico - e nao entra na sigma.
        audit["rej_fac_sem_suporte"] = 0
        obs["h_ice_res"] = obs.h_res

    obs["dx"] = obs.x - CX
    obs["dy"] = obs.y - CY
    obs["block"] = obs.rgt.astype("int64") * 100 + obs.cycle.astype("int64")
    audit["observacoes_finais"] = int(len(obs))
    audit["fac_vizinho_mais_proximo_km"] = float(np.median(obs.fac_nearest_m) / 1000.0)
    return obs, audit


# -------------------------------------------------------------- estimador
def joint_window_fit(obs):
    """_joint_window_fit da TEIS com valor h_ice_res e amplificacao 1."""
    support = (obs.groupby("window", observed=True)
               .agg(N_obs=("h_ice_res", "size"), N_blocks=("block", "nunique"))
               .reset_index())
    keep_windows = sorted(support.loc[support.N_obs >= MIN_OBS_WINDOW, "window"].tolist(),
                          key=WINDOW_ORDER.get)
    work = obs[obs.window.isin(keep_windows)].copy()
    cat = pd.Categorical(work.window, categories=keep_windows, ordered=True)
    dummy = np.zeros((len(work), len(keep_windows)))
    dummy[np.arange(len(work)), cat.codes] = 1.0
    design = np.column_stack([dummy, work.dx.to_numpy(float) / 1000.0,
                              work.dy.to_numpy(float) / 1000.0])
    values = work.h_ice_res.to_numpy(float)
    base_weight = block_balanced_weights(work.s_elv.to_numpy(float),
                                         work.block.to_numpy("int64"))
    aw = design * np.sqrt(base_weight)[:, None]
    norms = np.linalg.norm(aw, axis=0)
    norms = np.where(norms > 0, norms, 1.0)
    condition = float(np.linalg.cond(aw / norms))
    if not np.isfinite(condition) or condition > 100.0:
        raise ValueError(f"ajuste mal condicionado ({condition:.1f})")
    weight = base_weight.copy()
    beta = np.zeros(design.shape[1])
    for _ in range(12):
        root = np.sqrt(weight)
        new_beta, *_ = np.linalg.lstsq(design * root[:, None], values * root, rcond=None)
        residual = values - design @ new_beta
        scale = max(1.4826 * np.median(np.abs(residual - np.median(residual))), 0.02)
        huber = np.minimum(1.0, 1.345 * scale / np.maximum(np.abs(residual), 1e-12))
        new_weight = base_weight * huber
        if np.allclose(new_beta, beta, rtol=1e-8, atol=1e-9):
            beta, weight = new_beta, new_weight
            break
        beta, weight = new_beta, new_weight
    residual = values - design @ beta
    xtwx = design.T @ (weight[:, None] * design)
    bread = np.linalg.pinv(xtwx)
    meat = np.zeros_like(xtwx)
    blocks = work.block.to_numpy("int64")
    ub = np.unique(blocks)
    for b in ub:
        m = blocks == b
        score = design[m].T @ (weight[m] * residual[m])
        meat += np.outer(score, score)
    cov = bread @ meat @ bread
    n, p, g = len(work), design.shape[1], len(ub)
    if g > 1 and n > p:
        cov *= (g / (g - 1)) * ((n - 1) / (n - p))
    cluster_se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    lon, lat = Transformer.from_crs(3031, 4326, always_xy=True).transform(CX, CY)
    rows = []
    for k, w in enumerate(keep_windows):
        m = work.window.eq(w).to_numpy()
        part = work.loc[m]
        lw, lb = weight[m], base_weight[m]
        t_center = float(np.average(part.t_year.to_numpy(float), weights=lb))
        nb = int(part.block.nunique())
        rmse = float(np.sqrt(np.average(residual[m] ** 2, weights=lw)))
        intercept_se = max(float(cluster_se[k]), rmse / np.sqrt(max(nb, 1)))
        fv = part.fac_sigma_rep.to_numpy(float)
        fac_rep_ok = bool(np.isfinite(fv).any())
        fac_rep = float(np.nanmedian(fv)) if (COM_FIRN and fac_rep_ok) else 0.0
        rows.append({
            "cell_x": CX, "cell_y": CY, "lon": float(lon), "lat": float(lat),
            "window": w, "season": w[:3], "t_center_year": t_center,
            "t_center": t_year_to_datetime(t_center).isoformat(),
            "h_ice_res": float(beta[k]),
            "sigma_h_ice": float(np.hypot(intercept_se, fac_rep)),
            "sigma_fit": intercept_se, "fac_sigma_rep": fac_rep,
            "fac_window_median": float(np.median(part.fac)),
            "N_obs": int(len(part)), "N_blocks": nb,
            "fit_rmse_window_m": rmse, "fit_condition_number": condition,
        })
    res = pd.DataFrame(rows).sort_values("t_center_year").reset_index(drop=True)
    ref = res.window.eq("JJA 2019")
    h0 = float(res.loc[ref, "h_ice_res"].iloc[0])
    s0 = float(res.loc[ref, "sigma_h_ice"].iloc[0])
    res["dh_ice_m"] = res.h_ice_res - h0
    res["sigma_dh_ice_m"] = np.where(ref, 0.0, np.hypot(res.sigma_h_ice, s0))
    res["dM_Gt"] = res.dh_ice_m * GT_PER_M
    res["sigma_dM_Gt"] = res.sigma_dh_ice_m * GT_PER_M
    return res


def trend_and_seasonality(series, seed):
    """Copia de run_teis_hi_seasonal_timeseries.trend_and_seasonality (y = h_ice)."""
    from scipy.stats import theilslopes
    t = series.t_center_year.to_numpy(float)
    y = series.h_ice_res.to_numpy(float)
    sigma = series.sigma_h_ice.to_numpy(float)
    slope = float(theilslopes(y, t, alpha=0.95).slope)
    tref = float(np.mean(t))
    intercept = float(np.median(y - slope * (t - tref)))
    rng = np.random.default_rng(seed)
    draws = y[None, :] + rng.normal(size=(BOOTSTRAP_DRAWS, len(y))) * sigma[None, :]
    i, j = np.triu_indices(len(y), 1)
    slope_draws = np.median((draws[:, j] - draws[:, i]) / (t[j] - t[i])[None, :], axis=1)
    intercept_draws = np.median(draws - slope_draws[:, None] * (t - tref)[None, :], axis=1)
    slope_sigma = float(np.std(slope_draws, ddof=1))
    ci_low, ci_high = np.quantile(slope_draws, [0.025, 0.975])
    p_two = float(min(1.0, 2.0 * min(np.mean(slope_draws <= 0), np.mean(slope_draws >= 0))))
    is_jja = series.season.eq("JJA").to_numpy()
    residual = y - (intercept + slope * (t - tref))
    seasonal_difference = float(np.median(residual[is_jja]) - np.median(residual[~is_jja]))
    residual_draws = draws - (intercept_draws[:, None] + slope_draws[:, None] * (t - tref)[None, :])
    seasonal_draws = (np.median(residual_draws[:, is_jja], axis=1)
                      - np.median(residual_draws[:, ~is_jja], axis=1))
    seasonal_sigma = float(np.std(seasonal_draws, ddof=1))
    seasonal_ci = np.quantile(seasonal_draws, [0.025, 0.975])
    delta = np.diff(y)
    delta_sigma = np.sqrt(sigma[1:] ** 2 + sigma[:-1] ** 2)
    prev = series.season.iloc[:-1].to_numpy()
    cur = series.season.iloc[1:].to_numpy()
    jd = delta[(prev == "JJA") & (cur == "DJF")]
    dj = delta[(prev == "DJF") & (cur == "JJA")]
    sjd = delta_sigma[(prev == "JJA") & (cur == "DJF")]
    sdj = delta_sigma[(prev == "DJF") & (cur == "JJA")]
    sign_consistent = bool(len(jd) >= 2 and len(dj) >= 2
                           and np.all(np.sign(jd) == np.sign(jd[0]))
                           and np.all(np.sign(dj) == np.sign(dj[0]))
                           and np.sign(jd[0]) == -np.sign(dj[0]))
    exceeds = bool(len(jd) and len(dj) and np.all(np.abs(jd) > sjd) and np.all(np.abs(dj) > sdj))
    return {
        "method": "Theil-Sen; Monte Carlo parametrico condicional a sigma_h_ice",
        "n_windows": int(len(series)), "time_span_years": float(np.ptp(t)),
        "slope_m_per_year": slope, "slope_sigma_m_per_year": slope_sigma,
        "slope_ci95_low": float(ci_low), "slope_ci95_high": float(ci_high),
        "slope_p_two_sided": p_two, "intercept_at_mean_time": intercept, "t_reference": tref,
        "interannual_thinning_supported": bool(ci_high < 0),
        "JJA_minus_DJF_detrended_m": seasonal_difference,
        "JJA_minus_DJF_sigma_m": seasonal_sigma,
        "JJA_minus_DJF_ci95_low": float(seasonal_ci[0]),
        "JJA_minus_DJF_ci95_high": float(seasonal_ci[1]),
        "consecutive_signs_repeat": sign_consistent,
        "all_consecutive_differences_exceed_1sigma": exceeds,
        "repeatable_seasonal_variability": bool(
            sign_consistent and exceeds and (seasonal_ci[0] > 0 or seasonal_ci[1] < 0)),
        "jja_to_djf_deltas_m": jd.tolist(), "djf_to_jja_deltas_m": dj.tolist(),
    }


def main():
    import argparse

    global TARGET_WINDOWS, WINDOW_ORDER, COM_FIRN, ACEITA_JOIN_NAO_VERIFICADO
    global CELL_NAME, CX, CY, OUT, RAW_PARQUET, PREV_SERIE
    ap = argparse.ArgumentParser(description="Serie sazonal em gelo aterrado.")
    ap.add_argument("--cell", default=CELL_NAME, help="nome da celula no parquet")
    ap.add_argument("--cx", type=float, default=CX, help="centro x (EPSG:3031)")
    ap.add_argument("--cy", type=float, default=CY, help="centro y (EPSG:3031)")
    ap.add_argument("--input", default=None, help="parquet de pontos por celula")
    ap.add_argument("--outdir", default=None, help="pasta de saida do produto")
    ap.add_argument("--prev", default=None,
                    help="serie anterior p/ comparacao; 'none' desliga")
    ap.add_argument("--aceitar-join-nao-verificado", action="store_true",
                    help="admite granulos com QC_JOIN_UNVERIFIED. O pareamento "
                         "com o sidecar de QC deixa de ser garantido; os "
                         "granulos admitidos vao listados no relatorio.")
    ap.add_argument("--ultimo-jja", type=int, default=2025,
                    help="ultimo inverno da serie (padrao 2025)")
    ap.add_argument("--com-firn", action="store_true",
                    help="desconta o FAC (IMAU-FDM). So vai ate JJA 2023: o "
                         "recorte termina em 2023,98 e nao se extrapola.")
    args = ap.parse_args()
    CELL_NAME, CX, CY = args.cell, args.cx, args.cy
    if args.input:
        RAW_PARQUET = Path(args.input)
    if args.outdir:
        OUT = Path(args.outdir)
    if args.prev is not None:
        PREV_SERIE = None if args.prev.lower() == "none" else Path(args.prev)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"celula: {CELL_NAME} ({CX:.0f}, {CY:.0f}) | entrada: {RAW_PARQUET.name} "
          f"| saida: {OUT.name}")
    TARGET_WINDOWS = build_windows(args.ultimo_jja)
    WINDOW_ORDER = {w: i for i, w in enumerate(TARGET_WINDOWS)}
    COM_FIRN = bool(args.com_firn)
    ACEITA_JOIN_NAO_VERIFICADO = bool(args.aceitar_join_nao_verificado)
    if ACEITA_JOIN_NAO_VERIFICADO:
        print("ATENCAO: granulos com QC_JOIN_UNVERIFIED serao admitidos.")
    if COM_FIRN and args.ultimo_jja > 2023:
        raise SystemExit("--com-firn exige --ultimo-jja <= 2023 (limite do IMAU-FDM)")
    print("janelas:", len(TARGET_WINDOWS), "| com firn:", COM_FIRN)

    cfg = load_config()
    obs, audit = load_observations(cfg)
    print(json.dumps(audit, indent=1))
    print(obs.groupby("window").agg(n=("h_ice_res", "size"), blocos=("block", "nunique"),
                                    fac=("fac", "median")).reindex(TARGET_WINDOWS))
    ser = joint_window_fit(obs)
    met = trend_and_seasonality(ser, seed=9561)
    gia = GIAField.from_caron_table(ROOT / "data" / "gia" / "GIA_maps_Caron_et_al_2018")
    vlm, _ = gia.sample(np.array([ser.lon.iloc[0]]), np.array([ser.lat.iloc[0]]))
    # comparacao com a serie anterior (cadeia tiles, sem FAC) nas mesmas janelas
    prev_path = PREV_SERIE
    comp = None
    if prev_path is not None and Path(prev_path).exists():
        prev = pd.read_csv(prev_path)[["window", "dM_Gt", "sigma_dM_Gt"]]
        comp = ser[["window", "dM_Gt", "sigma_dM_Gt"]].merge(prev, on="window",
                                                              suffixes=("_teis_metodo", "_tiles_sem_fac"))
        comp["diferenca_Gt"] = comp.dM_Gt_teis_metodo - comp.dM_Gt_tiles_sem_fac
    ser.to_csv(OUT / "serie_metodo_TEIS.csv", index=False, encoding="utf-8")
    report = {
        "celula": {"x": CX, "y": CY, "lon": float(ser.lon.iloc[0]), "lat": float(ser.lat.iloc[0])},
        "com_firn": COM_FIRN,
        "join_nao_verificado_aceito": ACEITA_JOIN_NAO_VERIFICADO,
        "grandeza": ("h - FAC (espessura gelo-equivalente)" if COM_FIRN
                     else "h (elevacao de superficie, sem desconto de firn)"),
        "grandeza_simbolo": "h - FAC" if COM_FIRN else "h",
        "janelas": list(TARGET_WINDOWS),
        "Gt_por_metro": GT_PER_M, "audit_qc": audit, "tendencia": met,
        "dM_dt_Gt_ano": met["slope_m_per_year"] * GT_PER_M,
        "sigma_dM_dt_Gt_ano": met["slope_sigma_m_per_year"] * GT_PER_M,
        "gia_mm_ano": float(np.ravel(vlm)[0]) * 1000.0,
        "fac_fonte": str(THICK.FDM_FAC),
        "comparacao_cadeia_tiles_sem_FAC": None if comp is None else comp.to_dict(orient="records"),
    }
    (OUT / "relatorio_metodo_TEIS.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(ser[["window", "t_center_year", "h_ice_res", "sigma_h_ice", "sigma_fit",
               "fac_sigma_rep", "fac_window_median", "dh_ice_m", "dM_Gt", "sigma_dM_Gt",
               "N_obs", "N_blocks", "fit_rmse_window_m"]].to_string(float_format=lambda v: "%.4f" % v))
    print(json.dumps({k: v for k, v in met.items()}, indent=1))
    print("dM/dt Gt/ano:", report["dM_dt_Gt_ano"], "+-", report["sigma_dM_dt_Gt_ano"])
    if comp is not None:
        print(comp.to_string(float_format=lambda v: "%+.4f" % v))


if __name__ == "__main__":
    main()
