# -*- coding: utf-8 -*-
"""Serie de massa por TEMPORADA (corrige o binning de DJF).

Correcao 2 do diagnostico sazonal.

run_serie_massa.py atribui o ano com ``ano = np.floor(t_year)``. Em JJA isso e
inofensivo (jun-jul-ago caem no mesmo ano civil), mas em DJF parte a temporada
em duas caixas: dezembro de Y vai para Y e janeiro-fevereiro de Y+1 vao para
Y+1. Cada caixa anual passa a misturar observacoes separadas por ~10 meses, e a
proporcao de dezembro medida varia de 24,6% a 55,5% entre anos, deslocando a
epoca efetiva da caixa em ate 0,29 ano.

Aqui a temporada e a unidade:
    JJA Y          = jun/jul/ago do ano Y
    DJF Y/Y+1      = dez de Y + jan/fev de Y+1   (rotulada pelo ano final)
A conversao usa a convencao de ano juliano do ATLAS (t_year_to_datetime), nao
floor(t_year) - a diferenca aparece exatamente na fronteira dez/jan.

Alem da massa, grava a EPOCA MEDIA de cada temporada, para que a taxa seja
calculada sobre o tempo real e nao sobre rotulos supostamente equidistantes.

Saida:
    data/<estacao>/dhdt/serie_temporada.parquet
    outputs/<estacao>/tables/serie_massa_temporada.json

Uso: python pipelines/products/run_serie_massa_temporada.py --profile djf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from thwaites import load_config                                    # noqa: E402
from thwaites.atl06_granule_id import (                             # noqa: E402
    ATLAS_SDP_EPOCH, JULIAN_YEAR_SECONDS)
from thwaites.grid.tiles import load_manifest, assign_xy            # noqa: E402
from thwaites.logging import setup_logging                          # noqa: E402

RHO_ICE = 917.0
RHO_W = 1000.0
MIN_PTS_NODE_WINDOW = 12   # identico a run_serie_massa.py


def months_and_years(t_year: np.ndarray):
    """t_year (ano juliano ATLAS) -> (mes, ano civil), vetorizado.

    Mesma convencao de thwaites.atl06_granule_id.t_year_to_datetime.
    """
    epoch = np.datetime64(ATLAS_SDP_EPOCH.replace(tzinfo=None), "s")
    seconds = (np.asarray(t_year, dtype=float) - 2018.0) * JULIAN_YEAR_SECONDS
    stamps = epoch + seconds.astype("timedelta64[s]")
    meses = stamps.astype("datetime64[M]")
    anos = stamps.astype("datetime64[Y]")
    mes = (meses - anos).astype(int) + 1
    ano = anos.astype(int) + 1970
    return mes, ano


def season_key(t_year: np.ndarray, season: str):
    """Rotulo da temporada e ano de referencia (DJF pelo ano FINAL)."""
    mes, ano = months_and_years(t_year)
    if season == "JJA":
        dentro = np.isin(mes, [6, 7, 8])
        chave = np.where(dentro, ano, -1)
    else:
        dezembro = mes == 12
        janfev = np.isin(mes, [1, 2])
        dentro = dezembro | janfev
        chave = np.where(dezembro, ano + 1, np.where(janfev, ano, -1))
    return chave.astype(int), dentro


def por_no_e_temporada(df, cfg, x_min, x_max, y_min, y_max, season):
    from scipy.spatial import cKDTree

    d = cfg.dhdt
    df = assign_xy(df, cfg)
    col = "h_res" if "h_res" in df.columns else "h_corr"
    ok = ~(df["x"].isna() | df["y"].isna() | df[col].isna() | df["t_year"].isna())
    x = df["x"].to_numpy()[ok]
    y = df["y"].to_numpy()[ok]
    h = df[col].to_numpy()[ok].astype(float)
    t = df["t_year"].to_numpy()[ok].astype(float)
    if len(h) < MIN_PTS_NODE_WINDOW:
        return pd.DataFrame()
    chave, dentro = season_key(t, season)
    x, y, h, t, chave = x[dentro], y[dentro], h[dentro], t[dentro], chave[dentro]
    if len(h) < MIN_PTS_NODE_WINDOW:
        return pd.DataFrame()

    passo = d.node_spacing_m
    gx = np.arange(x_min + passo / 2, x_max, passo)
    gy = np.arange(y_min + passo / 2, y_max, passo)
    if not len(gx) or not len(gy):
        return pd.DataFrame()
    GX, GY = np.meshgrid(gx, gy)
    arv = cKDTree(np.c_[x, y])

    linhas = []
    for nx, ny in zip(GX.ravel(), GY.ravel()):
        idx = np.asarray(arv.query_ball_point([nx, ny], r=d.search_radius_m))
        if idx.size < MIN_PTS_NODE_WINDOW:
            continue
        hi, ki, ti = h[idx], chave[idx], t[idx]
        for k in np.unique(ki):
            m = ki == k
            if m.sum() < MIN_PTS_NODE_WINDOW:
                continue
            linhas.append((float(nx), float(ny), int(k), float(np.median(hi[m])),
                           float(np.mean(ti[m])), int(m.sum())))
    return pd.DataFrame(linhas, columns=["x", "y", "temporada", "h", "epoca", "n"])


def main():
    ap = argparse.ArgumentParser(description="Serie de massa por temporada.")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--max-tiles", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.profile)
    est = cfg.season.name
    season = "JJA" if est.lower() == "jja" else "DJF"
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level,
                        run_name=f"serie_massa_temporada_{est}")

    ent = load_manifest(cfg)
    if args.max_tiles:
        ent = ent[:args.max_tiles]
    log.info(f"[{est}] elevacao por no e temporada, {len(ent)} tiles")

    COLS = ["x", "y", "lon", "lat", "t_year", "h_res", "h_corr"]
    partes = []
    for i, e in enumerate(ent, 1):
        p = cfg.paths.tiles_dir / e["file"]
        disp = pq.ParquetFile(p).schema_arrow.names
        df = pd.read_parquet(p, columns=[c for c in COLS if c in disp],
                             engine="pyarrow")
        r = por_no_e_temporada(df, cfg, e["x_min"], e["x_max"], e["y_min"],
                               e["y_max"], season)
        del df
        if len(r):
            partes.append(r)
        if i % 10 == 0 or i == len(ent):
            log.info(f"  [{i}/{len(ent)}]")

    S = pd.concat(partes, ignore_index=True)
    qc = pd.read_parquet(cfg.paths.dhdt_dir / "dhdt_nodes_qc.parquet",
                         columns=["x", "y"])
    chave = set(zip(np.rint(qc.x).astype(np.int64), np.rint(qc.y).astype(np.int64)))
    S = S[np.fromiter(((a, b) in chave for a, b in
                       zip(np.rint(S.x).astype(np.int64),
                           np.rint(S.y).astype(np.int64))), bool, len(S))]

    # Temporadas parciais nas pontas (DJF sem dezembro no inicio, ou sem
    # jan/fev no fim) nao entram: elas e que produziam vaos diferentes entre as
    # estacoes. O criterio e a presenca dos dois sub-meses na temporada.
    cobertura = []
    for k, g in S.groupby("temporada"):
        cobertura.append({"temporada": int(k), "n_nos": int(g.x.nunique()),
                          "epoca_media": float(g.epoca.mean())})
    cob = pd.DataFrame(cobertura).sort_values("temporada")
    log.info("temporadas encontradas:\n" + cob.to_string(index=False))

    S = S.sort_values(["x", "y", "temporada"])
    base = S.groupby(["x", "y"]).h.transform("first")
    S["anom"] = S.h - base
    S.to_parquet(cfg.paths.dhdt_dir / "serie_temporada.parquet", index=False)

    cont = S.groupby(["x", "y"]).temporada.nunique()
    n_temp = S.temporada.nunique()
    completos = set(cont[cont == n_temp].index)
    C = S[[(a, b) in completos for a, b in zip(S.x, S.y)]]
    log.info(f"{len(completos):,} nos com todas as {n_temp} temporadas")

    area = cfg.dhdt.node_spacing_m ** 2
    serie = []
    for k, g in C.groupby("temporada"):
        dh = float(g.anom.mean())
        serie.append({"temporada": int(k),
                      "rotulo": (f"JJA {k}" if season == "JJA"
                                 else f"DJF {k - 1}/{k}"),
                      "epoca_media": float(g.epoca.mean()),
                      "dh_medio_m": dh, "dh_mediano_m": float(g.anom.median()),
                      "mwe": dh * RHO_ICE / RHO_W,
                      "massa_acumulada_Gt": dh * area * len(completos) * RHO_ICE / 1e12,
                      "n_nos": int(len(g))})
    ep = np.array([s["epoca_media"] for s in serie])
    ma = np.array([s["massa_acumulada_Gt"] for s in serie])
    taxa = float(np.polyfit(ep, ma, 1)[0]) if len(ep) >= 2 else float("nan")
    rel = {
        "estacao": est, "season": season,
        "correcao": "temporada como unidade; DJF = dez(Y) + jan/fev(Y+1), "
                    "rotulada pelo ano final; conversao por ano juliano ATLAS",
        "n_nos_completos": len(completos),
        "area_km2": len(completos) * area / 1e6,
        "densidade_gelo": RHO_ICE,
        "span_epocas_anos": float(ep.max() - ep.min()) if len(ep) else None,
        "taxa_Gt_por_ano_ajuste_linear": taxa,
        "serie": serie,
        "cobertura_por_temporada": cob.to_dict(orient="records"),
        "nota": ("massa INFERIDA de altura x densidade suposta; anomalia "
                 "relativa a primeira temporada de cada no; agregacao pela MEDIA"),
    }
    rp = cfg.paths.tables / "serie_massa_temporada.json"
    rp.write_text(json.dumps(rel, indent=2, ensure_ascii=False), encoding="utf-8")
    for s in serie:
        log.info(f"{s['rotulo']:>14} epoca {s['epoca_media']:.3f} "
                 f"dh {s['dh_medio_m']:+.3f} m  massa {s['massa_acumulada_Gt']:+.1f} Gt")
    log.info(f"taxa por ajuste linear sobre a epoca real: {taxa:+.1f} Gt/ano")
    log.info(f"-> {rp}")


if __name__ == "__main__":
    main()
