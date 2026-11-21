"""
thwaites.io.download
====================
Download + extração de grânulos ICESat-2 ATL06 seguindo o padrão obrigatório:
um grânulo por vez, sem preservar o .h5 bruto em disco.

Fluxo por grânulo:
    baixar (pasta temporária) -> extrair só as variáveis usadas (extract) ->
    salvar Parquet leve rastreável ao grânulo (store) -> DELETAR o .h5
    (try/finally, mesmo se a extração falhar no meio).

Nunca baixa o lote inteiro antes de processar. Autenticação uma única vez.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

from thwaites.config import Config
from thwaites.io.extract import extract_atl06
from thwaites.io.store import (
    POINT_SCHEMA_VERSION, is_current_points_parquet, save_points_parquet,
)
from thwaites.logging import get_logger

logger = get_logger()

MAX_RETRIES = 3
RETRY_WAIT_S = 5


def granule_filename(granule) -> str | None:
    """Nome do arquivo do grânulo (ATL06_YYYYMMDD..._...h5), ou None."""
    try:
        return granule.data_links()[0].split("/")[-1]
    except (IndexError, KeyError, AttributeError):
        return None


def granule_month(fname: str) -> int | None:
    """Mês do padrão ATL06_YYYYMMDDHHMMSS_...; None (com aviso) se fora do padrão."""
    try:
        return int(fname.split("_")[1][4:6])
    except (IndexError, ValueError):
        logger.warning(f"nome fora do padrão, grânulo ignorado: {fname}")
        return None


def authenticate():
    """
    Autentica no NASA Earthdata uma vez, por credencial já persistida.

    O contrato deste projeto exige explicitamente ``strategy="interactive"``.
    O earthaccess ainda pode reutilizar credenciais persistidas, mas não há
    fallback implícito para credenciais de ambiente ou outra estratégia.
    """
    import earthaccess
    return earthaccess.login(strategy="interactive")


def search_granules(cfg: Config):
    """Busca metadados (leve — não baixa dado pesado)."""
    import earthaccess
    spatial_scope = cfg.roi or cfg.area
    return earthaccess.search_data(
        short_name=cfg.product.short_name,
        version=cfg.product.version,
        # O produto basal é definido na ROI ASE. Buscar a bbox continental
        # ampla só transfere órbitas que serão descartadas no consolidate.
        bounding_box=spatial_scope.bounding_box,
        temporal=cfg.temporal.temporal_range,
        count=-1,
    )


def filter_season(granules, cfg: Config) -> list[tuple[str, object]]:
    """Mantém só os grânulos cujos meses estão em cfg.season.months."""
    months = set(cfg.season.months)
    out = []
    for g in granules:
        fname = granule_filename(g)
        if fname is None:
            logger.warning("grânulo sem data_link utilizável, ignorado.")
            continue
        if granule_month(fname) in months:
            out.append((fname, g))
    return out


def run_download(cfg: Config, authenticate_fn=authenticate) -> dict:
    """
    Executa o download+extração completo. Retorna um resumo (dict).

    `authenticate_fn` é injetável para permitir teste sem rede.
    """
    import earthaccess

    cfg.paths.ensure()
    raw_temp = cfg.paths.raw_temp
    processed = cfg.paths.processed

    leftover = list(raw_temp.glob("*.h5"))
    if leftover:
        logger.warning(f"{len(leftover)} .h5 residuais em {raw_temp} — serão limpos.")
        for h5 in leftover:
            try:
                h5.unlink()
            except OSError:
                pass

    logger.info("Autenticando no NASA Earthdata...")
    authenticate_fn()

    logger.info("Buscando metadados dos grânulos...")
    all_results = search_granules(cfg)
    logger.info(f"Grânulos na região/período: {len(all_results)}")

    season = filter_season(all_results, cfg)
    logger.info(f"Grânulos na estação '{cfg.season.name}' "
                f"(meses {sorted(cfg.season.months)}): {len(season)}")
    if not season:
        raise SystemExit("Nenhum grânulo na estação selecionada — verifique bbox/período.")

    with open(cfg.paths.logs / f"granules_{cfg.season.name}.json", "w") as fp:
        json.dump([fn for fn, _ in season], fp, indent=2)

    failed: list[str] = []
    n_done = n_skip = n_empty = n_stale = 0
    total_pts = 0
    qc_n_in = qc_n_out = 0
    qc_by_reason: dict[str, int] = {}

    for i, (fname, granule) in enumerate(season, start=1):
        out_parquet = processed / f"{Path(fname).stem}.parquet"
        if out_parquet.exists() and is_current_points_parquet(out_parquet):
            n_skip += 1
            continue
        if out_parquet.exists():
            n_stale += 1
            logger.warning(
                f"   {out_parquet.name} usa schema legado/corrompido; será "
                "substituído somente após nova extração válida.")

        logger.info(f"[{i}/{len(season)}] {fname}")
        h5_path = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                downloaded = earthaccess.download([granule], str(raw_temp))
                if not downloaded:
                    raise RuntimeError("earthaccess.download retornou vazio")
                h5_path = Path(downloaded[0])
                if not h5_path.exists() or h5_path.stat().st_size < 100_000:
                    raise RuntimeError("arquivo baixado ausente ou truncado")

                df = extract_atl06(h5_path, cfg)
                qc_n_in += int(df.attrs.get("qc_n_in", len(df)))
                qc_n_out += int(df.attrs.get("qc_n_out", len(df)))
                for reason, count in df.attrs.get("qc_by_reason", {}).items():
                    qc_by_reason[reason] = qc_by_reason.get(reason, 0) + int(count)
                if len(df) == 0:
                    logger.info("   sem pontos válidos após QC — nenhum Parquet gerado.")
                    n_empty += 1
                else:
                    save_points_parquet(df, out_parquet)
                    total_pts += len(df)
                    n_done += 1
                    logger.info(f"   OK {len(df):,} pontos -> {out_parquet.name}")
                break
            except Exception as e:
                logger.warning(f"   tentativa {attempt}/{MAX_RETRIES} falhou: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_WAIT_S)
                else:
                    failed.append(fname)
                    logger.error(traceback.format_exc())
            finally:
                # DELETA o .h5 bruto SEMPRE — mesmo se a extração falhou.
                if h5_path is not None and h5_path.exists():
                    try:
                        h5_path.unlink()
                    except OSError as del_err:
                        logger.warning(f"   não removeu {h5_path.name}: {del_err}")
                h5_path = None

    if failed:
        (cfg.paths.logs / "failed_downloads.txt").write_text("\n".join(failed), encoding="utf-8")

    summary = {
        "season_granules": len(season),
        "extracted": n_done,
        "skipped": n_skip,
        "stale_reprocessed": n_stale,
        "empty": n_empty,
        "failed": len(failed),
        "total_points": total_pts,
        "point_schema_version": POINT_SCHEMA_VERSION,
        "qc_n_in": qc_n_in,
        "qc_n_out": qc_n_out,
        "qc_by_reason": qc_by_reason,
    }
    summary_path = cfg.paths.tables / f"download_extract_{cfg.season.name}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"RESUMO: {summary}")
    logger.info(f"Relatório -> {summary_path}")
    return summary
