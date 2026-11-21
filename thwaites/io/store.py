"""
thwaites.io.store
=================
Leitura e escrita dos dados de pontos em Parquet (colunar, tipado, leve).

Parquet oferece menor uso de disco, preservação de tipos, leitura seletiva de
colunas e integração nativa com pandas/dask.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Schema v4 dos arquivos de pontos. Identificadores orbitais, flags e o estado
# completo das correções geofísicas ficam no
# MESMO registro da altura; não há mais pareamento posicional com outro Parquet.
POINT_SCHEMA_VERSION = 4
POINT_COLUMNS: dict[str, str] = {
    "granule_id": "string",
    "rgt": "int16",
    "cycle": "int8",
    "region": "int8",
    "lon": "float64",
    "lat": "float64",
    "h_elv": "float32",
    "s_elv": "float32",
    "delta_time": "float64",
    "t_year": "float64",
    "beam": "int8",
    "segment_id": "int32",
    "x_atc": "float64",
    "y_atc": "float64",
    "atl06_quality_summary": "int8",
    "h_robust_sprd": "float32",
    "n_fit_photons": "int32",
    "dh_fit_dx": "float32",
    "h_expected_rms": "float32",
    "snr_significance": "float32",
    "msw_flag": "int8",
    "cloud_flg_asr": "int8",
    "cloud_flg_atm": "int8",
    "bsnow_conf": "int8",
    "solar_elevation": "float32",
    "dem_h": "float32",
    "sigma_geo_h": "float32",
    "sigma_geo_at": "float32",
    "sigma_geo_xt": "float32",
    "tide_earth": "float32",
    "tide_load": "float32",
    "tide_ocean": "float32",
    "tide_pole": "float32",
    "tide_equilibrium": "float32",
    "dac": "float32",
    "neutat_delay_total": "float32",
    "geoid": "float32",
}

_PA_TYPES = {
    "string": pa.string(), "float64": pa.float64(), "float32": pa.float32(),
    "int8": pa.int8(), "int16": pa.int16(), "int32": pa.int32(),
}


def point_schema() -> "pa.Schema":
    """Schema Arrow alvo dos arquivos de pontos (ordem de POINT_COLUMNS)."""
    return pa.schema(
        [(c, _PA_TYPES[t]) for c, t in POINT_COLUMNS.items()],
        metadata={b"thwaites_point_schema_version": str(POINT_SCHEMA_VERSION).encode()},
    )


def _cast_table(table: "pa.Table") -> "pa.Table":
    """Seleciona as colunas do schema (ignora extras) e força os tipos."""
    missing = [c for c in POINT_COLUMNS if c not in table.column_names]
    if missing:
        raise ValueError(f"faltam colunas {missing}")
    return table.select(list(POINT_COLUMNS)).cast(point_schema())


def _validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in POINT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame sem colunas obrigatórias: {missing}")


def save_points_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Grava pontos em Parquet de forma transacional e com versão de schema."""
    _validate_schema(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.unlink(missing_ok=True)
    table = _cast_table(pa.Table.from_pandas(df, preserve_index=False))
    pq.write_table(table, partial, compression="zstd", use_dictionary=True)
    partial.replace(path)
    return path


def is_current_points_parquet(path: str | Path) -> bool:
    """True somente para Parquet íntegro no contrato de pontos atual."""
    try:
        pf = pq.ParquetFile(path)
        md = pf.schema_arrow.metadata or {}
        version = int(md.get(b"thwaites_point_schema_version", b"0"))
        return version == POINT_SCHEMA_VERSION and set(POINT_COLUMNS).issubset(
            pf.schema_arrow.names)
    except Exception:
        return False


def read_points_parquet(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Lê pontos de um Parquet (opcionalmente só algumas colunas)."""
    return pd.read_parquet(path, columns=columns, engine="pyarrow")


def _filter_roi(table: "pa.Table", roi) -> "pa.Table":
    """Mantém só os pontos com lon/lat dentro de roi=(lon_min,lat_min,lon_max,lat_max)."""
    import pyarrow.compute as pc
    lon_min, lat_min, lon_max, lat_max = roi
    lon, lat = table["lon"], table["lat"]
    m = pc.and_(
        pc.and_(pc.greater_equal(lon, lon_min), pc.less_equal(lon, lon_max)),
        pc.and_(pc.greater_equal(lat, lat_min), pc.less_equal(lat, lat_max)),
    )
    return table.filter(m)


def _write_exclusion_log(path: Path, roi, n_scanned, excluded, kept,
                         total_before, total_after, n_bad):
    from datetime import datetime
    lon_min, lat_min, lon_max, lat_max = roi
    lines = [
        "# Log de recorte por ROI (arquivos desconsiderados)",
        "",
        f"- Gerado em: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- ROI (lon): {lon_min}° a {lon_max}°",
        f"- ROI (lat): {lat_min}° a {lat_max}°",
        f"- Arquivos escaneados: {n_scanned}",
        f"- **Arquivos desconsiderados (0 pontos na ROI): {len(excluded)}**",
        f"- Arquivos mantidos: {kept}",
        f"- Arquivos ignorados por erro/schema: {n_bad}",
        f"- Pontos antes do recorte: {total_before:,}",
        f"- Pontos após o recorte: {total_after:,}"
        + (f"  ({100*total_after/total_before:.2f}% mantido)" if total_before else ""),
        "",
        "> Nenhum arquivo foi apagado — os desconsiderados apenas não entram no",
        "> processamento por não terem pontos dentro da ROI.",
        "",
        "## Arquivos desconsiderados",
        "",
    ]
    lines += [f"- {name}" for name in excluded] or ["(nenhum)"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def consolidate_parquets(
    processed_dir: str | Path,
    out_path: str | Path,
    pattern: str = "*.parquet",
    progress_every: int = 200,
    roi=None,
    exclusion_log: str | Path | None = None,
    qc_dir: str | Path | None = None,
    cfg=None,
    qc_stats: dict | None = None,
) -> tuple[Path, int]:
    """
    Consolida todos os Parquets por grânulo num único arquivo, em STREAMING.

    Lê um arquivo por vez e escreve imediatamente um row group no destino via
    `pyarrow.ParquetWriter` — a memória fica CONSTANTE (nunca segura tudo na
    RAM nem faz um `concat` gigante). É o que torna viável consolidar volumes
    grandes (bilhões de linhas / dezenas de GB). Ignora, com aviso, arquivos
    ilegíveis ou fora do schema.

    Se `roi=(lon_min, lat_min, lon_max, lat_max)` for dado, filtra os pontos por
    essa bbox (recorte de processamento). Os arquivos de origem NÃO são apagados;
    os que ficam totalmente fora da ROI são registrados em `exclusion_log` (.md).

    Se `qc_dir` e `cfg` forem dados (e `cfg.atl06_qc.enabled`), aplica o filtro
    de qualidade pelos flags nativos do ATL06 ANTES do recorte de ROI. Este é o
    ÚNICO ponto do pipeline onde isso pode ser feito: os flags são pareados
    POSICIONALMENTE com cada Parquet de grânulo, e a identidade do grânulo se
    perde na consolidação. O pareamento é verificado por contagem de linhas —
    se divergir, a execução científica falha fechada por padrão. Uma saída em
    construção usa sufixo `.partial` e só substitui a saída anterior depois de
    todos os grânulos passarem, evitando deixar um produto truncado.

    `qc_stats` (dict opcional) recebe as contagens agregadas por critério.

    Retorna (caminho_saida, n_pontos_totais_após_recorte).
    """
    processed_dir = Path(processed_dir)
    files = sorted(processed_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Nenhum Parquet em {processed_dir} ({pattern}).")

    use_qc = bool(qc_dir is not None and cfg is not None
                  and getattr(cfg, "atl06_qc", None) is not None
                  and cfg.atl06_qc.enabled)
    if use_qc:
        from thwaites.qc.atl06_flags import quality_mask
        qc_dir = Path(qc_dir)
        qcs = qc_stats if qc_stats is not None else {}
        qcs.setdefault("n_in", 0)
        qcs.setdefault("n_out", 0)
        qcs.setdefault("granules_with_flags", 0)
        qcs.setdefault("granules_missing_flags", [])
        qcs.setdefault("granules_row_mismatch", [])
        qcs.setdefault("by_reason", {})
        print(f"[qc] filtro de qualidade ATL06 ATIVO (flags em {qc_dir})")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = out_path.with_name(out_path.name + ".partial")
    partial_path.unlink(missing_ok=True)

    writer = None
    total = 0            # pontos escritos (após recorte)
    total_before = 0     # pontos lidos (antes do recorte)
    n_bad = 0
    kept = 0
    excluded: list[str] = []
    completed = False

    try:
        for i, fp in enumerate(files, start=1):
            try:
                table = _cast_table(pq.read_table(fp))
            except Exception as e:              # corrompido / schema inválido
                if use_qc and cfg.atl06_qc.require_complete_flags:
                    raise RuntimeError(
                        f"grânulo ilegível/schema inválido em modo fail-closed: "
                        f"{fp.name}: {e}") from e
                print(f"[aviso] ignorando {fp.name}: {e}")
                n_bad += 1
                continue

            # Grânulo inteiramente fora da ROI não contribui para o produto e
            # não precisa ser rebaixado só para obter flags. A filtragem real é
            # repetida depois do QC para preservar a ordem posicional.
            if roi is not None and _filter_roi(table, roi).num_rows == 0:
                total_before += table.num_rows
                excluded.append(fp.name)
                continue

            if use_qc:
                qp = qc_dir / fp.name
                if not qp.exists():
                    qcs["granules_missing_flags"].append(fp.name)
                    if cfg.atl06_qc.require_complete_flags:
                        raise FileNotFoundError(
                            f"flags ATL06 ausentes para grânulo relevante "
                            f"{fp.name}. Rode pipelines/fetch_qc_flags.py; a "
                            "consolidação foi abortada sem substituir a saída.")
                else:
                    qc = pq.read_table(qp).to_pandas()
                    if len(qc) != table.num_rows:
                        qcs["granules_row_mismatch"].append(
                            {"granule": fp.name, "n_qc": int(len(qc)),
                             "n_points": int(table.num_rows)})
                        if cfg.atl06_qc.require_complete_flags:
                            raise ValueError(
                                f"pareamento QC divergente para {fp.name}: "
                                f"{len(qc)} flags vs {table.num_rows} pontos. "
                                "Consolidação abortada.")
                    else:
                        h_li = table["h_elv"].to_numpy(zero_copy_only=False)
                        keep_q, reasons = quality_mask(qc, cfg, h_li=h_li)
                        qcs["n_in"] += int(table.num_rows)
                        qcs["n_out"] += int(keep_q.sum())
                        qcs["granules_with_flags"] += 1
                        for k, v in reasons.items():
                            qcs["by_reason"][k] = qcs["by_reason"].get(k, 0) + v
                        table = table.filter(pa.array(keep_q))
                        if table.num_rows == 0:
                            excluded.append(fp.name)
                            continue

            if roi is not None:
                total_before += table.num_rows
                table = _filter_roi(table, roi)
                if table.num_rows == 0:         # nada na ROI -> desconsiderado
                    excluded.append(fp.name)
                    continue

            if writer is None:
                writer = pq.ParquetWriter(partial_path, point_schema(),
                                          compression="snappy")
            writer.write_table(table)
            total += table.num_rows
            kept += 1
            if i % progress_every == 0:
                print(f"  {i}/{len(files)} arquivos | mantidos {kept} | "
                      f"desconsiderados {len(excluded)} | {total:,} pontos…", flush=True)
        completed = True
    finally:
        if writer is not None:
            writer.close()
        if not completed:
            partial_path.unlink(missing_ok=True)

    if writer is None:
        raise ValueError("Nenhum Parquet válido (ou nenhum ponto dentro da ROI).")
    if n_bad:
        print(f"[aviso] {n_bad} arquivos ignorados por erro/schema.")

    if roi is not None and exclusion_log is not None:
        _write_exclusion_log(Path(exclusion_log), roi, len(files), excluded, kept,
                             total_before, total, n_bad)
    # `replace` é atômico no mesmo volume: uma execução incompleta nunca deixa
    # um Parquet aparentemente válido no caminho oficial.
    partial_path.replace(out_path)
    return out_path, total
