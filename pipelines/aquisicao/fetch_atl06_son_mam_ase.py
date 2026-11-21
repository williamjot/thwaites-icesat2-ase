"""Download do ATL06 r007 de SON e MAM (2019–2025) sobre a ROI da ASE.

Pedido do pesquisador em 2026-11-14: completar as estações SON e MAM fora da Thwaites, para que a
série sazonal do derretimento basal cubra PIG, Thwaites, Crosson, Dotson e Cosgrove.

Por que um diretório novo (``data/ase_v2``) em vez de completar ``data/thwaites_v2``: os Parquet de
``thwaites_v2`` foram recortados a um retângulo de ±150 km em torno da Thwaites (lon −113,1 a
−111,5 nos arquivos conferidos). O mesmo grânulo precisa ser relido para a ROI maior; pular pelo
nome do arquivo deixaria a PIG e a Cosgrove sem dado sem nenhum aviso.

Recorte: a bbox ``roi`` de ``config/default.yaml`` (lon −115 a −95, lat −77,5 a −73), a mesma do
manifesto JJA/DJF ``data/processed/legacy_ase_manifest.csv``.

Padrão obrigatório do projeto: UM grânulo por vez — baixar para ``data/raw_temp`` → extrair com
``thwaites.io.extract.extract_atl06`` → recortar → gravar Parquet → apagar o ``.h5`` em
``finally``. Retomável: grânulo já gravado ou registrado como vazio é pulado.

- QC não aplicado na extração (``atl06_qc.enabled = False``): os flags ficam gravados para o
  ``quality_mask`` ser aplicado depois.
- Autenticação por ``~/.netrc`` (``strategy="netrc"``). O modo ``"interactive"``
  exige digitar a senha e não roda desassistido.

Saída: ``data/ase_v2/{son,mam}/<granulo>.parquet`` e ``outputs/download_ase_son_mam/``.
Uso: ``python pipelines/fetch_atl06_son_mam_ase.py [--dry-run] [--limit N]``.
"""

from __future__ import annotations

import argparse
import calendar
import copy
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from thwaites import load_config  # noqa: E402
from thwaites.io.extract import extract_atl06  # noqa: E402

SHORT_NAME, VERSION = "ATL06", "007"
# JJA e DJF entraram em 2026-11-14: as camadas antigas (data/jja, data/djf) têm só 10 colunas,
# flags num sidecar casado por posição de linha e, no DJF, sidecar para só 2.900 de 4.999
# grânulos. Rebaixar põe as quatro estações no mesmo contrato e no mesmo código.
SEASONS = {"son": (9, 10, 11), "mam": (3, 4, 5), "jja": (6, 7, 8), "djf": (12, 1, 2)}
MAX_RETRIES, RETRY_WAIT_S = 3, 10
STORE = ROOT / "data" / "ase_v2"
RAW = ROOT / "data" / "raw_temp"
OUT = ROOT / "outputs" / "download_ase_son_mam"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=0, help="só os N primeiros (teste)")
    ap.add_argument("--seasons", nargs="+", default=list(SEASONS), choices=list(SEASONS))
    ap.add_argument("--dry-run", action="store_true", help="só busca e conta, não baixa")
    args = ap.parse_args()
    import earthaccess
    from pyproj import Transformer

    cfg = copy.deepcopy(load_config())
    cfg.atl06_qc.enabled = False
    r = cfg.roi
    bbox = (float(r.lon_min), float(r.lat_min), float(r.lon_max), float(r.lat_max))
    years = range(int(cfg.temporal.year_start), int(cfg.temporal.year_end) + 1)
    to_xy = Transformer.from_crs(4326, 3031, always_xy=True)

    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    for h5 in RAW.glob("*.h5"):                       # resíduo de execução interrompida
        h5.unlink()
    earthaccess.login(strategy="netrc")

    todo = []
    for season in args.seasons:
        months = SEASONS[season]
        (STORE / season).mkdir(parents=True, exist_ok=True)
        for year in years:
            # DJF do ano Y = dez/(Y−1) + jan–fev/Y, para que cada verão fique inteiro;
            # com anos 2019–2025 isso cobre dez/2018 a fev/2025.
            spans = ([(year - 1, 12), (year, 1), (year, 2)] if season == "djf"
                     else [(year, m) for m in months])
            n = 0
            for yy, mm in spans:
                a = f"{yy}-{mm:02d}-01"
                b = f"{yy}-{mm:02d}-{calendar.monthrange(yy, mm)[1]}"
                res = earthaccess.search_data(short_name=SHORT_NAME, version=VERSION,
                                              bounding_box=bbox, temporal=(a, b), count=-1)
                for g in res:
                    name = g.data_links()[0].split("/")[-1]
                    if int(name.split("_")[1][4:6]) == mm:   # mês de aquisição no nome
                        todo.append((season, name, g))
                        n += 1
            print(f"{season} {year}: {n} grânulos", flush=True)
    seen = set()
    todo = [t for t in todo if not (t[1] in seen or seen.add(t[1]))]
    (OUT / f"granulos_buscados_{'_'.join(args.seasons)}.json").write_text(
        json.dumps([[s, n] for s, n, _ in todo], indent=1), encoding="utf-8")
    empty_log = OUT / "granulos_sem_pontos_na_roi.txt"
    empties = set(empty_log.read_text(encoding="utf-8").split()) if empty_log.is_file() else set()
    done = sum((STORE / s / f"{Path(n).stem}.parquet").is_file() or n in empties
               for s, n, _ in todo)
    print(f"total {len(todo)} grânulos; já feitos {done}; a baixar {len(todo) - done}",
          flush=True)
    if args.dry_run:
        return 0
    if args.limit:
        todo = todo[:args.limit]

    t0 = time.time()
    n_ok = n_skip = n_empty = n_pts = 0
    failed = []
    for k, (season, name, g) in enumerate(todo, 1):
        out = STORE / season / f"{Path(name).stem}.parquet"
        if out.is_file() or name in empties:
            n_skip += 1
            continue
        for attempt in range(1, MAX_RETRIES + 1):
            h5 = None
            try:
                got = earthaccess.download([g], str(RAW))
                if not got:
                    raise RuntimeError("earthaccess.download retornou vazio")
                h5 = Path(got[0])
                if not h5.exists() or h5.stat().st_size < 100_000:
                    raise RuntimeError("arquivo baixado ausente ou truncado")
                df = extract_atl06(h5, cfg)
                if len(df):
                    lon, lat = df["lon"].to_numpy(), df["lat"].to_numpy()
                    keep = ((lon >= bbox[0]) & (lon <= bbox[2])
                            & (lat >= bbox[1]) & (lat <= bbox[3]))
                    df = df[keep].copy()
                    df["x"], df["y"] = to_xy.transform(df["lon"].to_numpy(),
                                                       df["lat"].to_numpy())
                if len(df) == 0:
                    with open(empty_log, "a", encoding="utf-8") as fp:
                        fp.write(name + "\n")
                    n_empty += 1
                else:
                    tmp = out.with_suffix(".parquet.partial")
                    df.to_parquet(tmp, index=False, compression="zstd")
                    tmp.replace(out)
                    n_ok += 1
                    n_pts += len(df)
                break
            except Exception as e:  # noqa: BLE001
                print(f"   tentativa {attempt} falhou em {name}: {repr(e)[:120]}", flush=True)
                if attempt == MAX_RETRIES:
                    failed.append(name)
                    with open(OUT / "falhas.log", "a", encoding="utf-8") as fp:
                        fp.write(f"{name}\n{traceback.format_exc()}\n")
                else:
                    time.sleep(RETRY_WAIT_S)
            finally:
                # o .h5 bruto é apagado SEMPRE, mesmo se a extração falhar
                if h5 is not None and h5.exists():
                    h5.unlink()
                for stray in RAW.glob("*.h5"):
                    stray.unlink()
        if k % 20 == 0 or k == len(todo):
            print(f"[{k}/{len(todo)}] gravados {n_ok}, vazios {n_empty}, pulados {n_skip}, "
                  f"falhas {len(failed)} | {n_pts:,} pontos | {time.time() - t0:.0f}s", flush=True)
    summary = {"produto": f"{SHORT_NAME} r{VERSION}", "bbox_lonlat": bbox,
               "anos": [years.start, years.stop - 1], "estacoes": args.seasons,
               "granulos": len(todo), "gravados": n_ok, "vazios": n_empty, "pulados": n_skip,
               "falhas": failed, "pontos_gravados": n_pts,
               "h5_em_disco_no_fim": len(list(RAW.glob("*.h5")))}
    (OUT / f"resumo_download_{'_'.join(args.seasons)}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "falhas"}, indent=2,
                     ensure_ascii=False))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
