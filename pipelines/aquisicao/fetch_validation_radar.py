"""Baixa os dois conjuntos de radar para validação (autorizado pelo pesquisador em 2026-11-15).

1. **ApRES ITGC MELT, TEIS G1–G4, 2020** (BAS PDC 01641, doi:10.5285/b81bff87-3429-4754-b4b5-c6b58198e744,
   OGL v3). O conjunto inteiro tem 25 GB de netCDF brutos (um arquivo de ~11 MB a cada ~12 h).
   Para a taxa média de derretimento basta acompanhar a fase ao longo do ano, com deslocamento
   < λ/4 (~14 cm no gelo a 300 MHz) entre amostras; com derretimento + deformação de alguns m/a,
   isso exige ≤ 5 dias entre amostras. Baixa-se o primeiro arquivo de cada janela de 5 dias de
   cada ponto (~280 arquivos, ~3 GB).
2. **IceBridge MCoRDS L2 (IRMCR2)** sobre a ASE (bbox −115/−77/−95/−73,5): 84 CSV, ~499 MB,
   2009–2018. Usado no teste do erro hidrostático (superfície e espessura do mesmo voo).

Saídas: ``data/validation/apres_teis/<ponto>/*.nc`` e ``data/validation/mcords/*.csv``.
Retomável: arquivo já presente com o tamanho certo é pulado.
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import certifi

ROOT = Path(__file__).resolve().parents[2]
APRES = ROOT / "data" / "validation" / "apres_teis"
MCORDS = ROOT / "data" / "validation" / "mcords"
BASE = "https://ramadda.data.bas.ac.uk/repository/entry"
ROOT_ID = "b81bff87-3429-4754-b4b5-c6b58198e744"
CTX = ssl.create_default_context(cafile=certifi.where())
STEP_DAYS = 5


def ls(eid):
    u = f"{BASE}/show?entryid={urllib.parse.quote(eid, safe='')}&output=json"
    return json.loads(urllib.request.urlopen(u, context=CTX, timeout=120).read().decode())


def get(eid, dst, size):
    if dst.is_file() and dst.stat().st_size == size:
        return False
    u = f"{BASE}/get?entryid={urllib.parse.quote(eid, safe='')}"
    tmp = dst.with_suffix(dst.suffix + ".partial")
    with urllib.request.urlopen(u, context=CTX, timeout=600) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    if tmp.stat().st_size != size:
        tmp.unlink()
        raise RuntimeError(f"tamanho errado em {dst.name}")
    tmp.replace(dst)
    return True


def fetch_apres():
    manifest = []
    for site in ls(ROOT_ID):
        files = []
        for d in ls(site["id"]):
            for f in ls(d["id"]):
                if f.get("typeName") == "File" and f["name"].endswith(".nc"):
                    t = datetime.strptime(f["name"][4:19], "%Y-%m-%d-%H%M")
                    files.append((t, f["name"], f["id"], int(f["filesize"])))
        files.sort()
        picked, last = [], None
        for t, n, i, s in files:
            if last is None or (t - last).total_seconds() >= STEP_DAYS * 86400:
                picked.append((t, n, i, s))
                last = t
        out = APRES / site["name"]
        out.mkdir(parents=True, exist_ok=True)
        tot = sum(p[3] for p in picked) / 1e9
        print(f"{site['name']}: {len(files)} arquivos no total; {len(picked)} escolhidos "
              f"({tot:.2f} GB); {files[0][0]:%Y-%m-%d} a {files[-1][0]:%Y-%m-%d}", flush=True)
        for t, n, i, s in picked:
            for attempt in range(3):
                try:
                    get(i, out / n, s)
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"   tentativa {attempt + 1} falhou em {n}: {e!r}"[:160], flush=True)
            manifest.append({"ponto": site["name"], "arquivo": n, "tempo": t.isoformat(),
                             "bytes": s})
    (APRES / "manifesto.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")


def fetch_mcords():
    import earthaccess
    earthaccess.login(strategy="netrc")
    g = earthaccess.search_data(short_name="IRMCR2", bounding_box=(-115.0, -77.0, -95.0, -73.5),
                                count=-1)
    MCORDS.mkdir(parents=True, exist_ok=True)
    got = earthaccess.download(g, str(MCORDS))
    print(f"MCoRDS: {len(got)} arquivos em {MCORDS}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=["apres", "mcords"])
    a = ap.parse_args()
    fetch_apres() if a.what == "apres" else fetch_mcords()
    return 0


if __name__ == "__main__":
    sys.exit(main())
