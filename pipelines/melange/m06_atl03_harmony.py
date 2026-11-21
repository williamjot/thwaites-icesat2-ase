"""Mélange e fraturas da Thwaites, passo 6: fótons ATL03 recortados no servidor (NASA Harmony).

Trilhas RGT 1079 e 5 (cruzam a margem de cisalhamento da TEIS), ATL03 v007, 2018-10..2025-12.
Recorte espacial no servidor (caixa da margem, lon −108,69..−105,69, lat −75,54..−74,94) — chega um
HDF5 pequeno por grânulo. Padrão do projeto mantido: UM grânulo por vez; baixa para data/raw_temp,
extrai só as variáveis necessárias para parquet leve e apaga o .h5 num try/finally.
Só fótons com signal_conf_ph (terra) ≥ 2. Variáveis por feixe (gt1l..gt3r): heights/{lat_ph, lon_ph, h_ph, delta_time, signal_conf_ph[:,0] (terra),
quality_ph}; geolocation/{segment_id, ph_index_beg, segment_ph_cnt, ...} não são necessários para fendas.
Autenticação: credenciais Earthdata do usuário (arquivo .netrc já existente; nada é lido aqui).
Uso: python m06_atl03_harmony.py [--teste]   (--teste: só o primeiro grânulo, mostra o tamanho)
Saída: data/interim/melange/atl03/<granulo>.parquet; log em outputs/melange/atl03_log.csv
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TMP = ROOT / "data" / "raw_temp"
OUT = ROOT / "data" / "interim" / "melange" / "atl03"
LOG = ROOT / "outputs" / "melange" / "atl03_log.csv"
BBOX = (-108.69, -75.54, -105.69, -74.94)   # W, S, E, N
RGTS = (1079, 5)
BEAMS = ["gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"]


def granulos():
    import earthaccess
    g = earthaccess.search_data(short_name="ATL03", version="007", bounding_box=BBOX,
                                temporal=("2018-10-01", "2025-12-31"))
    sel = []
    for x in g:
        nm = x["umm"]["GranuleUR"]
        m = re.search(r"ATL03_\d{14}_(\d{4})(\d{2})", nm)
        if m and int(m.group(1)) in RGTS:
            sel.append((nm, x["meta"]["collection-concept-id"]))
    return sorted(set(sel))


def extrai(h5, nome):
    partes = []
    with h5py.File(h5, "r") as f:
        for b in BEAMS:
            if f"{b}/heights/h_ph" not in f:
                continue
            hgt = f[f"{b}/heights"]
            n = hgt["h_ph"].shape[0]
            if n == 0:
                continue
            partes.append(pd.DataFrame({
                "lat": hgt["lat_ph"][:].astype(np.float64), "lon": hgt["lon_ph"][:].astype(np.float64),
                "h": hgt["h_ph"][:].astype(np.float32), "dt": hgt["delta_time"][:],
                "conf": hgt["signal_conf_ph"][:, 0].astype(np.int8),
                "qph": hgt["quality_ph"][:].astype(np.int8), "beam": b,
                "forte": bool(f[b].attrs.get("atlas_beam_type", b"") in (b"strong", "strong")),
            }))
    if not partes:
        return None
    d = pd.concat(partes, ignore_index=True)
    d = d[d.conf >= 2]          # sinal de confiança média ou alta (superfície); ruído de fundo fora
    d["granulo"] = nome
    return d


def main():
    from harmony import BBox, Client, Collection, Request
    teste = "--teste" in sys.argv
    TMP.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    lista = granulos()
    print(f"{len(lista)} grânulos ATL03 v007 nas RGT {RGTS}")
    cli = Client()   # usa o .netrc do Earthdata
    log = []
    for nome, coll in (lista[:1] if teste else lista):
        alvo = OUT / f"{nome.replace('.h5', '')}.parquet"
        if alvo.exists():
            continue
        t0 = time.time()
        arquivos = []
        try:
            req = Request(collection=Collection(id=coll), spatial=BBox(*BBOX), granule_name=[nome])
            job = cli.submit(req)
            cli.wait_for_processing(job, show_progress=False)
            arquivos = [r.result() for r in cli.download_all(job, directory=str(TMP), overwrite=True)]
            tam = sum(Path(a).stat().st_size for a in arquivos) / 1e6
            dd = [extrai(a, nome) for a in arquivos if str(a).endswith((".h5", ".hdf5", ".nc4", ".h5.h5"))]
            dd = [x for x in dd if x is not None]
            n = 0
            if dd:
                d = pd.concat(dd, ignore_index=True)
                d.to_parquet(alvo, index=False)
                n = len(d)
            log.append(dict(granulo=nome, mb=round(tam, 2), fotons=n, s=round(time.time() - t0, 1), ok=True))
            print(f"{nome}: {tam:.1f} MB recortado, {n:,} fótons, {time.time() - t0:.0f} s")
        except Exception as e:      # registra e segue para o próximo grânulo
            log.append(dict(granulo=nome, mb=np.nan, fotons=0, s=round(time.time() - t0, 1), ok=False, erro=str(e)[:300]))
            print(f"{nome}: ERRO {e}")
        finally:
            for a in arquivos:
                Path(a).unlink(missing_ok=True)
        pd.DataFrame(log).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
        log = []
    sobra = list(TMP.glob("*"))
    print("data/raw_temp vazio:", not sobra, [s.name for s in sobra][:5])


if __name__ == "__main__":
    main()
