"""InSAR + ICESat-2, passo E: recuo da linha com desvio (offset) por sensor/processador.

Problema (passo D): os sensores InSAR divergem em até ~1 km na mesma época, e o CSK (até 2021) não
se sobrepõe aos sensores recentes (2023–2025); o recuo fica confundido com o viés entre sensores.
Novo dado: ESA Antarctic Ice Sheet CCI, Grounding line location for key glaciers, v3.1 (CEDA,
doi no registro e041b3b6d8b348afaff345a354ba382b): na ASE, Sentinel-1 2014–2021, delineação
automática, com a maré de cada aquisição.

Modelo por geleira (todas as linhas de fluxo do passo B):
    s_{f,i} = a_f + v·(t_i − 2021) + o_{k(i)} + ε
a_f: intercepto por linha de fluxo; v: recuo (m/ano); o_k: desvio do sensor/processador k
(referência: o_ref = 0). Os desvios são identificáveis só onde há sobreposição temporal. A ligação
entre os dois períodos exige uma hipótese, testada em duas versões:
  H1 "Sentinel-1 comum": S1-CCI (DLR), S1 NSIDC e S1 Andersen 2025 têm o mesmo desvio (mesmo sensor
     e mesma técnica DInSAR);
  H2 "CSK ≈ ICEYE": os dois sensores de banda X de alta resolução têm o mesmo desvio.
Estimação: Huber sobre um valor por (linha de fluxo, sensor, mês); IC por bootstrap em blocos de
10 km (1.000). Saída: outputs/gl_migration/recuo_vies_sensor.csv e desvios_sensor.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p03_migracao import huber  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "gl_migration"
RNG = np.random.default_rng(20261117)
ANO0 = float(sys.argv[1]) if len(sys.argv) > 1 else 2016.0
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
SUF = f"_{int(ANO0)}"


def cruzamentos_cci():
    vert = pd.read_parquet(OUT / "fluxo_linhas.parquet")
    cci = gpd.read_file(ROOT / "data" / "grounding" / "processed" / "ESACCI_GLL_v3.1_ASE.gpkg")
    cci = cci[cci.NAME == "SEN"].explode(index_parts=False)
    t = cci[["T1", "T2", "T3", "T4"]].apply(pd.to_datetime, errors="coerce").mean(axis=1)
    cci["data"] = t
    cci["ano_dec"] = t.dt.year + (t.dt.dayofyear - 0.5) / 365.25
    sidx = cci.sindex
    out = []
    for fid, v in vert.groupby("fid"):
        v = v.sort_values("s")
        ls = LineString(np.c_[v.x, v.y])
        s0 = v.s.iloc[0]
        for j in sidx.query(ls, predicate="intersects"):
            row = cci.iloc[j]
            inter = ls.intersection(row.geometry)
            pts = [inter] if inter.geom_type == "Point" else [q for q in getattr(inter, "geoms", []) if q.geom_type == "Point"]
            if not pts:
                continue
            ss = [ls.project(q) + s0 for q in pts]
            k = int(np.argmin(np.abs(ss)))
            if abs(ss[k]) <= 12000:
                out.append(dict(fid=fid, geleira=v.geleira.iloc[0], sensor="Sentinel-1 (CCI)", data=row.data,
                                ano_dec=row.ano_dec, s=ss[k], n_cruz=len(pts), dhf=row.DHF))
    return pd.DataFrame(out)


def ajusta(d, sensores, ref, liga):
    """liga: dict sensor -> sensor com o qual compartilha o desvio."""
    d = d.copy()
    d["k"] = d.sensor.map(lambda s: liga.get(s, s))
    ks = [k for k in sorted(d.k.unique()) if k != ref]
    # centra por linha de fluxo (efeito fixo) via variáveis indicadoras centradas
    g = d.groupby("fid")
    y = (d.s - g.s.transform("mean")).to_numpy()
    cols = [d.ano_dec - g.ano_dec.transform("mean")]
    for k in ks:
        ind = (d.k == k).astype(float)
        cols.append(ind - ind.groupby(d.fid).transform("mean"))
    X = np.column_stack(cols)
    b = huber(X, y)
    return b[0], dict(zip(ks, b[1:]))


def main():
    cr = pd.read_parquet(OUT / "recuo_insar_cruzamentos.parquet")
    cc = cruzamentos_cci()
    cc.to_parquet(OUT / "recuo_cci_cruzamentos.parquet", index=False)
    todos = pd.concat([cr, cc], ignore_index=True)
    todos = todos[todos.ano_dec >= ANO0]
    todos["mes"] = pd.to_datetime(todos.data).dt.to_period("M")
    d = todos.groupby(["geleira", "fid", "sensor", "mes"]).agg(s=("s", "median"), ano_dec=("ano_dec", "median")).reset_index()
    pos = pd.read_parquet(OUT / "fluxo_linhas.parquet").query("s == 0")[["fid", "x", "y"]]
    d = d.merge(pos, on="fid")
    d["bloco"] = (d.x // 10000).astype(int).astype(str) + "_" + (d.y // 10000).astype(int).astype(str)
    print(d.groupby(["geleira", "sensor"]).agg(n=("s", "size"), a0=("ano_dec", "min"), a1=("ano_dec", "max")).round(1).to_string())
    S1 = ["Sentinel-1 (CCI)", "Sentinel-1", "Sentinel-1 (Andersen 2025)"]
    hip = {"H1 Sentinel-1 comum": ({s: "S1" for s in S1}, "S1"),
           "H2 CSK = ICEYE": ({"ICEYE": "CSK"}, "CSK")}
    linhas, desv = [], []
    for gl, g in d.groupby("geleira"):
        for nome, (liga, ref) in hip.items():
            gg = g.copy()
            kk = gg.sensor.map(lambda s: liga.get(s, s))
            if ref not in set(kk):
                continue
            # só sensores ligados à referência por sobreposição ou pela hipótese
            v, o = ajusta(gg, None, ref, liga)
            blocos = gg.bloco.unique()
            reps = []
            for _ in range(NB):
                bb = RNG.choice(blocos, len(blocos))
                s = pd.concat([gg[gg.bloco == b].assign(fid=gg[gg.bloco == b].fid.astype(str) + f"#{i}") for i, b in enumerate(bb)])
                try:
                    reps.append(ajusta(s, None, ref, liga)[0])
                except Exception:
                    pass
            linhas.append(dict(geleira=gl, hipotese=nome, v_m_ano=v, lo=np.percentile(reps, 2.5),
                               hi=np.percentile(reps, 97.5), n=len(gg), n_blocos=len(blocos)))
            for k, val in o.items():
                desv.append(dict(geleira=gl, hipotese=nome, sensor=k, desvio_m=val))
            print(linhas[-1])
    pd.DataFrame(linhas).to_csv(OUT / f"recuo_vies_sensor{SUF}.csv", index=False)
    pd.DataFrame(desv).to_csv(OUT / f"desvios_sensor{SUF}.csv", index=False)
    print(pd.DataFrame(desv).round(0).to_string())


if __name__ == "__main__":
    main()
