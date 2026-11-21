"""Mélange e fraturas da Thwaites, passo 8: CSVs das figuras (pôster). Nada de método novo — só reorganiza
saídas dos passos 3–7, com a zona de mélange escolhida SÓ com 2019–20 (fora do período testado).

Saídas em manuscrito/figuras/dados/melange/:
  melange_zona_2km.csv      células da zona de mélange (linha de base 2019–20: ≥ 5 janelas, ≥ 30% gelo grosso)
  melange_dif_6km.csv       por bloco de 6 km na baía: fração de gelo grosso 2024–25 − 2021–22 (≥ 15 janelas
                            em cada período); frac_zona = fração das janelas dentro da zona de mélange; 2019–20 fica de fora porque escolheu a zona
  melange_serie_ano.csv     fração de gelo grosso por ano, zona de mélange e TEIS (controle); IC 95% por
                            bootstrap das passagens (RGT), 500 réplicas; flag selecao = ano usado para escolher
  melange_mudanca.json      2024–25 − 2021–22 na zona, IC emparelhado por réplica
  fendas_periodo.csv        fendas ATL03 por km, por zona (margem, meio, interior), RGT e período
                            (2018–21, 2022–23, 2024–25); IC 95% por bootstrap das passagens (granulo)
  fendas_perfil.csv         trechos de 1,5 km de um perfil ATL03 (margem e interior, mesma passagem e feixe),
                            fótons de sinal e fendas detectadas no passo 7, para ilustração
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "melange"
DST = ROOT / "manuscrito" / "figuras" / "dados" / "melange"
RNG = np.random.default_rng(20261118)
NB = 500


def boot_frac(g, col="grosso"):
    rg = g.rgt.to_numpy(); v = g[col].to_numpy(); u = np.unique(rg)
    idx = {r: np.flatnonzero(rg == r) for r in u}
    return np.array([v[np.concatenate([idx[r] for r in RNG.choice(u, len(u))])].mean() for _ in range(NB)])


def melange():
    j = pd.read_parquet(OUT / "classes_janelas.parquet")
    j = j[j.classe != "sem"].copy()
    j["grosso"] = j.classe.isin(["mélange", "coeso"])
    j["gx"] = (np.round(j.x / 2000) * 2).astype(int)
    j["gy"] = (np.round(j.y / 2000) * 2).astype(int)
    baia = j[j.regiao == "baía"]
    base = baia[baia.ano <= 2020].groupby(["gx", "gy"]).agg(n=("grosso", "size"), f=("grosso", "mean"))
    zm = base[(base.n >= 5) & (base.f >= 0.3)].reset_index()
    zm.to_csv(DST / "melange_zona_2km.csv", index=False)
    j = j.merge(zm[["gx", "gy"]].assign(z=True), on=["gx", "gy"], how="left")
    j["z"] = j.z.fillna(False).astype(bool)

    # mapa de diferença (fora do período de seleção), em blocos de 6 km (≥ 15 janelas em cada período);
    # a 2 km a célula tem poucas janelas por período e o mapa fica dominado por ruído de amostragem
    b6 = baia.assign(bx=(np.floor((baia.x / 1000 + 1650) / 6) * 6 - 1650 + 3).astype(int),
                     by=(np.floor((baia.y / 1000 + 550) / 6) * 6 - 550 + 3).astype(int))
    zk = set(zip(zm.gx, zm.gy))
    b6["z"] = [k in zk for k in zip(b6.gx, b6.gy)]
    a = b6[b6.ano.isin([2021, 2022])].groupby(["bx", "by"]).grosso.agg(["size", "mean"])
    d = b6[b6.ano.isin([2024, 2025])].groupby(["bx", "by"]).grosso.agg(["size", "mean"])
    m = a.join(d, lsuffix="_a", rsuffix="_d", how="inner")
    m = m[(m.size_a >= 15) & (m.size_d >= 15)]
    m = m.join(b6.groupby(["bx", "by"]).z.mean().rename("frac_zona"))
    m.assign(dif=m.mean_d - m.mean_a).reset_index().to_csv(DST / "melange_dif_6km.csv", index=False)

    lin, reps = [], {}
    for nome, msk in [("zona de mélange", j.z), ("TEIS (controle)", j.regiao == "TEIS")]:
        for ano, g in j[msk].groupby("ano"):
            if g.rgt.nunique() < 3:
                continue
            bt = boot_frac(g)
            reps[(nome, ano)] = bt
            lin.append(dict(zona=nome, ano=ano, n=len(g), n_rgt=g.rgt.nunique(), f=g.grosso.mean(),
                            lo=np.percentile(bt, 2.5), hi=np.percentile(bt, 97.5),
                            selecao=bool(nome == "zona de mélange" and ano <= 2020)))
    s = pd.DataFrame(lin)
    s.to_csv(DST / "melange_serie_ano.csv", index=False)
    z = "zona de mélange"
    dif = (reps[(z, 2024)] + reps[(z, 2025)]) / 2 - (reps[(z, 2021)] + reps[(z, 2022)]) / 2
    ss = s[s.zona == z].set_index("ano").f
    res = dict(dif=(ss[2024] + ss[2025]) / 2 - (ss[2021] + ss[2022]) / 2,
               lo=np.percentile(dif, 2.5), hi=np.percentile(dif, 97.5), n_celulas=len(zm))
    (DST / "melange_mudanca.json").write_text(json.dumps(res, indent=1))
    print(s.round(3).to_string()); print(res)


def fendas():
    ev = pd.read_parquet(OUT / "fendas_atl03_eventos.parquet")
    cp = pd.read_parquet(OUT / "fendas_atl03_comprimento.parquet")
    for t in (ev, cp):
        t["rgt"] = t.granulo.str.split("_").str[2].str[:4].astype(int)
        t["per"] = pd.cut(t.ano, [2017, 2021, 2023, 2026], labels=["2018–21", "2022–23", "2024–25"]).astype(str)
    lin = []
    for (z, r, p), c in cp[cp.zona.isin(["margem", "meio", "interior"])].groupby(["zona", "rgt", "per"]):
        e = ev[(ev.zona == z) & (ev.rgt == r) & (ev.per == p)]
        nr = e.groupby("granulo").size().reindex(c.granulo.unique(), fill_value=0)
        kr = c.groupby("granulo").km.sum()
        u = kr.index.to_numpy()
        bt = [nr.loc[s].sum() / kr.loc[s].sum() for s in (RNG.choice(u, len(u)) for _ in range(NB))]
        lin.append(dict(zona=z, rgt=r, periodo=p, km=kr.sum(), n=int(nr.sum()), dens=nr.sum() / kr.sum(),
                        lo=np.percentile(bt, 2.5), hi=np.percentile(bt, 97.5), n_passagens=len(u)))
    s = pd.DataFrame(lin)
    s.to_csv(DST / "fendas_periodo.csv", index=False)
    print(s.round(2).to_string())

    # perfil ilustrativo: passagem da RGT 5 em 2024–25 com mais km na margem e no interior
    c5 = cp[(cp.rgt == 5) & (cp.ano >= 2024)].pivot_table(index=["granulo", "beam"], columns="zona", values="km")
    gr, bm = (c5.margem.clip(upper=8) + c5.interior.clip(upper=8)).idxmax()
    f = next((ROOT / "data" / "interim" / "melange" / "atl03").glob(f"{gr}*.parquet"))
    d = pd.read_parquet(f, columns=["lat", "lon", "h", "dt", "beam"])
    d = d[d.beam == bm].sort_values("dt")
    x, y = Transformer.from_crs(4326, 3031, always_xy=True).transform(d.lon.to_numpy(), d.lat.to_numpy())
    s_ = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    d = d.assign(x=x, y=y, s=s_)
    e = ev[(ev.granulo == gr) & (ev.beam == bm)]
    from scipy.spatial import cKDTree
    t = cKDTree(np.c_[x, y])
    e = e.assign(s=s_[t.query(e[["x", "y"]].to_numpy())[1]])
    import sys; sys.path.insert(0, str(Path(__file__).parent))
    from m07_fendas_atl03 import zonas
    tz, lab = zonas()
    dz, iz = tz.query(np.c_[x, y])
    d["zona"] = np.where(dz <= 1500, lab[iz], "")
    out, oev = [], []
    for zona in ["margem", "interior"]:
        ez = e[e.zona == zona]
        # janela de 1,5 km centrada na maior concentração de fendas da zona (ou no meio da zona)
        sz = ez.s.to_numpy()
        # sem fendas na zona: centro do trecho da zona no perfil
        c0 = (sz[np.argmax([((sz > v - 750) & (sz < v + 750)).sum() for v in sz])] if len(sz)
              else np.median(d.s[d.zona == zona]))
        w = d[(d.s > c0 - 750) & (d.s < c0 + 750)]
        out.append(w.assign(zona=zona, s_rel=w.s - (c0 - 750), h_rel=w.h - np.percentile(w.h, 90))[
            ["zona", "s_rel", "h_rel"]])
        we = ez[(ez.s > c0 - 750) & (ez.s < c0 + 750)]
        oev.append(we.assign(s_rel=we.s - (c0 - 750))[["zona", "tipo", "s_rel", "prof", "largura"]])
    pf = pd.concat(out); pe = pd.concat(oev)
    pf.to_csv(DST / "fendas_perfil.csv", index=False)
    pe.to_csv(DST / "fendas_perfil_eventos.csv", index=False)
    # testes de advecção (m09, m10) para o esquema poster_esquema_fendas.R
    import shutil
    for a in ("advecao_persistencia.csv", "advecao_previsao.csv", "rug_previsao.csv"):
        shutil.copy(ROOT / "outputs" / "melange" / a, DST / a)
    print("perfil:", gr, bm, pf.groupby("zona").size().to_dict(), pe.groupby("zona").size().to_dict())


if __name__ == "__main__":
    DST.mkdir(parents=True, exist_ok=True)
    melange()
    fendas()
