"""Migração da linha de aterramento — passo 3: tendência interanual e migração com a maré.

Entrada: outputs/gl_migration/{flexao_por_ciclo,perfis_qc}_<plataforma>.parquet (passo 2).

Modelo por plataforma, com efeito fixo por perfil p (rgt, par, cruzamento):
    δ_pk = a_p + v·(t_pk − t̄_p) + m·(T_pk − T̄_p) + ε_pk
δ: deslocamento do ponto de meia flexão no ciclo k em relação ao gabarito do perfil (m; positivo =
a jusante). v (m/ano): migração interanual (negativo = recuo para o interior). m (m por m de maré):
migração com a maré (a hidrostática prevê m < 0: maré alta faz a linha recuar).
Estimação: regressão de Huber sobre os dados centrados por perfil; IC de 95% por bootstrap
agrupado por perfil (2.000 réplicas), porque os ciclos de um mesmo perfil não são independentes.

Filtros (decididos antes de ver v e m): β no gelo livre entre 0,7 e 1,3; ponto de meia flexão do
gabarito entre −3 e +6 km da linha do BedMachine; custo do ajuste < 0,6; deslocamento fora das
bordas da busca (±5 km); perfil com ≥ 4 ciclos válidos.

Saída: outputs/gl_migration/migracao_resumo.csv, migracao_por_perfil.parquet, ciclos_validos.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "gl_migration"
PLAT = ["Pine_Island", "Thwaites", "Crosson", "Dotson", "Cosgrove"]
RNG = np.random.default_rng(20261117)
NBOOT = 2000


def huber(X, y, k=1.345, it=30):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(it):
        r = y - X @ b
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        u = np.abs(r) / (k * s)
        w = np.where(u <= 1, 1.0, 1.0 / u)
        sw = np.sqrt(w)
        b = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    return b


def centra(d):
    g = d.groupby("pid")
    return d.assign(yc=d.delta - g.delta.transform("mean"), tc=d.t - g.t.transform("mean"),
                    Tc=d["T"] - g["T"].transform("mean"))


def ajusta(d):
    d = centra(d)
    X = np.c_[d.tc.values, d.Tc.values]
    return huber(X, d.yc.values)


def carrega(p):
    r = pd.read_parquet(OUT / f"flexao_por_ciclo_{p}.parquet")
    q = pd.read_parquet(OUT / f"perfis_qc_{p}.parquet")
    q = q[(q.beta_far.between(0.7, 1.3)) & (q.tx50.between(-3000, 6000))]
    r = r.merge(q[["rgt", "par", "cz", "beta_far", "tx50"]], on=["rgt", "par", "cz"])
    r = r[r.delta.notna() & (r.custo < 0.6) & (~r.borda.astype(bool))]
    r["pid"] = r.rgt.astype(str) + "_" + r.par.astype(str) + "_" + r.cz.astype(str)
    n = r.groupby("pid").delta.transform("size")
    r = r[n >= 4].copy()
    r["plataforma"] = p
    return r


def main(plats):
    linhas, perfis, todos = [], [], []
    for p in plats:
        d = carrega(p)
        print(p, "perfis válidos", d.pid.nunique(), "ciclos", len(d))
        if d.pid.nunique() < 5:
            continue
        todos.append(d)
        v, m = ajusta(d)
        pids = d.pid.unique()
        dc = centra(d)
        grupos = {k: (g[["tc", "Tc"]].to_numpy(), g.yc.to_numpy()) for k, g in dc.groupby("pid")}
        bs = []
        for _ in range(NBOOT):
            amostra = RNG.choice(pids, len(pids), replace=True)
            X = np.vstack([grupos[k][0] for k in amostra])
            y = np.concatenate([grupos[k][1] for k in amostra])
            bs.append(huber(X, y))
        bs = np.array(bs)
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        linhas.append(dict(plataforma=p, n_perfis=len(pids), n_ciclos=len(d), v_m_ano=v,
                           v_lo=lo[0], v_hi=hi[0], m_m_por_m=m, m_lo=lo[1], m_hi=hi[1],
                           dp_residuo_m=float(np.std(centra(d).yc - np.c_[centra(d).tc, centra(d).Tc] @ [v, m])),
                           tx50_mediana_m=float(d.groupby("pid").tx50.first().median())))
        for k, g in d.groupby("pid"):
            if len(g) >= 6 and g["T"].std() > 0.25 and (g.t.max() - g.t.min()) > 3:
                vv, mm = ajusta(g)
                perfis.append(dict(plataforma=p, pid=k, n=len(g), v=vv, m=mm, tx50=g.tx50.iloc[0],
                                   X=g.x50_X.median(), Y=g.x50_Y.median()))
        print(linhas[-1])
    pd.DataFrame(linhas).to_csv(OUT / "migracao_resumo.csv", index=False)
    pd.DataFrame(perfis).to_parquet(OUT / "migracao_por_perfil.parquet", index=False)
    pd.concat(todos).to_parquet(OUT / "ciclos_validos.parquet", index=False)


if __name__ == "__main__":
    main(sys.argv[1:] or PLAT)
