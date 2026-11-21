"""Migração da linha de aterramento — passo 6: agregação por plataforma dos estimadores do passo 4.

Por perfil: Δ_maré (m por m de maré) e Δ_tempo (m/ano), com σ do bootstrap de ciclos. Filtros
fixados antes de olhar os resultados: β no gelo livre entre 0,7 e 1,3; meia flexão do perfil médio
entre −3 e +6 km da linha do BedMachine; ≥ 80% das réplicas válidas; σ > 0.
Agregado por plataforma: média ponderada por 1/σ² e, como os σ de bootstrap podem estar subestimados,
também a mediana com IC de 95% por bootstrap sobre perfis (2.000 réplicas) e o qui-quadrado reduzido
dos perfis em torno da média ponderada (χ²_ν ≫ 1: dispersão real entre perfis ou σ otimista).
Saída: outputs/gl_migration/agregado_plataformas.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "gl_migration"
PLAT = ["Pine_Island", "Thwaites", "Crosson", "Dotson", "Cosgrove"]
RNG = np.random.default_rng(20261117)


def resumo(v, s):
    w = 1 / s ** 2
    mw = np.sum(w * v) / np.sum(w)
    chi = np.sum(w * (v - mw) ** 2) / max(len(v) - 1, 1)
    meds = [np.median(RNG.choice(v, len(v))) for _ in range(2000)]
    return dict(n=len(v), media_pond=mw, sigma_pond=np.sqrt(1 / np.sum(w)) * np.sqrt(max(chi, 1)),
                chi2_red=chi, mediana=np.median(v), med_lo=np.percentile(meds, 2.5),
                med_hi=np.percentile(meds, 97.5), frac_neg=np.mean(v < 0))


def main():
    linhas = []
    for p in PLAT:
        f = OUT / f"subconjuntos_{p}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f)
        d = d[d.beta_far.between(0.7, 1.3) & d.x50_ref.between(-3000, 6000)]
        for qual, col, sd, okc in (("maré (m por m)", "mare_m_por_m", "mare_sd", "mare_ok"),
                                   ("tempo (m/ano)", "tempo_m_ano", "tempo_sd", "tempo_ok")):
            q = d[(d[okc] >= 0.8) & (d[sd] > 0) & np.isfinite(d[col])]
            if len(q) < 3:
                linhas.append(dict(plataforma=p, grandeza=qual, n=len(q)))
                continue
            linhas.append(dict(plataforma=p, grandeza=qual, **resumo(q[col].to_numpy(), q[sd].to_numpy())))
    r = pd.DataFrame(linhas)
    r.to_csv(OUT / "agregado_plataformas.csv", index=False)
    with pd.option_context("display.width", 200):
        print(r.round(2).to_string())


if __name__ == "__main__":
    main()
