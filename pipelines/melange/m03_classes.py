"""Mélange e fraturas da Thwaites, passo 3: classe de superfície por janela, mapas por ano e séries.

Entrada: data/interim/melange/janelas_1km.parquet e limiares.json (passo 2).
Referência de gelo coeso: janelas da plataforma leste (TEIS, domínio do produto basal v2, grade de
2 km), com retorno em ≥ 50% da janela. Limiares = percentis dessa referência (P_COESO):
    coeso      f_grosso ≥ p05(TEIS) e rug ≤ p95(TEIS) e trans ≤ p95(TEIS)
    mélange    f_grosso ≥ 0,1 e não coeso
    fino       f_grosso < 0,1 e f_agua < 0,5    (gelo fino, brash, gelo marinho espesso)
    água       f_grosso < 0,1 e f_agua ≥ 0,5    (água aberta ou gelo marinho fino: < 2 m acima da água)
Janelas com retorno < 50% ficam sem classe (nuvem, fendas íngremes).
Regiões: TEIS e TWGT (domínio basal v2); "baía" = resto da caixa fora do gelo apoiado do BedMachine.
Sensibilidade: variante estrita (coeso exige f_grosso ≥ 0,9).
Saídas em outputs/melange/: classes_janelas.parquet (só colunas úteis), grade_ano_2km.csv,
serie_regioes.csv, limiares_classes.json e prévias PNG.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "data" / "interim" / "melange"
OUT = ROOT / "outputs" / "melange"
CL = ["água", "fino", "mélange", "coeso"]


def k2(x, y):
    return (np.round(x / 2000) * 2).astype(int).astype(str) + "_" + (np.round(y / 2000) * 2).astype(int).astype(str)


def classifica(j, lim, fmin_coeso):
    coeso = (j.f_grosso >= fmin_coeso) & (j.rug.fillna(np.inf) <= lim["rug_max"]) & (j.trans <= lim["trans_max"])
    c = np.where(coeso, "coeso",
                 np.where(j.f_grosso >= 0.1, "mélange", np.where(j.f_agua < 0.5, "fino", "água")))
    return np.where(j.f_ret >= 0.5, c, "sem")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    j = pd.read_parquet(DIR / "janelas_1km.parquet")
    dom = pd.read_csv(ROOT / "manuscrito" / "figuras" / "dados" / "R_fig1_dominio_basal_2km.csv")
    dom = dom[dom.unidade.str.startswith("Thwaites")].copy()
    dom["k"] = k2(dom.x_km.to_numpy() * 1000, dom.y_km.to_numpy() * 1000)
    j["k"] = k2(j.x.to_numpy(), j.y.to_numpy())
    j = j.merge(dom[["k", "unidade"]], on="k", how="left")
    j["regiao"] = np.where(j.unidade == "Thwaites_TEIS", "TEIS",
                           np.where(j.unidade == "Thwaites_TWGT", "TWGT", np.where(j.bm == 2, "apoiado", "baía")))
    ref = j[(j.regiao == "TEIS") & (j.f_ret >= 0.5)]
    lim = {"f_grosso_min": float(ref.f_grosso.quantile(0.05)), "rug_max": float(ref.rug.quantile(0.95)),
           "trans_max": float(ref.trans.quantile(0.95)), "n_ref": int(len(ref))}
    lim.update(json.loads((DIR / "limiares.json").read_text()))
    (OUT / "limiares_classes.json").write_text(json.dumps(lim, indent=1))
    print("limiares:", {k: round(v, 3) if isinstance(v, float) else v for k, v in lim.items()})
    j["classe"] = classifica(j, lim, lim["f_grosso_min"])
    j["classe_estrita"] = classifica(j, lim, 0.9)
    j = j[j.regiao != "apoiado"]
    j[["x", "y", "t", "est", "ano", "rgt", "cycle", "regiao", "classe", "classe_estrita", "f_grosso", "rug", "trans",
       "f_ret", "hfw_med"]].to_parquet(OUT / "classes_janelas.parquet", index=False)

    # grade por ano (2 km): fração de cada classe entre as janelas classificadas
    v = j[j.classe != "sem"].copy()
    v["gx"] = (np.round(v.x / 2000) * 2).astype(int)
    v["gy"] = (np.round(v.y / 2000) * 2).astype(int)
    gr = (v.groupby(["ano", "gx", "gy", "classe"]).size().unstack(fill_value=0).reindex(columns=CL, fill_value=0))
    gr["n"] = gr.sum(axis=1)
    for c in CL:
        gr["f_" + c] = gr[c] / gr.n
    gr = gr.reset_index()
    gr.to_csv(OUT / "grade_ano_2km.csv", index=False)

    # séries por região e época (estação × ano); bootstrap por RGT (unidade independente = passagem)
    v["epoca"] = v.ano + np.select([v.est == 0, v.est == 1, v.est == 2, v.est == 3], [0.54, 0.04, 0.79, 0.29])
    rng = np.random.default_rng(20261118)
    linhas = []
    for (rg, ep), g in v.groupby(["regiao", "epoca"]):
        if len(g) < 30:
            continue
        rgts = g.rgt.unique()
        fr = g.classe.value_counts(normalize=True).reindex(CL, fill_value=0)
        fe = (g.classe_estrita == "coeso").mean()
        boot = []
        for _ in range(200):
            s = rng.choice(rgts, len(rgts))
            b = pd.concat([g[g.rgt == r] for r in s])
            boot.append(b.classe.value_counts(normalize=True).reindex(CL, fill_value=0).to_numpy())
        boot = np.array(boot)
        lin = dict(regiao=rg, epoca=ep, n=len(g), n_rgt=len(rgts), coeso_estrito=fe)
        for i, c in enumerate(CL):
            lin[c] = fr[c]; lin[c + "_lo"] = np.percentile(boot[:, i], 2.5); lin[c + "_hi"] = np.percentile(boot[:, i], 97.5)
        linhas.append(lin)
    s = pd.DataFrame(linhas).sort_values(["regiao", "epoca"])
    s.to_csv(OUT / "serie_regioes.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(s[["regiao", "epoca", "n", "n_rgt", "água", "fino", "mélange", "coeso", "coeso_estrito"]].round(3).to_string())

    # prévias
    anos = sorted(gr.ano.unique())
    fig, ax = plt.subplots(2, len(anos), figsize=(3 * len(anos), 6.2), sharex=True, sharey=True)
    for i, a in enumerate(anos):
        g = gr[(gr.ano == a) & (gr.n >= 3)]
        for r, (c, cm) in enumerate([("f_coeso", "Blues"), ("f_mélange", "Oranges")]):
            ax[r, i].scatter(g.gx, g.gy, c=g[c], s=4, marker="s", cmap=cm, vmin=0, vmax=1)
            ax[r, i].set_title(f"{a} {c[2:]}", fontsize=8); ax[r, i].set_aspect("equal")
    plt.tight_layout(); plt.savefig(OUT / "previa_mapas_ano.png", dpi=110)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6), sharey=True)
    for i, rg in enumerate(["TEIS", "TWGT", "baía"]):
        g = s[s.regiao == rg]
        for c, cor in zip(CL, ["#2166AC", "#9ECAE1", "#E39B2C", "#444444"]):
            ax[i].plot(g.epoca, g[c], "-o", ms=3, color=cor, label=c)
            ax[i].fill_between(g.epoca, g[c + "_lo"], g[c + "_hi"], color=cor, alpha=0.15)
        ax[i].set_title(rg); ax[i].axvline(2024.15, color="gold")
    ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(OUT / "previa_series.png", dpi=110)


if __name__ == "__main__":
    main()
