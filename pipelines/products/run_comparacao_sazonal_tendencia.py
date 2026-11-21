# -*- coding: utf-8 -*-
"""Comparacao sazonal pela TENDENCIA, em nos comuns (correcao 3).

Por que nao comparar as massas acumuladas de outputs/<estacao>/tables/
serie_massa.json: aquelas curvas sao anomalias em caixas anuais, sensiveis ao
binning e a epoca efetiva de cada caixa, e cada estacao tem o seu proprio
conjunto de nos. Aqui:

  * mesmo conjunto de nos nas duas estacoes (interseccao exata);
  * mesmo produto dos dois lados (dhdt bruto de dhdt_nodes_qc.parquet) - nunca
    um lado corrigido de firn e o outro nao;
  * diferenca PAREADA por no (JJA - DJF), que cancela a estrutura espacial
    comum e e o teste certo para "uma estacao perde mais que a outra";
  * incerteza por bootstrap de blocos espaciais, reportada com o tamanho do
    bloco declarado e a sensibilidade a ele (o comprimento de correlacao deste
    produto varia por um fator ~4 conforme o estimador: 34,1 km dos residuos de
    CV, 154,4 km usado no balanco de JJA).

Saida: outputs/comparacao_sazonal/comparacao_tendencia_nos_comuns.{json,png,pdf}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from thwaites.legacy_uncertainty import spatial_block_bootstrap      # noqa: E402

RHO_I = 917.0
NODE_AREA_M2 = 25.0e6                 # nos de 5 km (dhdt.node_spacing_m)
BLOCOS_KM = (20.0, 34.1, 154.4)
OUT = ROOT / "outputs" / "comparacao_sazonal"
OUT.mkdir(parents=True, exist_ok=True)


def gt_por_ano(media_m_ano: float, n_nos: int) -> float:
    return media_m_ano * NODE_AREA_M2 * n_nos * RHO_I / 1e12


def main():
    j = pd.read_parquet(ROOT / "data/jja/dhdt/dhdt_nodes_qc.parquet")
    d = pd.read_parquet(ROOT / "data/djf/dhdt/dhdt_nodes_qc.parquet")
    m = j.merge(d, on=["x", "y"], suffixes=("_jja", "_djf"))
    m = m[np.isfinite(m.dhdt_jja) & np.isfinite(m.dhdt_djf)].copy()
    m["dif"] = m.dhdt_jja - m.dhdt_djf
    area = np.full(len(m), NODE_AREA_M2)
    n = len(m)
    area_km2 = n * NODE_AREA_M2 / 1e6

    rng = np.random.default_rng(20260920)
    res = {"n_nos_comuns": int(n), "area_km2": float(area_km2),
           "produto": "dhdt bruto (dhdt_nodes_qc.parquet) nas duas estacoes",
           "tspan_mediano_anos": {"JJA": float(j.tspan.median()),
                                  "DJF": float(d.tspan.median())},
           "por_bloco": {}}
    for bkm in BLOCOS_KM:
        linha = {}
        for nome, col in (("JJA", "dhdt_jja"), ("DJF", "dhdt_djf"), ("JJA_menos_DJF", "dif")):
            b = spatial_block_bootstrap(m.x, m.y, m[col], area, block_km=bkm,
                                        n_boot=2000, rng=rng)
            linha[nome] = {
                "media_m_ano": b["mean"], "sigma_m_ano": b["sigma_mean"],
                "ci95_m_ano": [b["ci_low"], b["ci_high"]],
                "Gt_ano": gt_por_ano(b["mean"], n),
                "sigma_Gt_ano": gt_por_ano(b["sigma_mean"], n),
                "ci95_Gt_ano": [gt_por_ano(b["ci_low"], n), gt_por_ano(b["ci_high"], n)],
                "n_blocos": b["n_blocks"],
            }
        linha["diferenca_significativa_95"] = bool(
            linha["JJA_menos_DJF"]["ci95_m_ano"][0] > 0
            or linha["JJA_menos_DJF"]["ci95_m_ano"][1] < 0)
        res["por_bloco"][f"{bkm:g}"] = linha

    conf = m[(m.reliability_jja.astype(str).str.startswith("conf"))
             & (m.reliability_djf.astype(str).str.startswith("conf"))]
    if len(conf) > 50:
        b = spatial_block_bootstrap(conf.x, conf.y, conf.dif,
                                    np.full(len(conf), NODE_AREA_M2),
                                    block_km=34.1, n_boot=2000, rng=rng)
        res["sensibilidade_so_nos_confiaveis"] = {
            "n_nos": int(len(conf)), "media_m_ano": b["mean"],
            "sigma_m_ano": b["sigma_mean"],
            "ci95_m_ano": [b["ci_low"], b["ci_high"]],
            "Gt_ano": gt_por_ano(b["mean"], len(conf)),
        }
    (OUT / "comparacao_tendencia_nos_comuns.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------ figura
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.2), facecolor="white")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.80, bottom=0.235, wspace=0.30)
    ref = res["por_bloco"]["34.1"]

    ax = axes[0]
    nomes = ["Inverno\n(JJA)", "Verão\n(DJF)"]
    vals = [ref["JJA"]["Gt_ano"], ref["DJF"]["Gt_ano"]]
    errs = [ref["JJA"]["sigma_Gt_ano"], ref["DJF"]["sigma_Gt_ano"]]
    ax.bar([0, 1], vals, color=["#1F5F99", "#C0620B"], edgecolor="0.25", lw=0.6,
           width=0.55, zorder=3)
    ax.errorbar([0, 1], vals, yerr=errs, fmt="none", ecolor="0.2", elinewidth=1.3,
                capsize=5, zorder=4)
    ax.axhline(0, color="0.3", lw=0.9)
    for xi, v, e in zip([0, 1], vals, errs):
        ax.annotate("%.1f ± %.1f" % (v, e), (xi, v - e), textcoords="offset points",
                    xytext=(0, -14), ha="center", va="top", fontsize=9.5,
                    fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(nomes)
    ax.set_ylabel("Taxa de massa integrada (Gt ano$^{-1}$)")
    ax.set_title("(a) Mesmos %d nós, mesmo produto" % n, fontsize=10.5, loc="left")
    ax.set_ylim(min(vals) * 1.45, 0)
    ax.grid(axis="y", color="0.92", lw=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    dif = ref["JJA_menos_DJF"]
    ax.bar([0], [dif["Gt_ano"]], color="#5B5F97", edgecolor="0.25", lw=0.6,
           width=0.4, zorder=3)
    ax.errorbar([0], [dif["Gt_ano"]],
                yerr=[[dif["Gt_ano"] - dif["ci95_Gt_ano"][0]],
                      [dif["ci95_Gt_ano"][1] - dif["Gt_ano"]]],
                fmt="none", ecolor="0.2", elinewidth=1.3, capsize=6, zorder=4)
    ax.axhline(0, color="#8B1E3F", lw=1.2, ls="--", zorder=2)
    ax.set_xticks([0]); ax.set_xticklabels(["Inverno − Verão\n(pareado por nó)"])
    ax.set_ylabel("Diferença de taxa (Gt ano$^{-1}$)")
    ax.set_title("(b) IC 95%% inclui zero: %s"
                 % ("não" if res["por_bloco"]["34.1"]["diferenca_significativa_95"] else "sim"),
                 fontsize=10.5, loc="left")
    ax.annotate("%+.1f Gt/ano\nIC 95%% [%+.1f, %+.1f]"
                % (dif["Gt_ano"], dif["ci95_Gt_ano"][0], dif["ci95_Gt_ano"][1]),
                (0, dif["Gt_ano"]), textcoords="offset points", xytext=(26, 0),
                ha="left", va="center", fontsize=9.5)
    lim = max(abs(dif["ci95_Gt_ano"][0]), abs(dif["ci95_Gt_ano"][1])) * 2.1
    ax.set_xlim(-0.6, 0.9); ax.set_ylim(-lim, lim)
    ax.grid(axis="y", color="0.92", lw=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    xs = np.arange(len(BLOCOS_KM))
    dv = [res["por_bloco"][f"{b:g}"]["JJA_menos_DJF"]["Gt_ano"] for b in BLOCOS_KM]
    lo = [res["por_bloco"][f"{b:g}"]["JJA_menos_DJF"]["ci95_Gt_ano"][0] for b in BLOCOS_KM]
    hi = [res["por_bloco"][f"{b:g}"]["JJA_menos_DJF"]["ci95_Gt_ano"][1] for b in BLOCOS_KM]
    ax.errorbar(xs, dv, yerr=[np.array(dv) - np.array(lo), np.array(hi) - np.array(dv)],
                fmt="o", ms=7, color="#5B5F97", ecolor="0.3", elinewidth=1.3,
                capsize=5, zorder=4)
    ax.axhline(0, color="#8B1E3F", lw=1.2, ls="--", zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(["%g km" % b for b in BLOCOS_KM])
    ax.set_xlabel("Tamanho do bloco do bootstrap")
    ax.set_ylabel("Diferença de taxa (Gt ano$^{-1}$)")
    ax.set_title("(c) Sensibilidade ao comprimento de correlação",
                 fontsize=10.5, loc="left")
    ax.set_xlim(-0.5, len(BLOCOS_KM) - 0.5)
    ax.grid(axis="y", color="0.92", lw=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Comparação sazonal pela tendência, em nós comuns — ASE, ICESat-2 ATL06\n"
                 "%d nós (%.0f km²), produto idêntico nos dois lados, diferença pareada por nó"
                 % (n, area_km2), fontsize=12, y=0.965)
    fig.text(0.5, 0.015,
             "Incerteza por bootstrap de blocos espaciais; o bloco de 34,1 km é o comprimento de correlação dos resíduos da validação cruzada, "
             "e o de 154,4 km é o usado no balanço de JJA.\n"
             "As duas estações são ajustadas sobre vãos ligeiramente diferentes (%.2f ano no inverno, %.2f no verão), o que afeta a taxa muito menos que a massa acumulada.\n"
             "Este painel substitui a comparação entre massas acumuladas, que dependia do binning anual e de conjuntos de nós distintos."
             % (res["tspan_mediano_anos"]["JJA"], res["tspan_mediano_anos"]["DJF"]),
             ha="center", va="bottom", fontsize=8.3, color="0.30")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"comparacao_tendencia_nos_comuns.{ext}",
                    dpi=300 if ext == "png" else None, facecolor="white")

    print(json.dumps(res, indent=1, ensure_ascii=False)[:2600])


if __name__ == "__main__":
    main()
