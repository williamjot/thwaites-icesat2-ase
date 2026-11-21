# -*- coding: utf-8 -*-
"""Efeito da correcao de binning sazonal na serie de massa (correcao 2).

(a) massa acumulada contra a EPOCA REAL de cada temporada, nas duas estacoes;
(b) passo entre caixas consecutivas: o binning antigo (floor(t_year)) fazia a
    epoca efetiva de DJF pular, enquanto o binning por temporada nao.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"D:/WILLIAM/ARTIGO PIBIC ANTARTICA")
OUT = ROOT / "outputs" / "comparacao_sazonal"
OUT.mkdir(parents=True, exist_ok=True)

COR_J, COR_D, COR_OLD = "#1F5F99", "#C0620B", "0.55"

nova = {s: json.loads((ROOT / f"outputs/{s}/tables/serie_massa_temporada.json")
                      .read_text(encoding="utf-8")) for s in ("jja", "djf")}
velha = {s: json.loads((ROOT / f"outputs/{s}/tables/serie_massa.json")
                       .read_text(encoding="utf-8")) for s in ("jja", "djf")}

# Epocas efetivas do binning ANTIGO, medidas numa amostra de 15 tiles por
# estacao (diagnostico, nao produto).
EPOCA_ANTIGA = {
    "jja": [2019.531, 2020.546, 2021.520, 2022.504, 2023.578, 2024.578, 2025.521],
    "djf": [2019.364, 2020.367, 2021.569, 2022.557, 2023.280, 2024.310, 2025.405],
}

fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6), facecolor="white")
fig.subplots_adjust(left=0.07, right=0.985, top=0.80, bottom=0.235, wspace=0.24)

# ------------------------------------------------------------------- (a)
ax = axes[0]
for s, cor, mk, nome in (("jja", COR_J, "o", "Inverno (JJA)"),
                         ("djf", COR_D, "^", "Verão (DJF)")):
    d = pd.DataFrame(nova[s]["serie"])
    ax.plot(d.epoca_media, d.massa_acumulada_Gt, color=cor, lw=1.6, marker=mk,
            ms=7, mfc="white", mew=1.6, zorder=4,
            label="%s, binning corrigido: %+.1f Gt/ano"
                  % (nome, nova[s]["taxa_Gt_por_ano_ajuste_linear"]))
dv = pd.DataFrame(velha["djf"]["serie"])
ax.plot(EPOCA_ANTIGA["djf"], dv.massa_acumulada_Gt, color=COR_OLD, lw=1.3,
        ls="--", marker="s", ms=6, mfc="white", mew=1.2, zorder=3,
        label="Verão, binning antigo (floor do ano): total %+.1f Gt"
              % dv.massa_acumulada_Gt.iloc[-1])
ax.axhline(0, color="0.6", lw=0.8)
ax.set_xlabel("Época média real da temporada (ano decimal)")
ax.set_ylabel("Massa acumulada (Gt)")
ax.set_title("(a) Série de massa contra a época real", fontsize=11, loc="left")
ax.grid(axis="y", color="0.92", lw=0.7)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8.6, loc="lower left")

# ------------------------------------------------------------------- (b)
ax = axes[1]
passos = {
    "Inverno\n(antigo = corrigido)": np.diff(EPOCA_ANTIGA["jja"]),
    "Verão\nbinning antigo": np.diff(EPOCA_ANTIGA["djf"]),
    "Verão\nbinning corrigido": np.diff(
        pd.DataFrame(nova["djf"]["serie"]).epoca_media.to_numpy()),
}
pos, cores = [], [COR_J, COR_OLD, COR_D]
for i, (nome, v) in enumerate(passos.items()):
    jitter = (np.arange(len(v)) - (len(v) - 1) / 2) * 0.055
    ax.plot(np.full(len(v), i) + jitter, v, "o", ms=6.5, color=cores[i],
            mec="white", mew=0.8, zorder=4)
    ax.plot([i - 0.28, i + 0.28], [v.mean()] * 2, color=cores[i], lw=2.2, zorder=3)
    pos.append(nome)
ax.axhline(1.0, color="#8B1E3F", ls="--", lw=1.2, zorder=2,
           label="passo nominal de 1 ano")
ax.set_xticks(range(len(pos)))
ax.set_xticklabels(pos, fontsize=9)
ax.set_ylabel("Passo entre caixas consecutivas (anos)")
ax.set_title("(b) O binning antigo desalinhava as épocas do verão",
             fontsize=11, loc="left")
ax.grid(axis="y", color="0.92", lw=0.7)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8.6, loc="lower right")
ax.set_ylim(0.6, 1.32)

fig.suptitle("Correção do binning sazonal — massa acumulada no ASE, ICESat-2 ATL06\n"
             "DJF passa a ser dez(Y) + jan/fev(Y+1) como uma temporada, rotulada pelo ano final",
             fontsize=12, y=0.965)
fig.text(0.5, 0.015,
         "Com o binning antigo (ano = parte inteira de t_year), uma temporada DJF era partida em duas caixas e a fração de dezembro variava de 24,6%% a 55,5%% entre anos, "
         "deslocando a época efetiva em até 0,29 ano.\n"
         "Totais: inverno %+.1f Gt (inalterado, o binning de JJA já estava correto); verão %+.1f Gt no binning antigo contra %+.1f Gt no corrigido (até a última temporada completa: %+.1f Gt).\n"
         "Épocas do binning antigo medidas numa amostra de 15 tiles por estação. Comparar totais acumulados continua sendo inadequado: use a taxa, como na figura comparacao_tendencia_nos_comuns."
         % (pd.DataFrame(nova["jja"]["serie"]).massa_acumulada_Gt.iloc[-1],
            dv.massa_acumulada_Gt.iloc[-1],
            pd.DataFrame(nova["djf"]["serie"]).massa_acumulada_Gt.iloc[-1],
            pd.DataFrame(nova["djf"]["serie"]).massa_acumulada_Gt.iloc[-2]),
         ha="center", va="bottom", fontsize=8.3, color="0.30")
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"correcao_binning_sazonal.{ext}",
                dpi=300 if ext == "png" else None, facecolor="white")

for s in ("jja", "djf"):
    d = pd.DataFrame(nova[s]["serie"])
    print("==", s, "| nos", nova[s]["n_nos_completos"], "| span %.2f ano"
          % nova[s]["span_epocas_anos"], "| taxa %+.1f Gt/ano"
          % nova[s]["taxa_Gt_por_ano_ajuste_linear"])
    print(d[["rotulo", "epoca_media", "dh_medio_m", "massa_acumulada_Gt", "n_nos"]]
          .to_string(index=False, float_format=lambda v: "%.3f" % v))
