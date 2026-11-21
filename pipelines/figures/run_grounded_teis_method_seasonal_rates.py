# -*- coding: utf-8 -*-
"""Taxa e massa por INTERVALO entre temporadas consecutivas (gelo aterrado).

Entrada: a serie por janela ja ajustada em
outputs/grounded_H_timeseries/nucleo_metodo_TEIS/serie_metodo_TEIS.csv
(metodologia da serie TEIS; grandeza h - FAC).

A taxa existe ENTRE janelas, nao em uma janela: 9 temporadas -> 8 intervalos.

Duas barras de erro, de proposito:
  sigma_TEIS  = hypot(sigma_h_ice_i, sigma_h_ice_j)  -> convencao da TEIS,
                propaga a representatividade do FAC (~1,05 m) duas vezes;
  sigma_ajuste = hypot(sigma_fit_i, sigma_fit_j)     -> so o erro do ajuste.
A dispersao do FAC entre vizinhos do modelo e um deslocamento espacial quase
constante no tempo: ele se cancela na diferenca entre duas janelas. A primeira
barra e conservadora; a segunda e a que mede o que muda de uma janela para a
outra. Nenhuma das duas e "a certa" sem quantificar a parte comum do FAC.
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
OUT = ROOT / "outputs" / "grounded_H_timeseries" / "nucleo_metodo_TEIS"

ser = (pd.read_csv(OUT / "serie_metodo_TEIS.csv")
       .sort_values("t_center_year").reset_index(drop=True))
rep = json.loads((OUT / "relatorio_metodo_TEIS.json").read_text(encoding="utf-8"))
met = rep["tendencia"]
GTM = rep["Gt_por_metro"]
COR_PERDA, COR_GANHO = "#2A6F97", "#B36B00"


def curto(w):
    return ("Inv %s" % w[-2:]) if w.startswith("JJA") else ("Ver %s/%s" % (w[6:8], w[-2:]))


h = ser.h_ice_res.to_numpy(float)
sig = ser.sigma_h_ice.to_numpy(float)
sfit = ser.sigma_fit.to_numpy(float)
t = ser.t_center_year.to_numpy(float)
fac = ser.fac_window_median.to_numpy(float)

tr = pd.DataFrame({
    "de": ser.window.values[:-1], "para": ser.window.values[1:],
    "dt_anos": np.diff(t),
    "dh_ice_m": np.diff(h),
    "sigma_dh_TEIS_m": np.hypot(sig[1:], sig[:-1]),
    "sigma_dh_ajuste_m": np.hypot(sfit[1:], sfit[:-1]),
    "dFAC_m": np.diff(fac),
})
tr["dhdt_m_ano"] = tr.dh_ice_m / tr.dt_anos
tr["sigma_dhdt_TEIS"] = tr.sigma_dh_TEIS_m / tr.dt_anos
tr["sigma_dhdt_ajuste"] = tr.sigma_dh_ajuste_m / tr.dt_anos
tr["dM_Gt"] = tr.dh_ice_m * GTM
tr["sigma_dM_TEIS_Gt"] = tr.sigma_dh_TEIS_m * GTM
tr["sigma_dM_ajuste_Gt"] = tr.sigma_dh_ajuste_m * GTM
tr["z_TEIS"] = tr.dh_ice_m.abs() / tr.sigma_dh_TEIS_m
tr["z_ajuste"] = tr.dh_ice_m.abs() / tr.sigma_dh_ajuste_m
tr["sentido"] = np.where(tr.de.str.startswith("JJA"), "inverno->verao", "verao->inverno")
tr["rotulo"] = [curto(a) + "\n\u2193\n" + curto(b) for a, b in zip(tr.de, tr.para)]
tr["balanco"] = np.where(tr.dM_Gt < 0, "perda", "ganho")

perdas = float(tr.loc[tr.dM_Gt < 0, "dM_Gt"].sum())
ganhos = float(tr.loc[tr.dM_Gt > 0, "dM_Gt"].sum())
para_verao = tr[tr.sentido == "inverno->verao"]
para_inverno = tr[tr.sentido == "verao->inverno"]
resumo = {
    "n_intervalos": int(len(tr)),
    "intervalos_com_perda": int((tr.dM_Gt < 0).sum()),
    "intervalos_com_ganho": int((tr.dM_Gt > 0).sum()),
    "massa_perdida_total_Gt": perdas,
    "massa_ganha_total_Gt": ganhos,
    "saldo_Gt": perdas + ganhos,
    "taxa_media_inverno_para_verao_m_ano": float(np.average(
        para_verao.dhdt_m_ano, weights=para_verao.dt_anos)),
    "taxa_media_verao_para_inverno_m_ano": float(np.average(
        para_inverno.dhdt_m_ano, weights=para_inverno.dt_anos)),
    "tendencia_global_m_ano": met["slope_m_per_year"],
    "intervalos_acima_de_1sigma_TEIS": int((tr.z_TEIS > 1).sum()),
    "intervalos_acima_de_1sigma_ajuste": int((tr.z_ajuste > 1).sum()),
    "dFAC_min_m": float(tr.dFAC_m.min()), "dFAC_max_m": float(tr.dFAC_m.max()),
}
tr.to_csv(OUT / "taxas_por_temporada.csv", index=False, encoding="utf-8")
(OUT / "taxas_por_temporada.json").write_text(
    json.dumps({"resumo": resumo, "intervalos": tr.to_dict(orient="records")},
               indent=1, ensure_ascii=False, default=str), encoding="utf-8")

x = np.arange(len(tr), dtype=float)
cores = [COR_PERDA if v < 0 else COR_GANHO for v in tr.dM_Gt]
fig, axes = plt.subplots(2, 1, figsize=(12.2, 9.4), facecolor="white", sharex=True)
fig.subplots_adjust(left=0.095, right=0.905, top=0.875, bottom=0.205, hspace=0.30)

# ------------------------------------------------------- (a) taxa por intervalo
ax = axes[0]
ax.bar(x, tr.dhdt_m_ano, color=cores, edgecolor="0.25", lw=0.6, width=0.62, zorder=3)
ax.errorbar(x, tr.dhdt_m_ano, yerr=tr.sigma_dhdt_TEIS, fmt="none", ecolor="0.45",
            elinewidth=1.0, capsize=4, zorder=4, label="±1σ convenção TEIS (com firn)")
ax.errorbar(x, tr.dhdt_m_ano, yerr=tr.sigma_dhdt_ajuste, fmt="none", ecolor="0.15",
            elinewidth=2.2, capsize=0, zorder=5, label="±1σ só do ajuste")
sl, ss = met["slope_m_per_year"], met["slope_sigma_m_per_year"]
ax.axhspan(sl - ss, sl + ss, color="#8B1E3F", alpha=0.10, zorder=1)
ax.axhline(sl, color="#8B1E3F", ls="--", lw=1.3, zorder=2,
           label="tendência 2019–2023: %+.2f ± %.2f m/ano" % (sl, ss))
ax.axhline(0, color="0.3", lw=0.9, zorder=2)
for xi, v, e in zip(x, tr.dhdt_m_ano, tr.sigma_dhdt_TEIS):
    ax.annotate("%+.2f" % v, (xi, v - e), textcoords="offset points", xytext=(0, -14),
                ha="center", va="top", fontsize=8.4, fontweight="bold")
ax.set_ylabel("$d(h-\\mathrm{FAC})/dt$ no intervalo (m ano$^{-1}$)", fontsize=10)
ax.set_title("(a) Taxa em cada intervalo entre temporadas consecutivas",
             fontsize=11, loc="left")
ax.grid(axis="y", color="0.92", lw=0.7, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
lim = float((tr.dhdt_m_ano.abs() + tr.sigma_dhdt_TEIS).max() * 1.45)
ax.set_ylim(-lim, lim * 0.55)
ax.legend(frameon=False, fontsize=8.4, loc="upper right", ncol=1)

# --------------------------------------------------- (b) massa por intervalo
ax = axes[1]
ax.bar(x, tr.dM_Gt, color=cores, edgecolor="0.25", lw=0.6, width=0.62, zorder=3)
ax.errorbar(x, tr.dM_Gt, yerr=tr.sigma_dM_TEIS_Gt, fmt="none", ecolor="0.45",
            elinewidth=1.0, capsize=4, zorder=4)
ax.errorbar(x, tr.dM_Gt, yerr=tr.sigma_dM_ajuste_Gt, fmt="none", ecolor="0.15",
            elinewidth=2.2, capsize=0, zorder=5)
ax.axhline(0, color="0.3", lw=0.9, zorder=2)
for xi, v, e in zip(x, tr.dM_Gt, tr.sigma_dM_TEIS_Gt):
    ax.annotate("%+.4f" % v, (xi, v - e), textcoords="offset points", xytext=(0, -14),
                ha="center", va="top", fontsize=8.4, fontweight="bold")
ax.set_ylabel("Massa ganha (+) ou perdida (−)\nno intervalo (Gt)", fontsize=10)
ax.set_title("(b) Massa em cada intervalo, e acumulada ao longo da série",
             fontsize=11, loc="left")
ax.grid(axis="y", color="0.92", lw=0.7, zorder=0)
ax.spines[["top"]].set_visible(False)
limb = float((tr.dM_Gt.abs() + tr.sigma_dM_TEIS_Gt).max() * 1.45)
ax.set_ylim(-limb, limb * 0.55)

ax2 = ax.twinx()
cum = np.concatenate([[0.0], np.cumsum(tr.dM_Gt.to_numpy())])
ax2.plot(np.concatenate([[-0.5], x]), cum, color="#3E4A59", lw=1.6, marker="o",
         ms=4.5, mfc="white", mew=1.2, zorder=6, label="massa acumulada")
ax2.set_ylabel("Massa acumulada (Gt)", fontsize=10, color="#3E4A59")
ax2.tick_params(axis="y", colors="#3E4A59")
ax2.spines[["top"]].set_visible(False)
ax2.legend(frameon=False, fontsize=8.6, loc="lower left")

ax.set_xticks(x)
ax.set_xticklabels(tr.rotulo, fontsize=8.2)
ax.set_xlim(-0.6, len(tr) - 0.4)
ax.set_xlabel("Intervalo entre temporadas", labelpad=6)

fig.suptitle("Taxa e massa por temporada — gelo aterrado, núcleo de afinamento de Thwaites\n"
             "célula de 5 km em (%.0f, %.0f) m EPSG:3031 · %.2f°W, %.2f°S · "
             "grandeza h − FAC, 9 temporadas → 8 intervalos"
             % (rep["celula"]["x"], rep["celula"]["y"],
                abs(rep["celula"]["lon"]), abs(rep["celula"]["lat"])),
             fontsize=12, y=0.975)
fig.text(0.5, 0.012,
         "Perda em todos os %d intervalos: %+.3f Gt somados; nenhum intervalo de ganho. "
         "Média ponderada: %+.2f m/ano de inverno para verão, %+.2f m/ano de verão para inverno.\n"
         "Barra fina cinza = ±1σ da convenção TEIS, que propaga a representatividade do FAC (~1,05 m por janela).\n"
         "Barra grossa escura = ±1σ só do ajuste: o termo do FAC é um deslocamento espacial quase constante, que se cancela na diferença entre janelas.\n"
         "Intervalos de 0,37 a 0,58 ano: dividir por um intervalo curto amplifica a taxa, então a barra (a) não deve ser lida como taxa anual sustentada.\n"
         "Em gelo aterrado a %.0f km da linha de aterramento, perda não é derretimento de superfície: é adelgaçamento dinâmico mais balanço de superfície, já sem o ar do firn."
         % (resumo["n_intervalos"], perdas,
            resumo["taxa_media_inverno_para_verao_m_ano"],
            resumo["taxa_media_verao_para_inverno_m_ano"],
            rep["celula"].get("dist_gl_m", 23409.4) / 1000.0),
         ha="center", va="bottom", fontsize=8.2, color="0.30")
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"taxas_por_temporada.{ext}", dpi=300 if ext == "png" else None,
                facecolor="white")
pd.set_option("display.width", 250)
print(tr[["de", "para", "dt_anos", "dh_ice_m", "sigma_dh_TEIS_m", "sigma_dh_ajuste_m",
          "dhdt_m_ano", "dM_Gt", "z_TEIS", "z_ajuste", "dFAC_m"]]
      .to_string(float_format=lambda v: "%+.4f" % v))
print(json.dumps(resumo, indent=1, ensure_ascii=False))
