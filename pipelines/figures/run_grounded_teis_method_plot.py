# -*- coding: utf-8 -*-
"""Anomalia de massa por temporada - estilo de figura de artigo (Nature).

Convencoes aplicadas:
  * largura exata de coluna dupla da Nature: 183 mm (7,20 in);
  * tipografia sans-serif (Helvetica/Arial) a 7 pt, rotulos de eixo a 7 pt,
    ticks a 6,5 pt - o minimo da Nature e ~5 pt depois da reducao;
  * linhas de eixo e ticks a 0,5 pt, ticks para fora, sem grade;
  * molduras superior e direita removidas;
  * SEM titulo e SEM rodape dentro da figura: em artigo isso vai na legenda,
    que e gravada em anomalia_massa_metodo_TEIS_legenda.md;
  * paleta Okabe-Ito (segura para daltonismo);
  * saida vetorial (PDF) alem do PNG a 600 dpi.

O eixo X continua categorico por temporada e a curva entre pontos continua
PCHIP (guia visual, monotonica por partes).
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

ROOT = Path(r"D:/WILLIAM/ARTIGO PIBIC ANTARTICA")

# --dir aponta para a pasta do produto (saida da serie). O padrao preserva o
# comportamento historico do script.
_ap = argparse.ArgumentParser(description="Figura da anomalia de massa sazonal.")
_ap.add_argument("--dir", default=None, help="pasta do produto da serie")
_ap.add_argument("--nome", default="anomalia_massa_metodo_TEIS",
                 help="prefixo dos arquivos de saida")
_ap.add_argument("--titulo", default="núcleo de afinamento da Geleira Thwaites",
                 help="complemento do titulo em negrito da legenda")
_ap.add_argument("--nota", default="",
                 help="frase extra da legenda, especifica da celula")
_args = _ap.parse_args()
OUT = (Path(_args.dir) if _args.dir else
       ROOT / "outputs" / "grounded_H_timeseries" / "nucleo_metodo_TEIS")
NOME = _args.nome
print("produto:", OUT)

MM = 1.0 / 25.4
LARGURA = 183 * MM          # coluna dupla Nature
ALTURA = 78 * MM
AZUL, LARANJA = "#0072B2", "#D55E00"   # Okabe-Ito
CINZA, PRETO = "#808080", "#1A1A1A"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 0.75,
    "legend.frameon": False,
    "pdf.fonttype": 42,     # TrueType: texto editavel no PDF, exigencia comum
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

ser = (pd.read_csv(OUT / "serie_metodo_TEIS.csv")
       .sort_values("t_center_year").reset_index(drop=True))
rep = json.loads((OUT / "relatorio_metodo_TEIS.json").read_text(encoding="utf-8"))
met = rep["tendencia"]
GTM = rep["Gt_por_metro"]
h0 = float(ser.loc[ser.window == "JJA 2019", "h_ice_res"].iloc[0])


SIMBOLO = rep.get("grandeza_simbolo", "h − FAC")
EIXO_DIR = ("Δ($h$ − FAC) (m)" if "FAC" in SIMBOLO else "Δ$h$ (m)")
TAXA = ("$d$($h$ − FAC)/$dt$" if "FAC" in SIMBOLO else "$dh$/$dt$")


def rotulo(w):
    return ("JJA\n%s" % w[4:]) if w.startswith("JJA") else ("DJF\n%s/%s" % (w[6:8], w[-2:]))


# O eixo guarda UMA posicao por temporada-alvo, mesmo quando a temporada nao
# tem ponto. Sem isso uma janela ausente encurta o eixo e um intervalo de um
# ano ocupa a mesma largura de um de meio ano: a curva ganha um degrau que nao
# existe no dado. A posicao vazia deixa a lacuna visivel.
JANELAS = list(rep["janelas"])
POS = {w: i for i, w in enumerate(JANELAS)}
x = ser.window.map(POS).to_numpy(float)
ausentes = [w for w in JANELAS if w not in set(ser.window)]
if ausentes:
    print("temporadas sem observacao (posicao reservada no eixo):", ausentes)
is_jja = ser.season.eq("JJA").to_numpy()

fig, ax = plt.subplots(figsize=(LARGURA, ALTURA), facecolor="white")
fig.subplots_adjust(left=0.075, right=0.915, top=0.965, bottom=0.235)

# guia PCHIP entre as temporadas
xs = np.linspace(x.min(), x.max(), 400)
ys = PchipInterpolator(x, ser.dM_Gt.to_numpy(float))(xs)
ax.plot(xs, ys, color=CINZA, lw=0.6, zorder=2)

# tendencia Theil-Sen, ajustada no tempo real e avaliada em cada temporada
t = ser.t_center_year.to_numpy(float)
trend = (met["intercept_at_mean_time"]
         + met["slope_m_per_year"] * (t - met["t_reference"]) - h0) * GTM
ax.plot(x, trend, ls=(0, (4, 2)), lw=0.7, color=PRETO, zorder=3,
        label="Tendência (Theil–Sen)")

for season, marker, cor, nome in (("JJA", "o", AZUL, "Inverno (JJA)"),
                                  ("DJF", "^", LARANJA, "Verão (DJF)")):
    m = ser.season.eq(season).to_numpy()
    ax.errorbar(x[m], ser.dM_Gt[m], yerr=ser.sigma_dM_Gt[m], fmt=marker, ms=3.6,
                mfc=cor, mec="white", mew=0.4, color=cor, ecolor=cor,
                elinewidth=0.6, capsize=1.4, capthick=0.6, linestyle="none",
                label=nome, zorder=4)


def menos(txt: str) -> str:
    """Sinal de menos tipografico (U+2212), nao hifen."""
    return txt.replace("-", "−")


import matplotlib.patheffects as _pe   # halo: o rotulo cruza a linha de tendencia


def casas(sigma, minimo=3):
    """Casas decimais que mostram o primeiro digito significativo da incerteza.

    Com 3 casas fixas, uma celula de sinal pequeno (massas de ~0,005 Gt)
    imprimia "−0.005" em tudo e "± 0.000" na taxa: o numero parecia exato.
    """
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        return minimo
    return max(minimo, int(-np.floor(np.log10(sigma))) + 1)


_sig_pos = ser.sigma_dM_Gt[ser.sigma_dM_Gt > 0]
ND_PONTO = casas(np.median(_sig_pos)) if len(_sig_pos) else 3

def rotulo_massa(v, janela):
    """Texto do valor sobre o ponto.

    * A temporada de referencia e zero por definicao: rotula-se "0 (ref.)",
      para nao parecer uma medida que deu exatamente zero.
    * Um valor medido que arredonda para zero NAO vira "+0.0000": mostra-se o
      primeiro algarismo significativo, com o sinal real. Na celula do interior,
      JJA 2020 ficou 1 mm abaixo de JJA 2019 (-0.00002 Gt, com sigma de 77 mm):
      "+0.0000" trocava o sinal e sugeria um retorno exato a origem.
    """
    if janela == "JJA 2019":
        return "0 (ref.)"
    v = float(v)
    if round(v, ND_PONTO) == 0 and v != 0:
        nd = int(-np.floor(np.log10(abs(v))))
        return menos(("%+." + str(nd) + "f Gt") % v)
    return menos(("%+." + str(ND_PONTO) + "f Gt") % (round(v, ND_PONTO) + 0.0))


# Valor de cada ponto (anomalia acumulada desde JJA 2019) acima do ponto.
for xi, v, e, jja, w in zip(x, ser.dM_Gt, ser.sigma_dM_Gt, is_jja, ser.window):
    ax.annotate(rotulo_massa(v, w),
                (xi, v + e), textcoords="offset points",
                path_effects=[_pe.withStroke(linewidth=1.6, foreground="white")],
                xytext=(0, 3.5), ha="center", va="bottom", fontsize=6,
                color=AZUL if jja else LARANJA, zorder=5)

ax.set_xticks(np.arange(len(JANELAS), dtype=float))
ax.set_xticklabels([rotulo(w) for w in JANELAS])
CINZA_AUSENTE = "#9E9E9E"
for tick, w in zip(ax.get_xticklabels(), JANELAS):
    if w in ausentes:
        tick.set_color(CINZA_AUSENTE)          # sem observacao utilizavel
    else:
        tick.set_color(AZUL if w.startswith("JJA") else LARANJA)
ax.set_xlim(-0.5, len(JANELAS) - 0.5)
ax.set_xlabel("Temporada", labelpad=2)
ax.set_ylabel("Anomalia de massa (Gt)", labelpad=3)

sec = ax.secondary_yaxis("right", functions=(lambda mm: mm / GTM, lambda hh: hh * GTM))
sec.set_ylabel(EIXO_DIR, labelpad=3)
sec.spines["right"].set_linewidth(0.5)
sec.tick_params(width=0.5, size=2.5)

lo = float((ser.dM_Gt - ser.sigma_dM_Gt).min())
hi = float((ser.dM_Gt + ser.sigma_dM_Gt).max())
ax.set_ylim(lo - 0.07 * (hi - lo), hi + 0.20 * (hi - lo))
ax.spines[["top"]].set_visible(False)

s, ss = met["slope_m_per_year"], met["slope_sigma_m_per_year"]
_nd = str(casas(ss * GTM))
# A anotacao vai para o canto vazio: numa serie que desce ele e o inferior
# esquerdo; numa que sobe, ali ficam os primeiros pontos e o texto os cobre.
_y_txt, _va_txt = (0.97, "top") if s > 0 else (0.06, "bottom")
ax.text(0.015, _y_txt,
        menos((TAXA + " = %+.2f ± %.2f m ano$^{-1}$\n"
               "$dM$/$dt$ = %+." + _nd + "f ± %." + _nd + "f Gt ano$^{-1}$")
              % (s, ss, s * GTM, ss * GTM)).replace("^{−", "^{-"),
        transform=ax.transAxes, va=_va_txt, ha="left", fontsize=6.5,
        color=PRETO, linespacing=1.35)
# Numa serie crescente o canto superior direito e onde terminam os pontos.
_leg_loc, _leg_anchor = (("lower right", (0.995, 0.02)) if s > 0
                         else ("upper right", (0.995, 0.99)))
ax.legend(loc=_leg_loc, bbox_to_anchor=_leg_anchor, handlelength=1.6,
          handletextpad=0.5, borderaxespad=0.0, labelspacing=0.35)

fig.savefig(OUT / f"{NOME}.pdf", facecolor="white")
fig.savefig(OUT / f"{NOME}.png", dpi=600, facecolor="white")

# ----------------------------------------------------------------- legenda
c = rep["celula"]
dt = np.diff(t)
# --- valores que dependem da celula, medidos aqui -------------------------
# Antes, a distancia a linha de aterramento e a comparacao com firn eram
# numeros fixos no texto, da celula original. Gerar a legenda para outra
# celula reproduzia esses numeros errados.
try:
    import geopandas as _gpd
    from shapely.geometry import Point as _Point
    from shapely.ops import unary_union as _uu
    # Linha OBSERVADA (InSAR, 2015 em diante). O limite aterrado derivado do
    # BedMachine inclui a borda do recorte como fronteira do poligono, e para
    # celulas do interior devolvia a distancia ate essa borda.
    _gl = _uu(_gpd.read_file(
        ROOT / "data/grounding/processed/InSAR_GL_ASE_v02.1.gpkg"
    ).query("Year >= 2015").geometry.tolist())
    DIST_GL_KM = "%.1f km" % (_Point(c["x"], c["y"]).distance(_gl) / 1000.0)
except Exception as _e:                       # sem o vetor em disco
    DIST_GL_KM = "[distancia a GL nao medida]"
    print("aviso: distancia a GL nao calculada (%s)" % _e)

_cmp = rep.get("comparacao_com_firn")
FRASE_FIRN = (" " + _cmp) if isinstance(_cmp, str) else (
    " Para esta célula a variante com desconto de firn não foi calculada, de"
    " modo que o custo dessa escolha aqui não está quantificado.")

# Granulos admitidos sem verificacao do pareamento com o sidecar de QC: se
# houver, a legenda tem de dizer, porque muda o que a barra de erro cobre.
_aud = rep.get("audit_qc", {})
_gr = _aud.get("granulos_join_nao_verificado_aceitos") or []
FRASE_JOIN = ""
if rep.get("join_nao_verificado_aceito") and _gr:
    _lin = _aud.get("linhas_de_join_nao_verificado", 0)
    _tot = _aud.get("observacoes_finais", 0) or 1
    FRASE_JOIN = (
        f" {len(_gr)} grânulo(s), com {_lin:,} segmentos ({100*_lin/_tot:.1f} % do"
        " total), foram admitidos apesar de o pareamento posicional com o"
        " sidecar de controle de qualidade não ter sido verificado: neles o"
        " teste por linha aprova o deslocamento zero em cinco dos seis feixes,"
        " e o critério exige unanimidade. Os testes estruturais (contagem,"
        " monotonicidade do segment_id, intervalo tempo-segmento) passam em"
        " todos.").replace(",", ".")

FRASE_NOTA = (" " + _args.nota.strip()) if _args.nota.strip() else ""

FRASE_LACUNA = ""
if ausentes:
    FRASE_LACUNA = (
        " A(s) temporada(s) " + ", ".join(ausentes) + " não aparece(m): a única"
        " passagem disponível sobre a célula nesse período está em grânulo cujo"
        " vínculo com o controle de qualidade não pôde ser verificado, e a"
        " posição correspondente fica vazia no eixo.")

legenda = f"""# Legenda da figura (texto para o artigo)

**Anomalia de massa por temporada num ponto euleriano fixo do gelo aterrado,
{_args.titulo}.** Célula fixa de 5 km centrada em
({c['x']:.0f}, {c['y']:.0f}) m (EPSG:3031; {abs(c['lon']):.2f}° W, {abs(c['lat']):.2f}° S),
a {DIST_GL_KM} da linha de aterramento observada por InSAR (MEaSUREs v2.1,
2015–2025), sobre gelo aterrado (BedMachine
`mask = 2`) com buffers de 5 km da linha de aterramento e 2 km da costa. Cada ponto é
uma temporada austral — inverno (JJA, azul) ou verão (DJF, laranja) — estimada a partir
de {int(ser.N_obs.sum()):,} segmentos ICESat-2 ATL06 v007 entre {int(t.min())} e
{int(t.max())}, sem interpolação temporal.{FRASE_LACUNA}{FRASE_JOIN}{FRASE_NOTA} A anomalia é referida ao inverno de
2019. O eixo direito traz a mesma quantidade em metros de elevação de superfície, Δ*h*,
com massa = ρ_i·*A*·Δ*h*, ρ_i = 917 kg m⁻³ e *A* = 25 km² (1 m ≡ {GTM:.4f} Gt).

A elevação de cada temporada vem de um ajuste conjunto com um intercepto por janela e um
plano local comum, resolvido por mínimos quadrados robustos (Huber), com pesos
balanceados por bloco orbital (RGT, ciclo), sobre *h* − REMA. O conteúdo de ar do firn
**não** é subtraído: o recorte do IMAU-FDM FDMv12A termina em 2023,98 e a extrapolação é
proibida, de modo que descontá-lo tornaria a série heterogênea.{FRASE_FIRN} As barras de erro são
±1 s.d., iguais ao maior valor entre o erro-padrão robusto a agrupamento por bloco e
RMSE/√N_blocos. A linha tracejada é a
tendência de Theil–Sen ajustada no tempo real e avaliada na posição de cada temporada;
a curva cinza é uma interpolação PCHIP entre as temporadas, apenas como guia visual, e
não representa observações. As temporadas estão igualmente espaçadas no eixo, mas os
intervalos reais variam de {dt.min():.2f} a {dt.max():.2f} ano, de modo que a inclinação
aparente não corresponde a taxa por ano; as taxas citadas vêm do tempo real. O sinal é
euleriano, num ponto fixo, e mistura adelgaçamento dinâmico com balanço de superfície.
O soerguimento glacioisostático no local (Caron et al., 2018) é
{rep['gia_mm_ano']:+.1f} mm ano⁻¹, ou {abs(rep['gia_mm_ano']/1000/s)*100:.1f} % da taxa
observada.

---

## Especificações de estilo aplicadas

| Item | Valor |
|---|---|
| Largura | 183 mm (coluna dupla) |
| Altura | 78 mm |
| Fonte | Helvetica/Arial, 7 pt (ticks 6,5 pt) |
| Linhas de eixo e ticks | 0,5 pt, para fora |
| Grade | ausente |
| Molduras | apenas esquerda, direita e inferior |
| Paleta | Okabe-Ito ({AZUL} e {LARANJA}), segura para daltonismo |
| Saída | PDF vetorial (fontes TrueType embutidas) e PNG a 600 dpi |
"""
(OUT / f"{NOME}_legenda.md").write_text(legenda, encoding="utf-8")
print("ok")
