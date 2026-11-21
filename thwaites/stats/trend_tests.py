"""Testes não-paramétricos de tendência monotônica e controle de multiplicidade espacial.

Fundamentação teórica e referências primárias
---------------------------------------------
1. **Teste de Mann-Kendall (Mann, 1945; Kendall, 1975):**
   Avalia a hipótese nula $H_0$ de ausência de tendência monotônica contra a alternativa
   bilateral $H_1$ a partir da estatística de soma de sinais sobre todos os pares ordenados
   no tempo $(t_i < t_j)$:
   $$S = \\sum_{i=1}^{n-1}\\sum_{j=i+1}^{n} \\operatorname{sgn}(y_j - y_i)$$
   Sob $H_0$ e observações independentes identicamente distribuídas, $\\mathbb{E}[S] = 0$ e
   a variância corrigida para grupos de valores empatados de tamanho $t_p$ é:
   $$\\operatorname{Var}(S) = \\frac{n(n-1)(2n+5) - \\sum_p t_p(t_p-1)(2t_p+5)}{18}$$
   Por ser baseado exclusivamente em postos relativos, o teste é invariante a transformações
   monotônicas e robusto à forte assimetria negativa observada nas taxas altimétricas
   costeiras ($\\text{skewness} \\approx -2{,}0$).

2. **Estimador de Declive de Theil-Sen (Theil, 1950; Sen, 1968):**
   Define a taxa de variação como a mediana das inclinações de todos os pares de épocas
   distintas:
   $$\\hat{\\beta}_{\\text{Sen}} = \\operatorname{mediana}_{i < j} \\left(\\frac{y_j - y_i}{t_j - t_i}\\right)$$
   Possui ponto de ruptura assintótico de $1 - 1/\\sqrt{2} \\approx 29{,}3\\%$ e eficiência
   assintótica de $3/\\pi \\approx 95{,}5\\%$ sob normalidade em relação ao estimador de
   mínimos quadrados ordinários.

3. **Mann-Kendall Sazonal (Hirsch, Slack & Smith, 1982):**
   Particiona a série em $M$ estratos sazonais (meses ou trimestres) e acumula a estatística
   intra-estrato $S_{\\text{tot}} = \\sum_{m=1}^{M} S_m$ com variância
   $\\operatorname{Var}(S_{\\text{tot}}) = \\sum_{m=1}^{M} \\operatorname{Var}(S_m)$,
   eliminando o ciclo sazonal sem comparar observações de fases distintas do ano.

4. **Controle da Taxa de Falsas Descobertas — FDR (Benjamini & Hochberg, 1995):**
   Para $N$ nós testados simultaneamente com p-valores ordenados $p_{(1)} \\le \\dots \\le p_{(N)}$,
   determina o maior índice $k$ tal que $p_{(k)} \\le \\frac{k}{N}\\alpha$. O procedimento
   controla $\\mathbb{E}[V / R] \\le \\alpha$ sob independência e sob dependência positiva
   por regressão (PRDS; Benjamini & Yekutieli, 2001), propriedade satisfeita por campos
   geofísicos com autocorrelação espacial positiva.

Referências
-----------
- Mann, H. B. (1945). Nonparametric tests against trend. *Econometrica*, 13(3), 245-259.
- Kendall, M. G. (1975). *Rank Correlation Methods* (4th ed.). Charles Griffin.
- Sen, P. K. (1968). Estimates of the regression coefficient based on Kendall's tau.
  *Journal of the American Statistical Association*, 63(324), 1379-1389.
- Hirsch, R. M., Slack, J. R., & Smith, R. A. (1982). Techniques of trend analysis for
  monthly water quality data. *Water Resources Research*, 18(1), 107-121.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical
  and powerful approach to multiple testing. *JRSS-B*, 57(1), 289-300.
- Benjamini, Y., & Yekutieli, D. (2001). The control of the false discovery rate in
  multiple testing under dependency. *Annals of Statistics*, 29(4), 1165-1188.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, norm, theilslopes


def mann_kendall_sen(years, values, alpha: float = 0.05) -> dict:
    """Executa o teste de Mann-Kendall e calcula o declive de Theil-Sen com IC de (1 - alpha)."""
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(years) & np.isfinite(values)
    years, values = years[ok], values[ok]
    order = np.argsort(years)
    years, values = years[order], values[order]
    n = len(values)
    if n < 3:
        return {
            "tau": np.nan, "p_value": np.nan, "sens_slope": np.nan,
            "sens_lo": np.nan, "sens_hi": np.nan, "intercept": np.nan,
            "trend": "insuficiente", "n": n,
        }

    tau, p = kendalltau(years, values)
    slope, intercept, lo, hi = theilslopes(values, years, alpha=1.0 - alpha)

    if np.isnan(p):
        trend = "insuficiente"
    elif p < alpha and slope < 0:
        trend = "decrescente"
    elif p < alpha and slope > 0:
        trend = "crescente"
    else:
        trend = "sem_tendencia"

    return {
        "tau": float(tau), "p_value": float(p), "sens_slope": float(slope),
        "sens_lo": float(lo), "sens_hi": float(hi), "intercept": float(intercept),
        "trend": trend, "n": int(n),
    }


def seasonal_mann_kendall_sen(years, months, values, alpha: float = 0.05) -> dict:
    """Executa o teste de Mann-Kendall sazonal (Hirsch, Slack & Smith, 1982) e declive de Sen."""
    years = np.asarray(years, dtype=float)
    months = np.asarray(months, dtype=int)
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(years) & np.isfinite(values) & (months >= 1) & (months <= 12)
    years, months, values = years[ok], months[ok], values[ok]
    if len(values) < 4:
        return {
            "tau": np.nan, "p_value": np.nan, "sens_slope": np.nan,
            "sens_lo": np.nan, "sens_hi": np.nan, "intercept": np.nan,
            "trend": "insuficiente", "n": len(values),
        }

    s_total = 0.0
    var_total = 0.0
    pair_count = 0
    n_used = 0
    slopes: list[float] = []

    for mon in range(1, 13):
        sel = months == mon
        yy, vv = years[sel], values[sel]
        order = np.argsort(yy)
        yy, vv = yy[order], vv[order]
        n = len(vv)
        if n < 2:
            continue
        n_used += n
        diff = vv[None, :] - vv[:, None]
        iu = np.triu_indices(n, k=1)
        d = diff[iu]
        s_total += float(np.sign(d).sum())
        pair_count += len(d)
        year_diff = (yy[None, :] - yy[:, None])[iu]
        pos = year_diff > 0
        slopes.extend((d[pos] / year_diff[pos]).tolist())
        _, ties = np.unique(vv, return_counts=True)
        var_total += (
            n * (n - 1) * (2 * n + 5)
            - np.sum(ties * (ties - 1) * (2 * ties + 5))
        ) / 18.0

    if pair_count == 0 or var_total <= 0 or not slopes:
        return {
            "tau": np.nan, "p_value": np.nan, "sens_slope": np.nan,
            "sens_lo": np.nan, "sens_hi": np.nan, "intercept": np.nan,
            "trend": "insuficiente", "n": n_used,
        }

    z = (
        (s_total - 1.0) / np.sqrt(var_total) if s_total > 0
        else (s_total + 1.0) / np.sqrt(var_total) if s_total < 0
        else 0.0
    )
    p = float(2.0 * norm.sf(abs(z)))
    slopes_arr = np.sort(np.asarray(slopes, dtype=float))
    slope = float(np.median(slopes_arr))
    c_alpha = norm.ppf(1.0 - alpha / 2.0) * np.sqrt(var_total)
    lo_idx = max(0, int(np.floor((len(slopes_arr) - c_alpha) / 2.0)))
    hi_idx = min(len(slopes_arr) - 1, int(np.ceil((len(slopes_arr) + c_alpha) / 2.0)))
    intercept = float(np.median(values - slope * years))
    trend = (
        "decrescente" if p < alpha and slope < 0
        else "crescente" if p < alpha and slope > 0
        else "sem_tendencia"
    )
    return {
        "tau": float(s_total / pair_count), "p_value": p,
        "sens_slope": slope, "sens_lo": float(slopes_arr[lo_idx]),
        "sens_hi": float(slopes_arr[hi_idx]), "intercept": intercept,
        "trend": trend, "n": int(n_used),
    }


def benjamini_hochberg_fdr(p_values, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Aplica o controle de FDR de Benjamini-Hochberg (1995) a um vetor de p-valores.

    Retorna ``(reject, p_adjusted)`` preservando a ordem original.
    """
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=float)
    order = np.argsort(p)
    ranks = np.arange(1, m + 1, dtype=float)
    p_sorted = p[order]
    q_sorted = np.minimum.accumulate((p_sorted * m / ranks)[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    p_adj = np.empty_like(p)
    p_adj[order] = q_sorted
    return p_adj <= alpha, p_adj
