"""Métodos de reamostragem agrupada para séries altimétricas com correlação intra-passagem.

Fundamentação teórica e referências primárias
---------------------------------------------
1. **Jackknife Agrupado por Ano (*Leave-One-Year-Out*; Tukey, 1958; Efron & Stein, 1981):**
   Em altimetria laser, um nó espacial reúne $\\sim 10^3\\text{--}10^5$ segmentos distribuídos em
   $k \\approx 7$ épocas anuais. Os resíduos dentro de um mesmo ano compartilham fase de maré,
   apontamento orbital e estado da superfície de neve, violando a hipótese de independência do
   erro formal de mínimos quadrados ($\\sigma_{\\text{OLS}} \\propto N_{\\text{seg}}^{-1/2}$).
   Ao tratar o ano $y \\in \\{1, \\dots, k\\}$ como a unidade amostral independente e recalcular o
   estimador $\\hat{\\theta}_{(-y)}$ omitindo um ano inteiro por vez, a variância de jackknife é:
   $$\\operatorname{Var}_{\\text{JK}}(\\hat{\\theta}) = \\frac{k - 1}{k} \\sum_{y=1}^{k} \\left(\\hat{\\theta}_{(-y)} - \\bar{\\theta}_{(\\cdot)}\\right)^2$$
   Teoricamente (Efron & Stein, 1981), essa expressão é um limitante superior conservador e
   não-paramétrico para a variância de qualquer funcional simétrico sob amostragem em clusters.

2. **Delete-a-Group Jackknife sobre Passagens Orbitais (Kott, 2001):**
   Para séries temporais de plataformas de gelo, a unidade primária de amostragem (PSU) é a
   passagem orbital individual ``(rgt, cycle)``. As passagens distintas são particionadas
   aleatoriamente em $G = 20$ grupos disjuntos e o modelo de painel é reajustado omitindo um
   grupo por vez:
   $$\\operatorname{Var}_{\\text{DAGJK}}(\\hat{\\mathbf{s}}) = \\frac{G - 1}{G} \\sum_{g=1}^{G} \\left(\\hat{\\mathbf{s}}_{(-g)} - \\bar{\\mathbf{s}}\\right)^2$$

3. **Block Bootstrap Exato por Equações Normais Pré-Acumuladas (Carlstein, 1986):**
   Para avaliar a estabilidade de termos de ordem superior (ex.: aceleração $\\frac{1}{2}\\beta_2 \\Delta t^2$),
   a reamostragem com reposição é feita sobre blocos anuais inteiros $y^*_1, \\dots, y^*_k$.
   Como as equações normais ponderadas são aditivas sobre blocos disjuntos,
   $$\\mathbf{S}_y = \\mathbf{A}_y^T \\mathbf{W}_y \\mathbf{A}_y, \\qquad \\mathbf{b}_y = \\mathbf{A}_y^T \\mathbf{W}_y \\mathbf{h}_y,$$
   cada réplica bootstrap resolve apenas o sistema $p \\times p$:
   $$\\left(\\sum_{j=1}^{k} \\mathbf{S}_{y^*_j}\\right) \\hat{\\boldsymbol{\\beta}}^* = \\sum_{j=1}^{k} \\mathbf{b}_{y^*_j},$$
   produzindo exatamente a mesma solução do reajuste completo nos pontos reamostrados com
   custo computacional $O(k p^3)$ por réplica em vez de $O(N_{\\text{seg}} p^2)$.

Referências
-----------
- Tukey, J. W. (1958). Bias and confidence in not-quite large samples. *Ann. Math. Stat.*, 29, 614.
- Efron, B., & Stein, C. (1981). The jackknife estimate of variance. *Annals of Statistics*, 9(3), 586-596.
- Kott, P. S. (2001). The delete-a-group jackknife. *Journal of Official Statistics*, 17(4), 521-526.
- Carlstein, E. (1986). The use of subseries values for estimating the variance of a general
  statistic from a stationary sequence. *Annals of Statistics*, 14(3), 1171-1179.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def leave_one_year_out_jackknife(
    years: np.ndarray,
    estimator_fn: Callable[[np.ndarray], float],
    min_years: int = 3,
) -> tuple[float, float, int]:
    """Calcula a incerteza por jackknife deixando um ano inteiro de fora por vez.

    Parâmetros
    ----------
    years : array-like
        Ano inteiro (ou época discreta) de cada observação.
    estimator_fn : callable
        Função que recebe uma máscara booleana ``keep`` e devolve a estimativa escalar.
    min_years : int
        Número mínimo de épocas válidas exigido.

    Retorna
    -------
    theta_full, se_jackknife, n_years_valid
    """
    yr = np.floor(np.asarray(years, dtype=float)).astype(int)
    uniq = np.unique(yr)
    theta_full = float(estimator_fn(np.ones(yr.size, dtype=bool)))
    if not np.isfinite(theta_full) or uniq.size < min_years:
        return theta_full, np.nan, int(uniq.size)

    replicates = []
    for y in uniq:
        val = float(estimator_fn(yr != y))
        if np.isfinite(val):
            replicates.append(val)

    k = len(replicates)
    if k < min_years:
        return theta_full, np.nan, k
    rep = np.asarray(replicates, dtype=float)
    var_jk = (k - 1.0) / k * float(np.sum((rep - rep.mean()) ** 2))
    return theta_full, float(np.sqrt(max(var_jk, 0.0))), k


def delete_group_jackknife(
    cluster_ids: np.ndarray,
    estimator_fn: Callable[[np.ndarray], np.ndarray],
    n_groups: int = 20,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Executa Delete-a-Group Jackknife (Kott, 2001) agrupando clusters primários (ex.: passagens).

    Retorna ``(replicates, se_vector)``.
    """
    cid = np.asarray(cluster_ids)
    uniq = np.unique(cid)
    rng = np.random.default_rng(seed)
    group_of_uniq = rng.integers(0, n_groups, size=uniq.size)
    obs_group = group_of_uniq[np.searchsorted(uniq, cid)]

    reps = [np.asarray(estimator_fn(obs_group != g), dtype=float) for g in range(n_groups)]
    jk = np.stack(reps, axis=0)
    mean_jk = np.nanmean(jk, axis=0, keepdims=True)
    var_jk = (n_groups - 1.0) / n_groups * np.nansum((jk - mean_jk) ** 2, axis=0)
    return jk, np.sqrt(np.maximum(var_jk, 0.0))


def block_bootstrap_normal_eq(
    A: np.ndarray,
    y: np.ndarray,
    block_ids: np.ndarray,
    param_index: int,
    weights: np.ndarray | None = None,
    n_boot: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """Block bootstrap sobre blocos temporais usando soma de equações normais pré-acumuladas."""
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float)
    blocks = np.asarray(block_ids)
    uniq = np.unique(blocks)
    if uniq.size < 3:
        return np.empty(0, dtype=float)

    if weights is None:
        w = np.ones(y.size, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        w = w / (np.mean(w) + 1e-30)

    sw = np.sqrt(w)
    Aw = A * sw[:, None]
    yw = y * sw

    S_b, rhs_b, n_b = {}, {}, {}
    for b in uniq:
        m = blocks == b
        Ab = Aw[m]
        S_b[b] = Ab.T @ Ab
        rhs_b[b] = Ab.T @ yw[m]
        n_b[b] = int(m.sum())

    p = A.shape[1]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        if np.unique(pick).size < 3 or sum(n_b[b] for b in pick) < p + 3:
            continue
        S = sum(S_b[b] for b in pick)
        rhs = sum(rhs_b[b] for b in pick)
        try:
            coef = np.linalg.solve(S, rhs)
        except np.linalg.LinAlgError:
            continue
        val = float(coef[param_index])
        if np.isfinite(val):
            draws.append(val)
    return np.asarray(draws, dtype=float)
