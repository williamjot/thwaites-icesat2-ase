"""Modelos em painel de efeitos fixos, complemento de Schur e variância agrupada CR1.

Fundamentação teórica e referências primárias
---------------------------------------------
1. **Eliminação Analítica por Complemento de Schur / Teorema de Frisch-Waugh-Lovell
   (Frisch & Waugh, 1933; Lovell, 1963):**
   Para estimar simultaneamente parâmetros locais de topografia móvel $\\boldsymbol{\\alpha}_b$
   (em $N_m$ células materiais Lagrangianas $b$) e coeficientes globais/regionais $\\boldsymbol{\\beta}$
   (como a resposta à maré $\\beta_c$ em células Eulerianas ou efeitos de época $s_k$) no modelo linear
   $$\\mathbf{h} = \\mathbf{X}\\boldsymbol{\\alpha} + \\mathbf{Z}\\boldsymbol{\\beta} + \\boldsymbol{\\varepsilon},$$
   onde $\\mathbf{X}^T\\mathbf{X} = \\operatorname{diag}(\\mathbf{A}_1, \\dots, \\mathbf{A}_{N_m})$ é
   bloco-diagonal por célula material, a projeção ortogonal para fora do espaço-coluna de $\\mathbf{X}$
   reduz o sistema exatamente às equações normais condensadas:
   $$\\left(\\mathbf{Z}^T\\mathbf{Z} - \\sum_{b=1}^{N_m} \\mathbf{V}_b^T \\mathbf{A}_b^{-1} \\mathbf{V}_b\\right) \\hat{\\boldsymbol{\\beta}} = \\mathbf{Z}^T\\mathbf{h} - \\sum_{b=1}^{N_m} \\mathbf{V}_b^T \\mathbf{A}_b^{-1} \\mathbf{X}_b^T\\mathbf{h},$$
   com $\\mathbf{V}_b = \\mathbf{X}_b^T \\mathbf{Z}_b$. Pelo teorema de Frisch-Waugh-Lovell,
   $\\hat{\\boldsymbol{\\beta}}$ coincide numericamente com a solução conjunta completa sem
   sofrer o viés de atenuação de ajustes sequenciais (*backfitting*).

2. **Série Temporal em Painel por Efeitos Fixos (*Within Estimator*; Wooldridge, 2010):**
   Caso particular em que $\\mathbf{Z}$ é a matriz indicadora de épocas sazonais $k \\in \\{1, \\dots, K\\}$
   e $\\mathbf{X}_b$ modela o plano topográfico local $(1, \\Delta x, \\Delta y)$ de cada célula
   material $b$. A eliminação exata de $(a_b, g_{x,b}, g_{y,b})$ impede que variações na
   amostragem espacial entre estações contaminem a série temporal $s_k$ (eliminação do viés
   de composição amostral).

3. **Estimador Sanduíche Agrupado CR1 (Liang & Zeger, 1986; Cameron & Miller, 2015):**
   Para observações agrupadas em $G$ passagens orbitais independentes $g \\in \\{1, \\dots, G\\}$,
   com vetor escore projetado $\\tilde{\\mathbf{z}}_{g} = \\sum_{i \\in g} \\tilde{\\mathbf{z}}_i \\hat{e}_i$,
   o estimador de variância robusto à heterocedasticidade e à correlação intra-passagem é:
   $$\\widehat{\\operatorname{Var}}_{\\text{CR1}}(\\hat{\\boldsymbol{\\beta}}) = \\frac{G}{G - 1} \\mathbf{H}^{-1} \\left(\\sum_{g=1}^{G} \\tilde{\\mathbf{z}}_g \\tilde{\\mathbf{z}}_g^T\\right) \\mathbf{H}^{-1}$$

Referências
-----------
- Frisch, R., & Waugh, F. V. (1933). Partial time regressions as compared with individual trends.
  *Econometrica*, 1(4), 387-401.
- Lovell, M. C. (1963). Seasonal adjustment of economic time series and multiple regression analysis.
  *Journal of the American Statistical Association*, 58(304), 993-1010.
- Liang, K.-Y., & Zeger, S. L. (1986). Longitudinal data analysis using generalized linear models.
  *Biometrika*, 73(1), 13-22.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press.
- Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to cluster-robust inference.
  *Journal of Human Resources*, 50(2), 317-372.
"""

from __future__ import annotations

import numpy as np


def fixed_effects_panel_series(
    px: np.ndarray,
    py: np.ndarray,
    values: np.ndarray,
    epoch_idx: np.ndarray,
    n_epochs: int,
    cell_size_m: float = 500.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estima a série temporal $s_k$ por efeitos fixos eliminando o plano local de cada célula.

    Retorna ``(s_centered, n_obs_per_epoch)``.
    """
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    H = np.asarray(values, dtype=float)
    k = np.asarray(epoch_idx, dtype=int)

    key = np.c_[np.floor(px / cell_size_m), np.floor(py / cell_size_m)].astype(np.int64)
    _, gb = np.unique(key, axis=0, return_inverse=True)
    gb = gb.ravel()
    nb = int(gb.max()) + 1

    cx = (np.floor(px / cell_size_m) + 0.5) * cell_size_m
    cy = (np.floor(py / cell_size_m) + 0.5) * cell_size_m
    X = np.c_[np.ones_like(H), (px - cx) / 100.0, (py - cy) / 100.0]
    p = 3

    A = np.zeros((nb, p, p))
    for a in range(p):
        for b in range(a, p):
            v = np.bincount(gb, weights=X[:, a] * X[:, b], minlength=nb)
            A[:, a, b] = v
            A[:, b, a] = v

    cond = np.linalg.cond(A)
    okb = np.isfinite(cond) & (cond < 1e8)
    Ai = np.zeros_like(A)
    Ai[okb] = np.linalg.inv(A[okb])

    rowok = okb[gb] & (k >= 0) & (k < n_epochs) & np.isfinite(H)
    gb, k, H, X = gb[rowok], k[rowok], H[rowok], X[rowok]

    pk = gb.astype(np.int64) * n_epochs + k
    upk, pinv = np.unique(pk, return_inverse=True)
    pinv = pinv.ravel()
    pb, pk_u = upk // n_epochs, upk % n_epochs

    V = np.stack([np.bincount(pinv, weights=X[:, a], minlength=len(upk)) for a in range(p)], axis=1)
    W = np.einsum("bij,bj->bi", Ai[pb], V)
    u = np.stack([np.bincount(gb, weights=X[:, a] * H, minlength=nb) for a in range(p)], axis=1)
    Aiu = np.einsum("bij,bj->bi", Ai, u)

    ZtZ = np.bincount(k, minlength=n_epochs).astype(float)
    Zth = np.bincount(k, weights=H, minlength=n_epochs)
    g = Zth - np.bincount(pk_u, weights=np.einsum("ij,ij->i", V, Aiu[pb]), minlength=n_epochs)

    M = np.diag(ZtZ)
    order = np.argsort(pb, kind="stable")
    pbs = pb[order]
    starts = np.r_[0, np.nonzero(np.diff(pbs))[0] + 1]
    counts = np.diff(np.r_[starts, len(pbs)])
    for c in np.unique(counts):
        st = starts[counts == c]
        idx = order[st[:, None] + np.arange(c)[None, :]]
        i1 = np.repeat(idx, c, axis=1).ravel()
        i2 = np.tile(idx, (1, c)).ravel()
        np.add.at(M, (pk_u[i1], pk_u[i2]), -np.einsum("ij,ij->i", V[i1], W[i2]))

    present = ZtZ > 0
    ref = int(np.argmax(present))
    keep = present.copy()
    keep[ref] = False
    s = np.full(n_epochs, np.nan)
    s[ref] = 0.0
    if keep.any():
        s[keep] = np.linalg.lstsq(M[np.ix_(keep, keep)], g[keep], rcond=None)[0]
    s -= np.nanmean(s)
    return s, ZtZ.astype(int)


def schur_complement_beta(
    X_nuisance: np.ndarray,
    group_nuisance: np.ndarray,
    z_target: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Resolve $y = X_b \\alpha_b + z \\beta + \\varepsilon$ para um escalar/vetor $\\beta$ via Schur.

    Retorna ``(beta_hat, h_schur, z_projected, residuals)`` para uso direto em ``cr1_clustered_se``.
    """
    X = np.asarray(X_nuisance, dtype=float)
    g = np.asarray(group_nuisance, dtype=int)
    z = np.asarray(z_target, dtype=float)
    y = np.asarray(y, dtype=float)

    _, g = np.unique(g, return_inverse=True)
    g = g.ravel()
    nb = int(g.max()) + 1
    p = X.shape[1]

    A = np.zeros((nb, p, p))
    Xz = np.zeros((nb, p))
    Xy = np.zeros((nb, p))
    for a in range(p):
        Xz[:, a] = np.bincount(g, weights=X[:, a] * z, minlength=nb)
        Xy[:, a] = np.bincount(g, weights=X[:, a] * y, minlength=nb)
        for b in range(a, p):
            v = np.bincount(g, weights=X[:, a] * X[:, b], minlength=nb)
            A[:, a, b] = v
            A[:, b, a] = v

    cond = np.linalg.cond(A)
    ok = np.isfinite(cond) & (cond < 1e10)
    Ai = np.zeros_like(A)
    Ai[ok] = np.linalg.inv(A[ok])

    rowok = ok[g]
    Aiz = np.einsum("bij,bj->bi", Ai, Xz)
    Aiy = np.einsum("bij,bj->bi", Ai, Xy)

    z_proj = z[rowok] - np.einsum("ij,ij->i", X[rowok], Aiz[g[rowok]])
    y_proj = y[rowok] - np.einsum("ij,ij->i", X[rowok], Aiy[g[rowok]])

    h_schur = float(np.sum(z_proj ** 2))
    if h_schur <= 0:
        return np.nan, 0.0, z_proj, np.full_like(y_proj, np.nan)
    beta_hat = float(np.sum(z_proj * y_proj) / h_schur)
    resid = y_proj - beta_hat * z_proj
    return beta_hat, h_schur, z_proj, resid


def cr1_clustered_se(
    z_projected: np.ndarray,
    residuals: np.ndarray,
    cluster_ids: np.ndarray,
    h_schur: float,
) -> tuple[float, int]:
    """Calcula o erro-padrão sanduíche agrupado CR1 (Liang & Zeger, 1986; Cameron & Miller, 2015)."""
    zp = np.asarray(z_projected, dtype=float)
    res = np.asarray(residuals, dtype=float)
    cid = np.asarray(cluster_ids)
    _, cg = np.unique(cid, return_inverse=True)
    cg = cg.ravel()
    n_clusters = int(cg.max()) + 1
    if n_clusters < 2 or h_schur <= 0:
        return np.nan, n_clusters

    score_g = np.bincount(cg, weights=zp * res, minlength=n_clusters)
    meat = float(np.sum(score_g ** 2))
    dof_factor = n_clusters / (n_clusters - 1.0)
    var_cr1 = dof_factor * meat / (h_schur ** 2)
    return float(np.sqrt(max(var_cr1, 0.0))), n_clusters
