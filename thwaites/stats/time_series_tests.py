"""Testes de hipótese para séries temporais geofísicas curtas e autocorrelacionadas.

Fundamentação teórica e referências primárias
---------------------------------------------
1. **Teste $Q$ / Qui-Quadrado de Homogeneidade Interanual (Cochran, 1954):**
   Dadas $K$ estimativas anuais independentes $y_1, \\dots, y_K$ com erros-padrão conhecidos
   $\\sigma_1, \\dots, \\sigma_K$, sob a hipótese nula de taxa constante $H_0: \\mathbb{E}[y_t] = \\mu$,
   o estimador de variância mínima é a média ponderada por variância inversa
   $\\hat{\\mu}_w = (\\sum w_t y_t) / (\\sum w_t)$, com $w_t = \\sigma_t^{-2}$, e a estatística
   $$Q = \\sum_{t=1}^{K} \\left(\\frac{y_t - \\hat{\\mu}_w}{\\sigma_t}\\right)^2 \\sim \\chi^2_{K-1}$$
   quantifica se a dispersão observada entre anos excede a variância explicada pelo ruído de medida.

2. **Teste de Diferença entre Janelas Disjuntas Pré/Pós-Perturbação (Shadish, Cook & Campbell, 2002):**
   Para testar a resposta a um evento datado (ex.: desprendimento da frente da Pine Island em 2020)
   sem o artefato de suavização de janelas móveis que misturam épocas antes e depois do evento,
   estima-se a inclinação linear ponderada em janelas estritamente disjuntas $T_{\\text{pré}}$ e
   $T_{\\text{pós}}$ (excluindo o trimestre de transição), avaliando:
   $$z = \\frac{\\hat{m}_{b,\\text{pós}} - \\hat{m}_{b,\\text{pré}}}{\\sqrt{\\sigma_{\\text{pré}}^2 + \\sigma_{\\text{pós}}^2}}$$
   em comparação com setores de controle ($> 10\\text{ km}$ da frente e plataformas vizinhas).

3. **Graus de Liberdade Efetivos sob Autocorrelação Serial (Bretherton et al., 1999; Pyper & Peterman, 1998):**
   Para duas séries estacionárias de comprimento $N$ com autocorrelações de lag 1 iguais a
   $\\rho_{1,a}$ e $\\rho_{1,b}$, o número efetivo de amostras independentes sob aproximação AR(1) é
   (Bretherton et al., 1999, Eq. 31):
   $$N_{\\text{ef}}^{\\text{AR1}} = N \\frac{1 - \\rho_{1,a}\\rho_{1,b}}{1 + \\rho_{1,a}\\rho_{1,b}}$$
   Quando a série foi submetida a filtro de média móvel (processo MA), a autocorrelação persiste
   em lags $k > 1$, sendo mais apropriado o estimador truncado de Bartlett/Pyper-Peterman (1998):
   $$N_{\\text{ef}}^{\\text{PP}} = \\frac{N}{1 + 2\\sum_{k=1}^{K_{\\max}} \\left(1 - \\frac{k}{N}\\right)\\hat{\\rho}_a(k)\\hat{\\rho}_b(k)}$$

4. **Séries Substitutas por Randomização de Fase de Fourier (Theiler et al., 1992; Ebisuzaki, 1997):**
   Pelo teorema de Wiener-Khinchin, a função de autocorrelação de uma série estacionária é
   determinada exclusivamente pelo módulo ao quadrado de sua transformada de Fourier $|\\hat{x}(\\omega)|^2$.
   Multiplicar cada componente espectral por uma fase uniforme independente $e^{i\\phi(\\omega)}$
   ($\\phi(\\omega) \\sim U(0, 2\\pi)$, preservando simetria hermitiana) gera realizações nulas
   independentes com o mesmo espectro de potência e mesma memória temporal da série original.

5. **Teste de Mudança de Regime em Degrau via $\\Delta\\text{BIC}$ (Schwarz, 1978):**
   Compara o modelo de tendência linear contínua ($k_1 = 2$ parâmetros) contra o modelo de
   degrau em $t_0$ ($k_2 = 3$ parâmetros) via $\\text{BIC} = n\\ln(\\text{RSS}/n) + k\\ln n$.
   Pela escala de Kass & Raftery (1995), $\\Delta\\text{BIC} = \\text{BIC}_{\\text{reta}} - \\text{BIC}_{\\text{degrau}} > 6$
   corresponde a evidência forte a favor da quebra estrutural.

Referências
-----------
- Cochran, W. G. (1954). The combination of estimates from different experiments. *Biometrics*, 10(1), 101-129.
- Bretherton, C. S., Widmann, M., Dymnikov, V. P., Wallace, J. M., & Bladé, I. (1999). The effective
  number of spatial degrees of freedom of a time-varying field. *Journal of Climate*, 12(7), 1990-2009.
- Pyper, B. J., & Peterman, R. M. (1998). Comparison of methods to account for autocorrelation in
  correlation analyses of fish data. *Can. J. Fish. Aquat. Sci.*, 55(9), 2127-2140.
- Theiler, J., Eubank, S., Longtin, A., Galdrikian, B., & Farmer, J. D. (1992). Testing for
  nonlinearity in time series: the method of surrogate data. *Physica D*, 58(1-4), 77-94.
- Ebisuzaki, W. (1997). A method to estimate the statistical significance of a correlation when
  the data are serially correlated. *Journal of Climate*, 10(9), 2147-2153.
- Schwarz, G. (1978). Estimating the dimension of a model. *Annals of Statistics*, 6(2), 461-464.
- Kass, R. E., & Raftery, A. E. (1995). Bayes factors. *JASA*, 90(430), 773-795.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2


def cochran_interannual_chi2(values, sigmas) -> dict:
    """Teste Q / qui-quadrado de Cochran (1954) para constância interanual."""
    y = np.asarray(values, dtype=float)
    s = np.asarray(sigmas, dtype=float)
    ok = np.isfinite(y) & np.isfinite(s) & (s > 0)
    y, s = y[ok], s[ok]
    k = len(y)
    if k < 2:
        return {"weighted_mean": np.nan, "sigma_mean": np.nan, "chi2": np.nan, "dof": 0, "p_value": np.nan}
    w = 1.0 / (s ** 2)
    mean_w = float(np.sum(w * y) / np.sum(w))
    se_mean = float(np.sqrt(1.0 / np.sum(w)))
    q_stat = float(np.sum(((y - mean_w) / s) ** 2))
    dof = k - 1
    p_val = float(chi2.sf(q_stat, dof))
    return {"weighted_mean": mean_w, "sigma_mean": se_mean, "chi2": q_stat, "dof": dof, "p_value": p_val}


def weighted_linear_slope(t, y, sigma) -> tuple[float, float]:
    """Declive linear ponderado por variância inversa $w_i = \\sigma_i^{-2}$ e seu erro-padrão formal."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    e = np.asarray(sigma, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y) & np.isfinite(e) & (e > 0)
    t, y, e = t[ok], y[ok], e[ok]
    if len(t) < 2:
        return np.nan, np.nan
    w = 1.0 / np.maximum(e, 1e-6) ** 2
    tb = np.sum(w * t) / np.sum(w)
    yb = np.sum(w * y) / np.sum(w)
    sxx = np.sum(w * (t - tb) ** 2)
    if sxx <= 0:
        return np.nan, np.nan
    slope = float(np.sum(w * (t - tb) * (y - yb)) / sxx)
    se = float(np.sqrt(1.0 / sxx))
    return slope, se


def pre_post_window_difference(
    t, y, sigma, pre_idx, post_idx, smb_pre=0.0, smb_post=0.0, hdiv_pre=0.0, hdiv_post=0.0
) -> dict:
    """Compara taxas de derretimento basal entre janelas temporais disjuntas antes e após um evento."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    dhdt_pre, se_pre = weighted_linear_slope(t[pre_idx], y[pre_idx], sigma[pre_idx])
    dhdt_post, se_post = weighted_linear_slope(t[post_idx], y[post_idx], sigma[post_idx])
    mb_pre = float(smb_pre - dhdt_pre - hdiv_pre)
    mb_post = float(smb_post - dhdt_post - hdiv_post)
    delta = mb_post - mb_pre
    se_delta = float(np.hypot(se_pre, se_post))
    z_score = float(delta / se_delta) if se_delta > 0 else np.nan
    return {
        "mb_pre": mb_pre, "sigma_pre": se_pre,
        "mb_post": mb_post, "sigma_post": se_post,
        "delta": delta, "sigma_delta": se_delta, "z_score": z_score,
    }


def bretherton_effective_dof(a: np.ndarray, b: np.ndarray) -> float:
    """Graus de liberdade efetivos sob aproximação AR(1) (Bretherton et al., 1999, Eq. 31)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(a)
    if n < 4:
        return float(n)
    r1_a = float(np.corrcoef(a[:-1], a[1:])[0, 1])
    r1_b = float(np.corrcoef(b[:-1], b[1:])[0, 1])
    prod = np.clip(r1_a * r1_b, -0.99, 0.99)
    return float(n * (1.0 - prod) / (1.0 + prod))


def pyper_peterman_effective_dof(a: np.ndarray, b: np.ndarray, max_lag: int | None = None) -> float:
    """Graus de liberdade efetivos integrando múltiplos lags de autocorrelação (Pyper & Peterman, 1998)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok] - np.mean(a[ok]), b[ok] - np.mean(b[ok])
    n = len(a)
    if n < 5:
        return float(n)
    k_max = min(max_lag or (n // 3), n - 2)
    va, vb = float(np.sum(a ** 2)), float(np.sum(b ** 2))
    if va <= 0 or vb <= 0:
        return float(n)
    corr_sum = 0.0
    for k in range(1, k_max + 1):
        ra = float(np.sum(a[:-k] * a[k:]) / va)
        rb = float(np.sum(b[:-k] * b[k:]) / vb)
        corr_sum += (1.0 - k / n) * ra * rb
    denom = max(1.0 + 2.0 * corr_sum, 1.0)
    return float(np.clip(n / denom, 1.0, float(n)))


def fourier_phase_surrogate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Gera série substituta isoespectral por randomização de fase de Fourier (Ebisuzaki, 1997)."""
    x = np.asarray(x, dtype=float)
    mu = float(np.mean(x))
    spec = np.fft.rfft(x - mu)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=len(spec))
    phases[0] = 0.0
    if len(x) % 2 == 0:
        phases[-1] = 0.0
    surr = np.fft.irfft(np.abs(spec) * np.exp(1j * phases), n=len(x))
    return surr + mu


def surrogate_correlation_test(
    a_detrended: np.ndarray,
    b_detrended: np.ndarray,
    smooth_window: int = 4,
    n_surrogates: int = 2000,
    seed: int = 20261115,
) -> dict:
    """Testa correlação entre séries filtradas usando substitutas de fase de Fourier."""
    a_raw = np.asarray(a_detrended, dtype=float)
    b_raw = np.asarray(b_detrended, dtype=float)

    def _smooth(v):
        if smooth_window <= 1:
            return v.copy()
        kernel = np.ones(smooth_window, dtype=float) / smooth_window
        out = np.full(len(v), np.nan)
        valid = np.isfinite(v)
        if valid.sum() < smooth_window:
            return out
        # média móvel centrada apenas onde todos os pontos da janela são finitos
        for i in range(len(v) - smooth_window + 1):
            seg = v[i:i + smooth_window]
            if np.isfinite(seg).all():
                out[i + smooth_window // 2] = float(np.mean(seg))
        return out

    a_sm = _smooth(a_raw)
    b_sm = _smooth(b_raw)
    ok = np.isfinite(a_sm) & np.isfinite(b_sm)
    if ok.sum() < 6:
        return {"r": np.nan, "p_surrogate": np.nan, "n_eff_ar1": np.nan, "n_eff_pp": np.nan, "n": int(ok.sum())}

    r_obs = float(np.corrcoef(a_sm[ok], b_sm[ok])[0, 1])
    neff_ar1 = bretherton_effective_dof(a_sm[ok], b_sm[ok])
    neff_pp = pyper_peterman_effective_dof(a_sm[ok], b_sm[ok])

    good = np.isfinite(a_raw) & np.isfinite(b_raw)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_surrogates):
        sa = np.full(len(a_raw), np.nan)
        sa[good] = fourier_phase_surrogate(a_raw[good], rng)
        fa = _smooth(sa)
        m = np.isfinite(fa) & np.isfinite(b_sm)
        if m.sum() >= 6 and abs(np.corrcoef(fa[m], b_sm[m])[0, 1]) >= abs(r_obs):
            exceed += 1

    return {
        "r": r_obs,
        "p_surrogate": float(exceed / n_surrogates),
        "n_eff_ar1": neff_ar1,
        "n_eff_pp": neff_pp,
        "n": int(ok.sum()),
    }


def step_change_bic_test(t: np.ndarray, y: np.ndarray, candidate_breaks: np.ndarray | None = None) -> dict:
    """Compara ajuste de degrau vs. tendência linear contínua via critério de informação bayesiano (BIC)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    n = len(y)
    if n < 6:
        return {"delta_bic": np.nan, "best_break": np.nan, "step_size": np.nan}

    # Modelo 1: tendência linear (k = 2)
    p_lin = np.polyfit(t, y, 1)
    rss_lin = float(np.sum((y - np.polyval(p_lin, t)) ** 2))
    bic_lin = n * np.log(max(rss_lin / n, 1e-30)) + 2.0 * np.log(n)

    # Modelo 2: degrau ótimo (k = 3: nível antes, nível depois, época de quebra)
    if candidate_breaks is None:
        s_t = np.sort(t)
        candidate_breaks = 0.5 * (s_t[2:-2] + s_t[3:-1])

    best_rss = np.inf
    best_t0 = np.nan
    best_step = np.nan
    for t0 in candidate_breaks:
        pre = t < t0
        post = ~pre
        if pre.sum() < 2 or post.sum() < 2:
            continue
        mu_pre = float(np.mean(y[pre]))
        mu_post = float(np.mean(y[post]))
        rss = float(np.sum((y[pre] - mu_pre) ** 2) + np.sum((y[post] - mu_post) ** 2))
        if rss < best_rss:
            best_rss = rss
            best_t0 = float(t0)
            best_step = mu_post - mu_pre

    bic_step = n * np.log(max(best_rss / n, 1e-30)) + 3.0 * np.log(n)
    return {
        "delta_bic": float(bic_lin - bic_step),
        "best_break": best_t0,
        "step_size": best_step,
        "bic_linear": float(bic_lin),
        "bic_step": float(bic_step),
    }
