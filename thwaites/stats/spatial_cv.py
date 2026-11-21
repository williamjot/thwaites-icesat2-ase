"""Validação cruzada espacial em blocos, derivadas gaussianas normalizadas e correlação espacial do erro.

Fundamentação teórica e referências primárias
---------------------------------------------
1. **Validação Cruzada Espacial em Blocos (*Spatial Block CV*; Roberts et al., 2017; Ploton et al., 2020):**
   Quando as observações $z(\\mathbf{x}_i)$ possuem alcance de autocorrelação espacial $L > 0$,
   a validação cruzada $K$-fold aleatória coloca pontos separados por $\\Delta x \\ll L$ simultaneamente
   nos conjuntos de treino e teste. Isso viola a independência entre treino e avaliação e
   favorece interpoladores de curto alcance que subestimam o erro fora das trilhas. Particionar
   o domínio em blocos contíguos de lado $B$ garante que o erro médio quadrático avalie a
   capacidade preditiva em lacunas espaciais reais.

2. **Calibração Probabilística da Variância de Predição ($z_{\\text{std}}$; Cressie, 1993):**
   Para que a variância de krigagem $\\hat{\\sigma}^2(\\mathbf{x}_i)$ possa ser propagada para o
   balanço de massa regional, os resíduos padronizados fora da amostra
   $$z_i = \\frac{\\hat{y}_{(-b)}(\\mathbf{x}_i) - y(\\mathbf{x}_i)}{\\hat{\\sigma}_{(-b)}(\\mathbf{x}_i)}$$
   devem ter desvio-padrão $z_{\\text{std}} = \\operatorname{SD}(z) \\approx 1$ e cobertura de
   $1\\sigma$ próxima de $\\operatorname{erf}(1/\\sqrt{2}) \\approx 0{,}6827$.

3. **Convolução Diferencial Gaussiana Normalizada e CV Anos Pares vs. Ímpares
   (Knutsson & Westin, 1993):**
   Para calcular $\\nabla\\cdot\\mathbf{u}$ em campos de velocidade com lacunas (`NaN`), define-se
   o indicador de validade $w(\\mathbf{x}) \\in \\{0, 1\\}$ e aplica-se a regra do quociente ao
   campo suavizado $\\tilde{f} = (G_\\sigma * (w f)) / (G_\\sigma * w)$:
   $$\\partial_x \\tilde{f} = \\frac{(\\partial_x G_\\sigma * (w f))(G_\\sigma * w) - (G_\\sigma * (w f))(\\partial_x G_\\sigma * w)}{(G_\\sigma * w)^2}$$
   A escala ótima $\\sigma^*$ é escolhida minimizando a discrepância quadrática média cruzada entre
   a divergência média dos mosaicos de anos pares e a divergência de referência dos anos ímpares.

4. **Área Efetiva de Correlação de Resíduos Altimétricos (Rolstad, Haug & Denby, 2009):**
   Integrando um campo de erro estacionário isotrópico com variância $\\bar{\\sigma}^2$ e raio de
   correlação efetivo $L$ sobre uma área total $A_{\\text{total}} \\gg \\pi L^2$, a variância do
   volume integrado é $\\operatorname{Var}(\\Delta V) \\approx \\bar{\\sigma}^2 A_{\\text{total}} (\\pi L^2)$,
   correspondendo a $n_{\\text{eff}} = A_{\\text{total}} / (\\pi L^2)$ graus de liberdade espaciais.
   Se o semivariograma do erro possui efeito pepita (*nugget*) $c_0$ e patamar (*sill*) $c_0 + c_1$,
   apenas a fração estrutural $f_{\\text{corr}} = c_1 / (c_0 + c_1)$ é espacialmente correlacionada na
   escala $L$, enquanto a fração $c_0 / (c_0 + c_1)$ se integra na escala da célula $A_{\\text{cell}}$.

Referências
-----------
- Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial,
  hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913-929.
- Ploton, P., et al. (2020). Spatial validation reveals poor predictive performance of large-scale
  ecological mapping models. *Nature Communications*, 11, 4540.
- Cressie, N. A. C. (1993). *Statistics for Spatial Data* (Revised ed.). Wiley.
- Knutsson, H., & Westin, C.-F. (1993). Normalized and differential convolution. In *Proc. IEEE CVPR*
  (pp. 515-523).
- Rolstad, C., Haug, T., & Denby, B. (2009). Spatially integrated geodetic glacier mass balance and
  its uncertainty based on geostatistical analysis: application to the western Svartisen ice cap,
  Norway. *Journal of Glaciology*, 55(192), 666-680.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage


def spatial_block_assignments(x: np.ndarray, y: np.ndarray, block_size_m: float, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Atribui observações $(x, y)$ a $K$ dobras agrupando blocos espaciais inteiros de lado ``block_size_m``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    bi = np.floor(x / block_size_m).astype(np.int64)
    bj = np.floor(y / block_size_m).astype(np.int64)
    keys = bi * 1_000_003 + bj
    uniq, inv = np.unique(keys, return_inverse=True)
    rng = np.random.default_rng(seed)
    order = np.arange(len(uniq))
    rng.shuffle(order)
    fold_of_uniq = np.empty(len(uniq), dtype=int)
    fold_of_uniq[order] = np.arange(len(uniq)) % n_folds
    return fold_of_uniq[inv.ravel()]


def standardized_error_calibration(obs: np.ndarray, pred: np.ndarray, var_pred: np.ndarray) -> dict:
    """Calcula RMSE, MAE, viés, cobertura de 1 sigma e desvio-padrão do erro padronizado $z_{\\text{std}}$."""
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    var = np.asarray(var_pred, dtype=float)
    ok = np.isfinite(obs) & np.isfinite(pred)
    if not ok.any():
        return {"rmse": np.nan, "mae": np.nan, "bias": np.nan, "coverage_1sigma": np.nan, "z_std": np.nan}
    res = pred[ok] - obs[ok]
    rmse = float(np.sqrt(np.mean(res ** 2)))
    mae = float(np.mean(np.abs(res)))
    bias = float(np.mean(res))
    v_ok = ok & np.isfinite(var) & (var > 0)
    if v_ok.sum() > 1:
        z = (pred[v_ok] - obs[v_ok]) / np.sqrt(var[v_ok])
        cov1 = float(np.mean(np.abs(z) < 1.0))
        z_std = float(np.std(z))
    else:
        cov1 = z_std = np.nan
    return {"rmse": rmse, "mae": mae, "bias": bias, "coverage_1sigma": cov1, "z_std": z_std}


def normalized_gaussian_derivative(field: np.ndarray, sigma_px: float, axis: int) -> np.ndarray:
    """Derivada gaussiana de 1ª ordem por convolução normalizada resistente a lacunas (Knutsson & Westin, 1993)."""
    a = np.asarray(field, dtype=np.float32)
    w = np.isfinite(a).astype(np.float32)
    f = np.where(w > 0, a, 0.0).astype(np.float32)
    order = [0, 0]
    order[axis] = 1
    num_d = ndimage.gaussian_filter(f, sigma_px, order=order, mode="nearest")
    num = ndimage.gaussian_filter(f, sigma_px, mode="nearest")
    den = ndimage.gaussian_filter(w, sigma_px, mode="nearest")
    den_d = ndimage.gaussian_filter(w, sigma_px, order=order, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (num_d * den - num * den_d) / (den ** 2)
    out[den < 0.5] = np.nan
    return out


def even_odd_divergence_cv(div_even: np.ndarray, div_odd: np.ndarray, ref_even: np.ndarray, ref_odd: np.ndarray, mask: np.ndarray) -> dict:
    """Calcula a perda de validação cruzada split-sample entre anos pares e ímpares."""
    dom = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(div_even) & np.isfinite(div_odd)
        & np.isfinite(ref_even) & np.isfinite(ref_odd)
    )
    if not dom.any():
        return {"cv_loss": np.nan, "noise_7yr": np.nan, "signal_rms": np.nan}
    cv = float(np.sqrt(0.5 * (
        np.mean((div_even[dom] - ref_odd[dom]) ** 2)
        + np.mean((div_odd[dom] - ref_even[dom]) ** 2)
    )))
    noise = float(np.std(div_even[dom] - div_odd[dom]) / 2.0)
    mean_field = 0.5 * (div_even[dom] + div_odd[dom])
    sig = float(np.sqrt(np.mean(mean_field ** 2)))
    return {"cv_loss": cv, "noise_7yr": noise, "signal_rms": sig}


def rolstad_effective_samples(
    area_total_m2: float,
    correlation_length_m: float,
    cell_area_m2: float | None = None,
    nugget_fraction: float = 0.0,
) -> dict:
    """Calcula a área de correlação e o número efetivo de graus de liberdade espaciais (Rolstad et al., 2009)."""
    a_corr = math.pi * (correlation_length_m ** 2)
    f_nug = float(np.clip(nugget_fraction, 0.0, 1.0))
    a_cell = cell_area_m2 if cell_area_m2 is not None else a_corr
    a_eff = f_nug * a_cell + (1.0 - f_nug) * a_corr
    n_eff = area_total_m2 / max(a_eff, 1.0)
    return {
        "area_corr_m2": a_corr,
        "area_eff_m2": a_eff,
        "n_eff": n_eff,
        "nugget_fraction": f_nug,
    }
