"""Subpacote de inferência estatística, reamostragem em blocos, modelos em painel e validação cruzada espacial.

Cada módulo implementa estimadores com rastreabilidade teórica explícita:
- ``trend_tests``: Mann-Kendall (Mann, 1945; Kendall, 1975), Mann-Kendall sazonal
  (Hirsch, Slack & Smith, 1982), declive de Theil-Sen (Sen, 1968) e controle de taxa
  de falsas descobertas (Benjamini & Hochberg, 1995; Benjamini & Yekutieli, 2001).
- ``resampling``: Jackknife agrupado por ano (Tukey, 1958; Efron & Stein, 1981),
  Delete-a-Group Jackknife sobre passagens orbitais (Kott, 2001) e Block Bootstrap
  por equações normais pré-acumuladas (Carlstein, 1986).
- ``panel_schur``: Estimador de efeitos fixos de célula material (Wooldridge, 2010),
  eliminação analítica por complemento de Schur / teorema de Frisch-Waugh-Lovell
  (Frisch & Waugh, 1933; Lovell, 1963) e variância sanduíche agrupada CR1
  (Liang & Zeger, 1986; Cameron & Miller, 2015).
- ``time_series_tests``: Teste Q / qui-quadrado de homogeneidade interanual (Cochran, 1954),
  teste pré/pós-desprendimento em janelas disjuntas, graus de liberdade efetivos
  (Bretherton et al., 1999; Pyper & Peterman, 1998), séries substitutas por fase
  de Fourier aleatória (Theiler et al., 1992; Ebisuzaki, 1997) e detecção de degrau
  por critério de informação bayesiano (Schwarz, 1978).
- ``spatial_cv``: Validação cruzada espacial em blocos (Roberts et al., 2017),
  calibração probabilística de incerteza (Cressie, 1993), validação cruzada
  anos pares vs. ímpares para derivadas gaussianas normalizadas (Knutsson & Westin, 1993)
  e comprimento de correlação de resíduos altimétricos (Rolstad, Haug & Denby, 2009).
"""

from thwaites.stats.panel_schur import (
    cr1_clustered_se,
    fixed_effects_panel_series,
    schur_complement_beta,
)
from thwaites.stats.resampling import (
    block_bootstrap_normal_eq,
    delete_group_jackknife,
    leave_one_year_out_jackknife,
)
from thwaites.stats.spatial_cv import (
    even_odd_divergence_cv,
    normalized_gaussian_derivative,
    rolstad_effective_samples,
    spatial_block_assignments,
    standardized_error_calibration,
)
from thwaites.stats.time_series_tests import (
    bretherton_effective_dof,
    cochran_interannual_chi2,
    fourier_phase_surrogate,
    pre_post_window_difference,
    pyper_peterman_effective_dof,
    step_change_bic_test,
    surrogate_correlation_test,
)
from thwaites.stats.trend_tests import (
    benjamini_hochberg_fdr,
    mann_kendall_sen,
    seasonal_mann_kendall_sen,
)

__all__ = [
    "benjamini_hochberg_fdr",
    "block_bootstrap_normal_eq",
    "bretherton_effective_dof",
    "cochran_interannual_chi2",
    "cr1_clustered_se",
    "delete_group_jackknife",
    "even_odd_divergence_cv",
    "fixed_effects_panel_series",
    "fourier_phase_surrogate",
    "leave_one_year_out_jackknife",
    "mann_kendall_sen",
    "normalized_gaussian_derivative",
    "pre_post_window_difference",
    "pyper_peterman_effective_dof",
    "rolstad_effective_samples",
    "schur_complement_beta",
    "seasonal_mann_kendall_sen",
    "spatial_block_assignments",
    "standardized_error_calibration",
    "step_change_bic_test",
    "surrogate_correlation_test",
]
