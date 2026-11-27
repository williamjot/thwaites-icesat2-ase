"""
Interpolação espacial de dh/dt.

O MÉTODO não é assumido: `select.py` roda validação cruzada com blocos
espaciais entre os candidatos (krigagem ordinária, OI/Markov, IDW) e escolhe
por RMSE + calibração da incerteza. Os parâmetros do variograma são derivados
dos dados (`variogram.py`), não fixados por convenção.

`spacetime.py` acrescenta a interpolação ESPAÇO-TEMPO por kernel gaussiano
(inspirada no `interpgaus3d.py` do captoolkit), que monta cubos tempo × y × x
de anomalias de altura.
"""

from thwaites.interp.variogram import empirical_variogram, fit_variogram, make_gamma
from thwaites.interp.methods import (
    idw_predict, oi_markov_predict, ordinary_kriging_predict,
    gaussian_kernel_predict, median_kernel_predict, PREDICTORS,
)
from thwaites.interp.select import spatial_block_folds, cross_validate, select_interpolator
from thwaites.interp.spacetime import gaussian_spacetime_predict, build_spacetime_cube

__all__ = [
    "empirical_variogram", "fit_variogram", "make_gamma",
    "idw_predict", "oi_markov_predict", "ordinary_kriging_predict",
    "gaussian_kernel_predict", "median_kernel_predict", "PREDICTORS",
    "spatial_block_folds", "cross_validate", "select_interpolator",
    "gaussian_spacetime_predict", "build_spacetime_cube",
]
