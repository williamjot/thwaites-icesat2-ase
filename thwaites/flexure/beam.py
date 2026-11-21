"""Modelo direto de flexão por maré: viga elástica de rigidez variável apoiada na água.

    (D(x) w'')'' + k (w − A) = 0,     k = ρw g,     D = E h³ / (12 (1 − ν²))

Condições de contorno (Hendley Elgart 2026, cap. 3): engaste no lado apoiado (x = 0: w = 0,
w' = 0) e flutuação livre no lado do mar (x = L: w = A, w' = 0).

O problema é linear em A, então basta resolver com A = 1: w(x) = A·β(x), onde β(x) é a resposta
normalizada à maré, a mesma grandeza medida por ``pipelines/run_tide_response_lagrangian.py``.

**Identificabilidade:** a equação só contém D(x). E e h entram apenas pelo produto E·h³. A flexão
determina a rigidez; a espessura absoluta só sai com E conhecido. Com E constante ao longo da
zona, a forma relativa h(x)/h(L) = (D(x)/D(L))^(1/3) é determinada sem conhecer E.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_bvp

RHO_W, G, NU = 1027.0, 9.81, 0.3
K = RHO_W * G


def rigidity(E, h):
    return E * h ** 3 / (12 * (1 - NU ** 2))


def hermite(x, L, h0, hL, s0, sL):
    """h(x) como spline cúbica de Hermite em [0, L]: valores e inclinações (m/m) nas pontas."""
    t = np.clip(x / L, 0, 1)
    h00 = 2 * t ** 3 - 3 * t ** 2 + 1
    h10 = t ** 3 - 2 * t ** 2 + t
    h01 = -2 * t ** 3 + 3 * t ** 2
    h11 = t ** 3 - t ** 2
    return h00 * h0 + h10 * L * s0 + h01 * hL + h11 * L * sL


def beta_profile(x_eval, L, D_fun, n_mesh=400):
    """β(x) = w/A para rigidez D_fun(x) [N·m]. Devolve β em x_eval (fora de [0, L]: 0 ou 1)."""
    xm = np.linspace(0, L, n_mesh)

    def ode(x, y):
        w, w1, M, V = y
        D = D_fun(x)
        return np.vstack([w1, M / D, V, -K * (w - 1.0)])

    def bc(ya, yb):
        return np.array([ya[0], ya[1], yb[0] - 1.0, yb[1]])

    lam = (4 * D_fun(np.array([L]))[0] / K) ** 0.25
    w0 = 1 - np.exp(-xm / lam) * (np.cos(xm / lam) + np.sin(xm / lam))
    y0 = np.vstack([w0, np.gradient(w0, xm), np.zeros_like(xm), np.zeros_like(xm)])
    sol = solve_bvp(ode, bc, xm, y0, tol=1e-6, max_nodes=20000)
    if not sol.success:
        raise RuntimeError(f"solve_bvp falhou: {sol.message}")
    out = np.where(x_eval <= 0, 0.0, 1.0)
    inside = (x_eval > 0) & (x_eval < L)
    out[inside] = sol.sol(x_eval[inside])[0]
    return out


def beta_analytic_uniform(x, D):
    """Viga semi-infinita com engaste e rigidez uniforme (verificação)."""
    lam = (4 * D / K) ** 0.25
    x = np.asarray(x, float)
    return np.where(x <= 0, 0.0, 1 - np.exp(-x / lam) * (np.cos(x / lam) + np.sin(x / lam)))
