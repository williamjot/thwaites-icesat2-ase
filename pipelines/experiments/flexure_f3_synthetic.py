"""F3 do plano de inversão de flexão: teste sintético (docs/PLANO_INVERSAO_FLEXAO_MARE.md).

Pergunta: com o ruído do ATL06, o relevo carregado pelo fluxo da ASE e uma maré prevista
imperfeita, a forma relativa da espessura h(x)/h(L) e o seu gradiente perto da linha de
aterramento são recuperados com erro < 10%?

Dois métodos:
- **A, por passagem (como Hendley Elgart 2026):** w_j(x) = altura − média euleriana das
  passagens; ajusta-se w_j = A_j·β(x; θ), com A_j livre por passagem;
- **B, perfil de β Lagrangiano (proposta deste projeto):** mínimos quadrados conjuntos
  h = relevo material (bins de 100 m na posição de 2022, com tendência) + β_c·T_j, com β_c em
  bins eulerianos de 250 m e T_j = maré prevista com erro de amplitude; depois ajusta-se
  β_c ≈ P·β(x; θ), com o patamar P livre (absorve o erro de amplitude do modelo de maré).

θ: h "efetivo" (com E fixo em 1 GPa) como spline de Hermite (h0, hL, s0, sL). Como só a rigidez
E·h³ é identificável, avalia-se a forma relativa ĥ(x)/ĥ(L) e o gradiente relativo.

Verdade sintética e ruídos (justificativa entre parênteses):
- h de 600 m na linha a 380 m em L = 15 km, E = 1 GPa (λ ≈ 1,7 km; o β medido indica
  λ de 0,75 a 1,5 km);
- 28 passagens em 7 anos; maré A_j ~ N(0; 0,37 m) (σ do CATS medido na faixa);
- ruído por segmento de 40 m: 0,10 m; desvio por passagem: 0,05 m;
- relevo superficial: 3 m RMS em comprimentos de 0,5 a 5 km, carregado pelo fluxo sobre o
  flutuante e fixo sobre o apoiado; afinamento de 0,5 m/a na superfície flutuante;
- maré prevista com +5% de erro de amplitude e 0,02 m de ruído (patamar de β medido 0,95–1,07);
- velocidade usada na inversão com +2% de erro.
Cenários: v = 0, 700 e 2.000 m/a; e v = 700 m/a com a linha de aterramento migrando
1 km por metro de maré (estresse; o ICEYE mostra quilômetros na Thwaites).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from thwaites.flexure.beam import beta_profile, hermite, rigidity  # noqa: E402

OUT = ROOT / "outputs" / "flexure_f3"
L = 15_000.0
E_REF = 1e9
TRUTH = dict(h0=600.0, hL=380.0, s0=-0.03, sL=-0.002)
XS = np.arange(-3000.0, L + 1, 40.0)
N_PASS = 28


def beta_of(theta, x):
    h0, hL, s0, sL = theta
    D = lambda xx: rigidity(E_REF, np.maximum(hermite(np.atleast_1d(xx), L, h0, hL, s0, sL), 20.0))  # noqa: E731
    return beta_profile(x, L, D)


def topo_field(rng, rms=3.0):
    """Campo 1D aleatório com potência em 0,5–5 km, definido numa grade fina."""
    X = np.arange(-60e3, 60e3, 20.0)
    f = np.fft.rfftfreq(len(X), 20.0)
    amp = np.where((f > 1 / 5000) & (f < 1 / 500), 1.0, 0.0)
    spec = amp * (rng.normal(size=len(f)) + 1j * rng.normal(size=len(f)))
    z = np.fft.irfft(spec, n=len(X))
    z *= rms / z.std()
    return lambda q: np.interp(q, X, z)


def simulate(rng, v, migrate=0.0):
    beta_true = beta_of(list(TRUTH.values()), XS)
    t = 2019.0 + np.arange(N_PASS) * 0.25 + rng.uniform(-0.05, 0.05, N_PASS)
    A = rng.normal(0, 0.37, N_PASS)
    topo_f, topo_g = topo_field(rng), topo_field(rng)
    H = np.empty((N_PASS, len(XS)))
    for j in range(N_PASS):
        shift = migrate * A[j]                       # linha de aterramento desloca com a maré
        b = beta_of(list(TRUTH.values()), XS - shift) if migrate else beta_true
        topo = np.where(XS > 0, topo_f(XS - v * (t[j] - 2022.0)), topo_g(XS))
        thin = np.where(XS > 0, -0.5 * (t[j] - 2022.0), 0.0)
        H[j] = 50.0 + topo + thin + A[j] * b + rng.normal(0, 0.05) + rng.normal(0, 0.10, len(XS))
    T_pred = A * 1.05 + rng.normal(0, 0.02, N_PASS)
    return t, A, T_pred, H, beta_true


# ------------------------------------------------------------------ inversões
def fit_theta(resid_fun, n_start=4, rng=None):
    best = None
    starts = [[600, 380, -0.03, -0.002]]
    rng = rng or np.random.default_rng(0)
    for _ in range(n_start - 1):
        starts.append([rng.uniform(300, 900), rng.uniform(250, 600), rng.uniform(-0.06, 0.0),
                       rng.uniform(-0.006, 0.0)])
    for s in starts:
        try:
            r = least_squares(resid_fun, s, bounds=([100, 100, -0.1, -0.02], [2000, 1500, 0.02, 0.0]),
                              x_scale=[100, 100, 0.01, 0.002], max_nfev=200)
        except RuntimeError:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best


def method_a(t, H):
    """Por passagem: anomalia contra a média euleriana; A_j livre (linear, eliminado)."""
    W = H - H.mean(0)
    sel = XS >= 0

    def resid(theta):
        b = beta_of(theta, XS[sel])
        bc = b - b.mean()                               # a média das passagens também tira β·Ā
        Aj = W[:, sel] @ bc / (bc @ bc)
        return (W[:, sel] - Aj[:, None] * bc[None, :]).ravel()

    return fit_theta(resid)


def method_b(t, T_pred, H, v_est):
    """Perfil de β: relevo material (flutuante) / fixo (apoiado) + β_c·T, depois ajuste da viga."""
    nb_e = 250.0
    rows_t = np.repeat(t, len(XS))
    rows_x = np.tile(XS, N_PASS)
    y = H.ravel()
    T = np.repeat(T_pred, len(XS))
    flo = rows_x > 0
    pos = np.where(flo, rows_x - v_est * (rows_t - 2022.0), rows_x)
    kb = np.floor(pos / 100.0).astype(int) * 2 + flo.astype(int)
    ub, gb = np.unique(kb, return_inverse=True)
    ke = np.floor(rows_x / nb_e).astype(int)
    ue, ge = np.unique(ke, return_inverse=True)
    nb, ne = len(ub), len(ue)
    from scipy import sparse
    from scipy.sparse.linalg import lsqr
    n = len(y)
    tt = rows_t - 2022.0
    cols = np.r_[gb, nb + gb, 2 * nb + ge]
    vals = np.r_[np.ones(n), tt, T]
    rr = np.r_[np.arange(n), np.arange(n), np.arange(n)]
    M = sparse.csr_matrix((vals, (rr, cols)), shape=(n, 2 * nb + ne))
    sol = lsqr(M, y, atol=1e-10, btol=1e-10, iter_lim=20000)[0]
    beta_c = sol[2 * nb:]
    res = y - M @ sol
    # erro de β_c por agrupamento em passagens
    pj = np.repeat(np.arange(N_PASS), len(XS))
    sc = np.zeros((ne, N_PASS))
    np.add.at(sc, (ge, pj), T * res)
    stt = np.bincount(ge, weights=T ** 2, minlength=ne)
    se = np.sqrt(N_PASS / (N_PASS - 1) * (sc ** 2).sum(1)) / stt
    xc = (ue + 0.5) * nb_e
    sel = (xc > 0) & (xc < L)

    def resid(theta_p):
        theta, P = theta_p[:4], theta_p[4]
        return (beta_c[sel] - P * beta_of(theta, xc[sel])) / np.maximum(se[sel], 0.02)

    best = None
    for s in ([600, 380, -0.03, -0.002, 1.0], [400, 300, -0.01, -0.001, 1.0],
              [900, 500, -0.05, -0.004, 1.0]):
        try:
            r = least_squares(resid, s, bounds=([100, 100, -0.1, -0.02, 0.5], [2000, 1500, 0.02, 0.0, 1.5]),
                              x_scale=[100, 100, 0.01, 0.002, 0.05], max_nfev=200)
        except RuntimeError:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best, xc[sel], beta_c[sel], se[sel]


def scores(theta):
    x = np.linspace(0, 10e3, 101)
    h = hermite(x, L, *theta)
    ht = hermite(x, L, *TRUTH.values())
    rel = h / hermite(np.array([L]), L, *theta)[0]
    rel_t = ht / TRUTH["hL"]
    g = (h[30] - h[0]) / 3000 / h[-1]                 # gradiente relativo médio em 0–3 km
    g_t = (ht[30] - ht[0]) / 3000 / ht[-1]
    return {"erro_forma_rms_pct": float(100 * np.sqrt(np.mean((rel / rel_t - 1) ** 2))),
            "erro_grad_0_3km_pct": float(100 * (g / g_t - 1))}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    n_real = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    scen = [("v0", 0.0, 0.0), ("v700", 700.0, 0.0), ("v2000", 2000.0, 0.0),
            ("v700_migra1km", 700.0, 1000.0)]
    results = []
    for name, v, mig in scen:
        for k in range(n_real):
            rng = np.random.default_rng(1000 * len(name) + k)
            t, A, T_pred, H, _ = simulate(rng, v, mig)
            ra = method_a(t, H)
            rb, *_ = method_b(t, T_pred, H, v * 1.02)
            row = {"cenario": name, "real": k}
            if ra is not None:
                row.update({f"A_{a}": b for a, b in scores(ra.x).items()})
            if rb is not None:
                row.update({f"B_{a}": b for a, b in scores(rb.x[:4]).items()})
                row["B_patamar"] = float(rb.x[4])
            results.append(row)
            print(json.dumps(row), flush=True)
    import pandas as pd
    d = pd.DataFrame(results)
    d.to_csv(OUT / "f3_resultados.csv", index=False)
    s = d.groupby("cenario").agg(
        A_forma_med=("A_erro_forma_rms_pct", "median"), A_forma_p90=("A_erro_forma_rms_pct", lambda v: v.quantile(.9)),
        A_grad_med=("A_erro_grad_0_3km_pct", lambda v: v.abs().median()),
        B_forma_med=("B_erro_forma_rms_pct", "median"), B_forma_p90=("B_erro_forma_rms_pct", lambda v: v.quantile(.9)),
        B_grad_med=("B_erro_grad_0_3km_pct", lambda v: v.abs().median()),
        B_patamar=("B_patamar", "median"))
    s.round(2).to_csv(OUT / "f3_resumo.csv")
    print(s.round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
