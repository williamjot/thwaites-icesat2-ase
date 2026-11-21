"""Teste da hipótese de Meng et al. 2025: coerência ENSO das anomalias de derretimento basal.

Três provas, todas sobre a série trimestral do passo 7 (``mb_serie_trimestral.csv``):

1. **Coerência entre plataformas.** Anomalia = série sem a tendência linear e suavizada por média
   móvel de 4 trimestres (variante A); a variante B, mais próxima de Meng, tira uma média móvel de
   12 trimestres (passa-baixa de 3 anos) antes da suavização. As pontas da série (2 trimestres de
   cada lado, onde a derivada usa janela unilateral) são descartadas.
   Significância: (i) graus de liberdade efetivos de Bretherton, N_ef = N(1 − ρ1ρ2)/(1 + ρ1ρ2);
   (ii) 2.000 substitutas com fase de Fourier aleatória da série sem tendência (preservam o
   espectro), filtradas do mesmo modo.
2. **Troca do firn.** A mesma coerência com o DH/Dt calculado com o firn IMAU (2019–2023).
   Se a coerência for do oceano, ela sobrevive à troca do modelo de firn.
3. **Contribuição direta do firn.** Como H = 9,34·(h − FAC), o firn entra em m_b como
   +9,34·dFAC/dt. Mede-se a correlação de cada anomalia de m_b com essa contribuição (GEMB) e
   com o ONI (NOAA CPC, defasagens de 0 a 4 trimestres).

Saída: ``outputs/coherence_test/``.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
SER = ROOT / "outputs" / "basal_melt_ase" / "mb_serie_trimestral.csv"
THK = ROOT / "outputs" / "thickness_ase" / "serie_espessura_classe.csv"
MB = ROOT / "data" / "interim" / "basal_melt" / "mb_continuo_500m.nc"
GEMB_FAC = ROOT / "data" / "basal_forcing" / "gemb_fac_roi_2018_2025.nc"
ONI = ROOT / "data" / "external_indices" / "oni.ascii.txt"
OUT = ROOT / "outputs" / "coherence_test"
UNITS = ["Pine_Island", "Thwaites_TEIS", "Crosson", "Dotson", "Cosgrove"]   # TWGT fora (mélange)
FACTOR = 1027.0 / (1027.0 - 917.0)
N_SURR = 2000
RNG = np.random.default_rng(20261115)


def window_slope(t, y, e=None, half=2):
    out = np.full(len(y), np.nan)
    for k in range(len(y)):
        s = slice(max(k - half, 0), min(k + half + 1, len(y)))
        tw, yw = t[s], y[s]
        w = np.ones_like(tw) if e is None else 1 / np.maximum(e[s], 1e-3) ** 2
        tb = np.sum(w * tw) / w.sum()
        out[k] = np.sum(w * (tw - tb) * (yw - np.sum(w * yw) / w.sum())) / np.sum(w * (tw - tb) ** 2)
    return out


def run_mean(y, n):
    return pd.Series(y).rolling(n, center=True, min_periods=n).mean().to_numpy()


def anomaly(t, y, variant="A"):
    ok = np.isfinite(y)
    p = np.polyfit(t[ok], y[ok], 1)
    r = y - np.polyval(p, t)
    if variant == "B":
        r = r - run_mean(r, 12)
    return run_mean(r, 4), r


def surrogate(x):
    """Substituta com o mesmo espectro e fases aleatórias."""
    f = np.fft.rfft(x - x.mean())
    ph = RNG.uniform(0, 2 * np.pi, len(f))
    ph[0] = 0
    if len(x) % 2 == 0:
        ph[-1] = 0
    return np.fft.irfft(np.abs(f) * np.exp(1j * ph), n=len(x)) + x.mean()


def corr_test(a_raw, b_raw, variant):
    """r entre as anomalias filtradas; p por substitutas; N_ef de Bretherton."""
    a = run_mean(a_raw, 4) if variant == "A" else run_mean(a_raw - run_mean(a_raw, 12), 4)
    b = run_mean(b_raw, 4) if variant == "A" else run_mean(b_raw - run_mean(b_raw, 12), 4)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 6:
        return np.nan, np.nan, np.nan, int(ok.sum())
    r = np.corrcoef(a[ok], b[ok])[0, 1]
    r1 = lambda z: np.corrcoef(z[:-1], z[1:])[0, 1]  # noqa: E731
    n = ok.sum()
    neff = n * (1 - r1(a[ok]) * r1(b[ok])) / (1 + r1(a[ok]) * r1(b[ok]))
    good = np.isfinite(a_raw) & np.isfinite(b_raw)
    cnt = 0
    for _ in range(N_SURR):
        sa = np.full(len(a_raw), np.nan)
        sa[good] = surrogate(a_raw[good])
        fa = run_mean(sa, 4) if variant == "A" else run_mean(sa - run_mean(sa, 12), 4)
        m = np.isfinite(fa) & np.isfinite(b)
        if m.sum() >= 6 and abs(np.corrcoef(fa[m], b[m])[0, 1]) >= abs(r):
            cnt += 1
    return float(r), float(cnt / N_SURR), float(neff), int(n)


def oni_quarterly(tm):
    rows = [l.split() for l in ONI.read_text().splitlines()[1:] if l.strip()]
    d = pd.DataFrame(rows, columns=["s", "y", "tot", "anom"])
    d["y"] = d.y.astype(int)
    d["anom"] = d.anom.astype(float)
    key = {"DJF": 0.04, "MAM": 0.29, "JJA": 0.54, "SON": 0.79}
    d = d[d.s.isin(key)]
    d["t"] = d.y + d.s.map(key)
    return np.interp(tm, d.t.to_numpy(), d.anom.to_numpy())


def unit_fac_series(units, tm):
    g = xr.open_dataset(GEMB_FAC)
    fac = np.ma.filled(np.ma.masked_invalid(g.fac.values.astype(float)), np.nan)
    tf = g.time.values.astype(float)
    tree = cKDTree(np.c_[g.x.values, g.y.values])
    mb = xr.open_dataset(MB)
    XX, YY = np.meshgrid(mb.x.values, mb.y.values)
    out = {}
    for u in units:
        s = mb.unidade.values == u
        idx = np.unique(tree.query(np.c_[XX[s], YY[s]])[1])
        series = np.nanmean(fac[:, idx], 1)
        out[u] = np.interp(tm, tf, series)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(SER)
    th = pd.read_csv(THK)
    tm = np.sort(s.t_meio.unique())
    keep = (np.arange(len(tm)) >= 2) & (np.arange(len(tm)) <= len(tm) - 3)
    fac = unit_fac_series(UNITS, tm)
    oni = oni_quarterly(tm)
    mb, mb_imau, firn = {}, {}, {}
    for u in UNITS:
        g = s[s.unidade == u].sort_values("t_meio")
        y = g.mb_mpa.to_numpy().copy()
        y[~keep] = np.nan
        mb[u] = y
        h = th[th.plataforma == u].sort_values("t_meio")
        dI = window_slope(h.t_meio.to_numpy(), h.dH_lagr_imau_m.to_numpy(), h.se_lagr_m.to_numpy())
        yi = g.SMB_mpa.to_numpy() - dI - g.Hdiv_mpa.to_numpy()
        yi[~keep | (tm > 2024.0)] = np.nan            # IMAU termina em 2023,98
        mb_imau[u] = yi
        firn[u] = FACTOR * window_slope(tm, fac[u])     # contribuição do firn em m_b
    rows = []
    for variant in ("A", "B"):
        for label, src in (("GEMB", mb), ("IMAU_2019_23", mb_imau)):
            for a, b in combinations(UNITS, 2):
                ra = anomaly(tm, src[a], variant)[1] if np.isfinite(src[a]).sum() > 6 else src[a]
                rb = anomaly(tm, src[b], variant)[1] if np.isfinite(src[b]).sum() > 6 else src[b]
                ra_ = src[a] - np.polyval(np.polyfit(tm[np.isfinite(src[a])], src[a][np.isfinite(src[a])], 1), tm)
                rb_ = src[b] - np.polyval(np.polyfit(tm[np.isfinite(src[b])], src[b][np.isfinite(src[b])], 1), tm)
                r, p, neff, n = corr_test(ra_, rb_, variant)
                rows.append({"variante": variant, "firn": label, "par": f"{a}-{b}",
                             "r": r, "p_substitutas": p, "N_ef": neff, "N": n})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "coerencia_pares.csv", index=False)
    summ = t.groupby(["variante", "firn"]).agg(r_mediano=("r", "median"),
                                              frac_p_lt_005=("p_substitutas", lambda v: (v < 0.05).mean()),
                                              N_ef_mediano=("N_ef", "median"), N=("N", "median"))
    print(summ.round(2).to_string())

    # GEMB × IMAU par a par, no mesmo período (2019–2023)
    comp = t.pivot_table(index=["variante", "par"], columns="firn", values="r")
    comp["diferenca"] = comp["IMAU_2019_23"] - comp["GEMB"]
    comp.to_csv(OUT / "coerencia_gemb_vs_imau.csv")

    # contribuição do firn e ONI
    rows = []
    for u in UNITS:
        y = mb[u]
        ok = np.isfinite(y)
        yd = y - np.polyval(np.polyfit(tm[ok], y[ok], 1), tm)
        fd = firn[u] - np.polyval(np.polyfit(tm[ok], firn[u][ok], 1), tm)
        r_f, p_f, _, _ = corr_test(yd, fd, "A")
        frac_var = np.nanvar(run_mean(fd, 4)) / np.nanvar(run_mean(yd, 4))
        row = {"unidade": u, "r_mb_vs_firn": r_f, "p_firn": p_f,
               "var_firn_sobre_var_mb": frac_var}
        for lag in range(5):
            o = np.roll(oni, lag)
            o[:lag] = np.nan
            od = o - np.nanmean(o)
            r_o, p_o, _, _ = corr_test(yd, od, "A")
            row[f"r_ONI_lag{lag}"] = r_o
            row[f"p_ONI_lag{lag}"] = p_o
        rows.append(row)
    f = pd.DataFrame(rows)
    f.to_csv(OUT / "firn_e_oni.csv", index=False)
    pd.set_option("display.width", 250)
    print(f.round(2).to_string(index=False))

    # figura
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for i, u in enumerate(UNITS):
        y = mb[u]
        ok = np.isfinite(y)
        an = run_mean(y - np.polyval(np.polyfit(tm[ok], y[ok], 1), tm), 4)
        ax[0].plot(tm, an / np.nanstd(an) + 0 * i, label=u.replace("_", " "))
        fi = run_mean(firn[u] - np.nanmean(firn[u]), 4)
        ax[1].plot(tm, fi, label=u.replace("_", " "))
    ax0b = ax[0].twinx()
    ax0b.plot(tm, oni, "k--", lw=1, label="ONI")
    ax0b.set_ylabel("ONI (°C)")
    ax[0].set_ylabel("anomalia de m_b padronizada")
    ax[1].set_ylabel("+9,34·dFAC/dt, GEMB (m/a)")
    ax[0].legend(fontsize=7, ncol=3)
    ax[1].legend(fontsize=7, ncol=3)
    plt.tight_layout()
    plt.savefig(OUT / "anomalias_mb_firn_oni.png", dpi=120)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
