"""Canais basais (ASE, ICESat-2), passo 3: séries de profundidade dos canais e tendência/ciclo sazonal.

Para cada seção (canal × trilha × feixe), série D(t) a cada ~91 dias (passo 2). Modelo:
    D(t) = a + b·(t − 2022) + c·cos 2πt + d·sin 2πt + ε
ajustado por Huber (robusto a ciclos com nuvem/neve soprada). b > 0: o canal se aprofunda em relação
às quilhas (bordo livre); em espessura hidrostática, ×ρw/(ρw−ρi) = 9,34 (ρw 1027, ρi 917; firn
diferencial desprezado — ver limitações). O ciclo sazonal é t em ano juliano a partir de 2018 (é a
convenção do extrator), o que desloca a fase em < 1 dia.

Independência: os dois feixes de um par (~90 m) veem o mesmo canal; a unidade é o par
(beam 1-2, 3-4, 5-6) → média dos feixes por ciclo. Seções de trilhas vizinhas sobre o mesmo canal
não são independentes: a agregação por plataforma usa bootstrap em blocos espaciais de 10 km.
IC de cada seção: bootstrap por blocos móveis de 4 ciclos (~1 ano), 500 réplicas.
Saída: outputs/canais/tendencias_secoes.csv, resumo_plataformas.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gl_migration"))
from p03_migracao import huber  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "data" / "interim" / "canais"
OUT = ROOT / "outputs" / "canais"
RNG = np.random.default_rng(20261117)
K_HID = 1027.0 / (1027.0 - 917.0)
MIN_N = 12
NB = 500


def X_de(t):
    return np.column_stack([np.ones_like(t), t - 2022.0, np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)])


def ajusta(t, D):
    return huber(X_de(t), D)


def boot_blocos(t, D, L=4):
    n = len(t)
    o = np.argsort(t)
    t, D = t[o], D[o]
    nb = int(np.ceil(n / L))
    reps = []
    for _ in range(NB):
        ini = RNG.integers(0, n - L + 1, nb)
        idx = np.concatenate([np.arange(i, i + L) for i in ini])[:n]
        try:
            reps.append(ajusta(t[idx], D[idx]))
        except Exception:
            pass
    return np.array(reps)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = pd.read_parquet(IN / "secoes.parquet")
    s["par"] = (s.beam + 1) // 2
    # canal identificado por feixe; une os feixes de um par pelo centro (< 500 m) — aqui, pelo id do
    # par + posição empilhada arredondada a 1 km
    e = pd.read_parquet(IN / "canais_validados.parquet")
    e = e[e.canal_basal]
    e["par"] = (e.beam + 1) // 2
    e["chave"] = e.plataforma + "_" + e.rgt.astype(str) + "_" + e.par.astype(str) + "_" + (e.b0 // 10).astype(str)
    s = s.merge(e[["canal", "chave", "prom", "largura_m", "vel", "dDds_rema", "theta"]], on="canal")
    u = s.groupby(["chave", "plataforma", "cycle"]).agg(t=("t", "median"), D=("D", "mean"), x=("x", "median"),
                                                        y=("y", "median"), desloc=("desloc_m", "mean"),
                                                        prom=("prom", "first"), larg=("largura_m", "first"),
                                                        vel=("vel", "first"), grad=("dDds_rema", "mean"),
                                                        theta=("theta", "first")).reset_index()
    linhas = []
    for ch, g in u.groupby("chave"):
        g = g.sort_values("t")
        t, D = g.t.to_numpy(), g.D.to_numpy()
        med, mad = np.median(D), 1.4826 * np.median(np.abs(D - np.median(D)))
        ok = np.abs(D - med) <= 4 * max(mad, 0.05)
        t, D = t[ok], D[ok]
        if len(t) < MIN_N or (t.max() - t.min()) < 3:
            continue
        b = ajusta(t, D)
        r = boot_blocos(t, D)
        res = D - X_de(t) @ b
        # nulo do ciclo sazonal: a amplitude estimada de ruído puro é positiva (Rayleigh); permutar os
        # resíduos da tendência entre as datas dá a distribuição sob ausência de ciclo
        Xt = X_de(t)[:, :2]
        bt = np.linalg.lstsq(Xt, D, rcond=None)[0]
        rt = D - Xt @ bt
        amp0 = np.hypot(b[2], b[3])
        nulo = []
        for _ in range(200):
            Dp = Xt @ bt + RNG.permutation(rt)
            bp = ajusta(t, Dp)
            nulo.append(np.hypot(bp[2], bp[3]))
        p_saz = (np.sum(np.array(nulo) >= amp0) + 1) / (len(nulo) + 1)
        linhas.append(dict(chave=ch, plataforma=g.plataforma.iloc[0], x=g.x.median(), y=g.y.median(), n=len(t),
                           t0=t.min(), t1=t.max(), D_med=np.median(D), prom=g.prom.iloc[0], larg=g.larg.iloc[0],
                           dDdt=b[1], dDdt_lo=np.percentile(r[:, 1], 2.5), dDdt_hi=np.percentile(r[:, 1], 97.5),
                           amp_saz=np.hypot(b[2], b[3]), fase_dia=(np.degrees(np.arctan2(b[3], b[2])) % 360) / 360 * 365.25,
                           amp_lo=np.percentile(np.hypot(r[:, 2], r[:, 3]), 2.5),
                           amp_hi=np.percentile(np.hypot(r[:, 2], r[:, 3]), 97.5),
                           rms_res=np.sqrt(np.mean(res ** 2)), p_saz=p_saz, vel=g.vel.iloc[0],
                           grad=g.grad.iloc[0], theta=g.theta.iloc[0],
                           migr_m_ano=np.polyfit(g.t, g.desloc, 1)[0]))
    d = pd.DataFrame(linhas)
    # Euleriano → Lagrangiano: numa seção fixa, D muda também porque o fluxo traz a geometria de
    # montante: ∂D/∂t|E = ∂D/∂t|L − u·∂D/∂s. Termo advectivo com o gradiente do REMA (geometria de
    # ~2010s, pode estar defasada) e a velocidade ITS_LIVE média 2019–2025.
    d["adv"] = -d.vel * d.grad
    d["dDdt_L"] = d.dDdt - d.adv
    d["dHdt_m_ano"] = d.dDdt * K_HID
    d["dHdt_L_m_ano"] = d.dDdt_L * K_HID
    d.to_csv(OUT / "tendencias_secoes.csv", index=False)
    d["bloco"] = (d.x // 10000).astype(int).astype(str) + "_" + (d.y // 10000).astype(int).astype(str)
    res = []
    for pl, g in d.groupby("plataforma"):
        bl = g.bloco.unique()
        reps = []
        for _ in range(2000):
            bb = RNG.choice(bl, len(bl))
            q = pd.concat([g[g.bloco == b] for b in bb])
            reps.append((q.dDdt.median(), q.amp_saz.median(), q.dDdt_L.median()))
        reps = np.array(reps)
        res.append(dict(plataforma=pl, n_secoes=len(g), n_blocos=len(bl), D_med=g.D_med.median(),
                        dDdt_L=g.dDdt_L.median(), dDdt_L_lo=np.percentile(reps[:, 2], 2.5),
                        dDdt_L_hi=np.percentile(reps[:, 2], 97.5), adv_med=g.adv.median(),
                        frac_saz_p05=(g.p_saz < 0.05).mean(),
                        dDdt=g.dDdt.median(), dDdt_lo=np.percentile(reps[:, 0], 2.5), dDdt_hi=np.percentile(reps[:, 0], 97.5),
                        frac_aprofunda=(g.dDdt_lo > 0).mean(), frac_rasa=(g.dDdt_hi < 0).mean(),
                        amp_saz=g.amp_saz.median(), amp_lo=np.percentile(reps[:, 1], 2.5), amp_hi=np.percentile(reps[:, 1], 97.5),
                        rms_res=g.rms_res.median()))
    r = pd.DataFrame(res)
    r["dHdt"] = r.dDdt * K_HID
    r.to_csv(OUT / "resumo_plataformas.csv", index=False)
    with pd.option_context("display.width", 220):
        print(r.round(3).to_string())


if __name__ == "__main__":
    main()
