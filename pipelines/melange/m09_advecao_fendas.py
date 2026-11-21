"""Mélange e fraturas da Thwaites, passo 9: o aumento de fendas na margem da TEIS é dano novo ou advecção?

Geometria (medida, não suposta): as RGT 5 e 1079 cruzam a margem quase PARALELAS ao fluxo (1° e 10°),
em linhas de fluxo separadas ~4,7 km; uma não está a montante da outra. Com trilha ∥ fluxo, uma fenda
que persiste anda ao longo da própria trilha, ~1,2 km/ano — o que permite dois testes:

A. Persistência: para pares de passagens da mesma RGT e do mesmo par de feixes (gt1/gt2/gt3), com os
   traços no solo a ≤ D_TRACO m um do outro, conta as fendas da passagem 2 que caem a ≤ TOL m de uma
   fenda da passagem 1 depois de levar ambas ao referencial do gelo (retroadvecção até T_REF). Compara
   com (i) sem advecção (posição euleriana) e (ii) nulo: deslocamentos aleatórios uniformes ao longo
   da sobreposição. Se as fendas persistem e o campo de velocidade está certo, (A-lagr) ≫ (nulo).
B. Densidade no referencial do gelo: cobertura (células de 5 m com superfície, passo 7) e fendas de
   cada passagem são retroadvectadas até T_REF; conta fendas por km em blocos materiais de 2 km ao
   longo do fluxo, por período. Dano novo → densidade sobe no MESMO bloco material; advecção pura →
   densidade fica no bloco e o aumento euleriano vem de blocos mais densos entrando na faixa da margem.
   IC por bootstrap das passagens.

Velocidade: ITS_LIVE v2 anual (data/velocity_itslive_annual.nc), interpolada no tempo entre os
compostos anuais (a Thwaites acelera; média temporal erraria ~0,1 km/ano). RK2, passo de ~15 dias.
Saídas: outputs/melange/advecao_persistencia.csv, advecao_blocos.csv, advecao_eventos_lagr.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from m07_fendas_atl03 import perfis  # noqa: E402

OUT = ROOT / "outputs" / "melange"
T_REF = 2019.0
TOL, D_TRACO, NNULO = 40.0, 150.0, 200
BLOCO = 2000.0
RNG = np.random.default_rng(20261118)


def t_dec(nome):
    s = nome.split("_")[1]
    t = pd.Timestamp(f"{s[:4]}-{s[4:6]}-{s[6:8]} {s[8:10]}:{s[10:12]}")
    a = t.year
    return a + (t - pd.Timestamp(f"{a}-01-01")).total_seconds() / (pd.Timestamp(f"{a + 1}-01-01")
                                                                  - pd.Timestamp(f"{a}-01-01")).total_seconds()


class Campo:
    """vx, vy (m/ano) interpolados em (t, y, x), compostos anuais centrados no meio do ano."""

    def __init__(self, x, y):
        v = xr.open_dataset(ROOT / "data" / "velocity_itslive_annual.nc")
        sub = v[["vx", "vy"]].sel(x=slice(x.min() - 15e3, x.max() + 15e3),
                                  y=slice(y.min() - 15e3, y.max() + 15e3)).load()
        tt = sub.time.dt.year.values + 0.5
        # lacunas pontuais num ano: preenche com a média dos anos
        for k in ("vx", "vy"):
            a = sub[k].values
            m = np.nanmean(a, axis=0)
            sub[k].values = np.where(np.isfinite(a), a, m[None])
        self.f = {k: RegularGridInterpolator((tt, sub.y.values, sub.x.values), sub[k].values,
                                             bounds_error=False, fill_value=None) for k in ("vx", "vy")}
        self.t0, self.t1 = tt.min(), tt.max()

    def v(self, t, x, y):
        tc = np.clip(t, self.t0, self.t1)
        p = np.c_[tc, y, x]
        return self.f["vx"](p), self.f["vy"](p)

    def leva(self, x, y, t, t_alvo, n=160):
        """integra cada ponto de t até t_alvo (RK2, n passos por ponto)."""
        x, y, t = x.astype(float).copy(), y.astype(float).copy(), np.asarray(t, float).copy()
        h = (t_alvo - t) / n
        for _ in range(n):
            vx, vy = self.v(t, x, y)
            xm, ym = x + 0.5 * h * vx, y + 0.5 * h * vy
            vx2, vy2 = self.v(t + 0.5 * h, xm, ym)
            x, y, t = x + h * vx2, y + h * vy2, t + h
        return x, y


def main():
    ev = pd.read_parquet(OUT / "fendas_atl03_eventos.parquet")
    ev["rgt"] = ev.granulo.str.split("_").str[2].str[:4].astype(int)
    ev["t"] = ev.granulo.map(t_dec)
    ev["par"] = ev.beam.str[2]

    # cobertura (células com superfície) dos perfis aprovados no passo 7, a cada 4 células (20 m)
    cov = []
    for nome, b, ano, mes, sup in perfis():
        s = sup[np.isfinite(sup.h)].iloc[::4]
        cov.append(pd.DataFrame({"granulo": nome, "beam": b, "x": s.x.to_numpy(), "y": s.y.to_numpy(),
                                 "zona": s.zona.to_numpy()}))
    cov = pd.concat(cov, ignore_index=True)
    cov["rgt"] = cov.granulo.str.split("_").str[2].str[:4].astype(int)
    cov["t"] = cov.granulo.map(t_dec)
    cov["par"] = cov.beam.str[2]
    aprov = set(zip(cov.granulo, cov.beam))
    ev = ev[[k in aprov for k in zip(ev.granulo, ev.beam)]].copy()
    print(f"{len(ev)} fendas em {cov.groupby(['granulo', 'beam']).ngroups} perfis aprovados")

    campo = Campo(pd.concat([ev.x, cov.x]).to_numpy(), pd.concat([ev.y, cov.y]).to_numpy())
    ev["x0"], ev["y0"] = campo.leva(ev.x.to_numpy(), ev.y.to_numpy(), ev.t.to_numpy(), T_REF)
    cov["x0"], cov["y0"] = campo.leva(cov.x.to_numpy(), cov.y.to_numpy(), cov.t.to_numpy(), T_REF)

    # eixo ao longo do fluxo (direção média da velocidade nas margens) e transversal
    vx, vy = campo.v(np.full(len(ev), 2022.0), ev.x.to_numpy(), ev.y.to_numpy())
    u = np.array([np.nanmean(vx), np.nanmean(vy)]); u /= np.hypot(*u)
    nrm = np.array([-u[1], u[0]])
    o = np.array([-1551.2e3, -439.5e3])
    for d in (ev, cov):
        d["s"] = (d.x - o[0]) * u[0] + (d.y - o[1]) * u[1]          # euleriano, ao longo do fluxo
        d["n"] = (d.x - o[0]) * nrm[0] + (d.y - o[1]) * nrm[1]
        d["s0"] = (d.x0 - o[0]) * u[0] + (d.y0 - o[1]) * u[1]       # material (posição em T_REF)
        d["n0"] = (d.x0 - o[0]) * nrm[0] + (d.y0 - o[1]) * nrm[1]
    ev.to_parquet(OUT / "advecao_eventos_lagr.parquet", index=False)

    # ---------------------------------------------------------------- A. persistência
    lin = []
    perf = cov.groupby(["granulo", "beam"])
    chaves = list(perf.groups)
    meta = pd.DataFrame([(g, b, int(g.split("_")[2][:4]), b[2], cov.t[perf.groups[(g, b)][0]]) for g, b in chaves],
                        columns=["granulo", "beam", "rgt", "par", "t"])
    for (r, p), m in meta.groupby(["rgt", "par"]):
        m = m.sort_values("t").reset_index(drop=True)
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                a, b = m.loc[i], m.loc[j]
                dt = b.t - a.t
                if dt > 1.6:
                    continue
                ca, cb = perf.get_group((a.granulo, a.beam)), perf.get_group((b.granulo, b.beam))
                ca, cb = ca[ca.zona.isin(["margem", "meio"])], cb[cb.zona.isin(["margem", "meio"])]
                if len(ca) < 50 or len(cb) < 50:
                    continue
                # distância entre os traços no solo (euleriana): mediana do vizinho mais próximo
                dtr = np.median(cKDTree(ca[["x", "y"]].to_numpy()).query(cb[["x", "y"]].to_numpy())[0])
                if dtr > D_TRACO:
                    continue
                ea = ev[(ev.granulo == a.granulo) & (ev.beam == a.beam) & ev.zona.isin(["margem", "meio"])]
                eb = ev[(ev.granulo == b.granulo) & (ev.beam == b.beam) & ev.zona.isin(["margem", "meio"])]
                if len(ea) < 3 or len(eb) < 3:
                    continue
                for ref, col in [("lagrangiano", "s0"), ("euleriano", "s")]:
                    sa = np.sort(ea[col].to_numpy()); sb = eb[col].to_numpy()
                    lo_, hi_ = max(ca[col].min(), cb[col].min()), min(ca[col].max(), cb[col].max())
                    sb = sb[(sb > lo_) & (sb < hi_)]
                    if sb.size < 3:
                        continue
                    tr = cKDTree(sa[:, None])
                    obs = (tr.query(sb[:, None])[0] <= TOL).mean()
                    nul = np.array([(tr.query(((sb - lo_ + RNG.uniform(0, hi_ - lo_)) % (hi_ - lo_) + lo_)[:, None])[0]
                                     <= TOL).mean() for _ in range(NNULO)])
                    lin.append(dict(rgt=r, par=p, g1=a.granulo, g2=b.granulo, dt=dt, d_traco=dtr, ref=ref,
                                    n2=sb.size, frac=obs, nulo=nul.mean(), nulo_p95=np.percentile(nul, 95)))
    A = pd.DataFrame(lin)
    A.to_csv(OUT / "advecao_persistencia.csv", index=False)
    print("\nA. persistência (fração de fendas da passagem 2 a ≤ %.0f m de uma da passagem 1)" % TOL)
    print(A.groupby("ref")[["frac", "nulo"]].agg(["mean", "count"]).round(3).to_string())
    for ref, g in A.groupby("ref"):
        k = g.frac.to_numpy() - g.nulo.to_numpy()
        bt = [k[RNG.integers(0, len(k), len(k))].mean() for _ in range(2000)]
        print(f"  {ref}: excesso sobre o nulo {k.mean():.3f} [IC95 {np.percentile(bt, 2.5):.3f}, "
              f"{np.percentile(bt, 97.5):.3f}], pares {len(k)}, acima do p95 do nulo em "
              f"{(g.frac > g.nulo_p95).mean():.0%}")
    print(A[A.ref == "lagrangiano"].groupby(pd.cut(A[A.ref == "lagrangiano"].dt, [0, 0.3, 0.6, 1.0, 1.6]),
                                             observed=True)[["frac", "nulo"]].mean().round(3).to_string())

    # ---------------------------------------------------------------- B. blocos materiais
    per = lambda t: np.where(t < 2022, "2018–21", np.where(t < 2024, "2022–23", "2024–25"))
    for d in (ev, cov):
        d["per"] = per(d.t)
        d["bloco"] = np.floor(d.s0 / BLOCO).astype(int)
        d["bloco_e"] = np.floor(d.s / BLOCO).astype(int)
    lin = []
    for r in (5, 1079):
        for ref, col in [("material", "bloco"), ("euleriano", "bloco_e")]:
            c = cov[(cov.rgt == r) & cov.zona.isin(["margem"])]
            e = ev[(ev.rgt == r) & ev.zona.isin(["margem"])]
            km = c.groupby([col, "per", "granulo"]).size().mul(0.02).rename("km")
            n = e.groupby([col, "per", "granulo"]).size().rename("n")
            t = pd.concat([km, n], axis=1).fillna({"n": 0}).dropna(subset=["km"]).reset_index()
            for (bl, pe), g in t.groupby([col, "per"]):
                if g.km.sum() < 3 or g.granulo.nunique() < 2:
                    continue
                kk, nn = g.km.to_numpy(), g.n.to_numpy()
                bt = []
                for _ in range(500):
                    ii = RNG.integers(0, len(g), len(g))
                    bt.append(nn[ii].sum() / kk[ii].sum())
                lin.append(dict(rgt=r, ref=ref, bloco_km=bl * BLOCO / 1000, periodo=pe, km=kk.sum(),
                                n=int(nn.sum()), dens=nn.sum() / kk.sum(), lo=np.percentile(bt, 2.5),
                                hi=np.percentile(bt, 97.5), passagens=g.granulo.nunique()))
    B = pd.DataFrame(lin)
    B.to_csv(OUT / "advecao_blocos.csv", index=False)

    # ---------------------------------------------------------------- C. previsão por advecção pura (2D em s0 × n0)
    # densidade de 2024–25 na faixa euleriana da margem PREVISTA pelo campo material 2D de 2018–21
    # (bins 2D de 500 m ao longo do fluxo × 300 m transversais, sem agrupar por número de par — imune a yaw flip).
    # Garante que apenas o mesmo filete de gelo (desvio lateral ≤ 300 m) seja comparado entre épocas.
    BIN_S, BIN_N = 500.0, 300.0
    lin_2d = []
    for r in (5, 1079):
        c0 = cov[(cov.rgt == r) & (cov.per == "2018–21")]
        e0 = ev[(ev.rgt == r) & (ev.per == "2018–21")]
        c1 = cov[(cov.rgt == r) & (cov.per == "2024–25") & (cov.zona == "margem")]
        e1 = ev[(ev.rgt == r) & (ev.per == "2024–25") & (ev.zona == "margem")]
        if c0.empty or c1.empty:
            continue
        g0, g1 = c0.granulo.unique(), c1.granulo.unique()
        reps = []
        for k in range(501):
            s0_g = g0 if k == 0 else RNG.choice(g0, len(g0))
            s1_g = g1 if k == 0 else RNG.choice(g1, len(g1))
            cc0 = pd.concat([c0[c0.granulo == g] for g in s0_g])
            ee0 = pd.concat([e0[e0.granulo == g] for g in s0_g])
            cc1 = pd.concat([c1[c1.granulo == g] for g in s1_g])
            ee1 = pd.concat([e1[e1.granulo == g] for g in s1_g])
            b0 = np.floor(cc0.s0 / BIN_S).astype(int).astype(str) + "_" + np.floor(cc0.n0 / BIN_N).astype(int).astype(str)
            be0 = np.floor(ee0.s0 / BIN_S).astype(int).astype(str) + "_" + np.floor(ee0.n0 / BIN_N).astype(int).astype(str)
            km0 = b0.value_counts() * 0.02
            n0 = be0.value_counts()
            dens0 = (n0.reindex(km0.index, fill_value=0) / km0)[km0 >= 0.1]
            b1 = np.floor(cc1.s0 / BIN_S).astype(int).astype(str) + "_" + np.floor(cc1.n0 / BIN_N).astype(int).astype(str)
            be1 = np.floor(ee1.s0 / BIN_S).astype(int).astype(str) + "_" + np.floor(ee1.n0 / BIN_N).astype(int).astype(str)
            ok = b1.isin(dens0.index).to_numpy()
            km_ok = ok.sum() * 0.02
            if km_ok < 1:
                reps.append((np.nan, np.nan, km_ok, ok.mean()))
                continue
            prev = dens0.reindex(b1[ok]).sum() * 0.02 / km_ok
            obs = be1.isin(dens0.index).sum() / km_ok
            reps.append((prev, obs, km_ok, ok.mean()))
        reps = np.array(reps, float)
        d = reps[1:, 1] - reps[1:, 0]
        lin_2d.append(dict(rgt=r, bin_s_m=BIN_S, bin_n_m=BIN_N, km_2425_vistos=reps[0, 2], frac_vista=reps[0, 3],
                           previsto=reps[0, 0], observado=reps[0, 1], obs_menos_prev=reps[0, 1] - reps[0, 0],
                           lo=np.nanpercentile(d, 2.5), hi=np.nanpercentile(d, 97.5),
                           pass_1821=len(g0), pass_2425=len(g1)))
    C2D = pd.DataFrame(lin_2d)
    C2D.to_csv(OUT / "advecao_previsao_2d.csv", index=False)
    C2D.to_csv(ROOT / "manuscrito" / "figuras" / "dados" / "melange" / "advecao_previsao_2d.csv", index=False)
    print("\nC (2D, s0×n0 = 500×300 m, imune a yaw flip): 2024–25 na margem previsto × observado (fendas/km)")
    print(C2D.round(3).to_string(index=False))

    # legado 1D mantido apenas para comparação/auditoria
    BIN = 500.0
    lin = []
    for r in (5, 1079):
        for p in ("1", "2", "3"):
            c0 = cov[(cov.rgt == r) & (cov.par == p) & (cov.per == "2018–21")]
            e0 = ev[(ev.rgt == r) & (ev.par == p) & (ev.per == "2018–21")]
            c1 = cov[(cov.rgt == r) & (cov.par == p) & (cov.per == "2024–25") & (cov.zona == "margem")]
            e1 = ev[(ev.rgt == r) & (ev.par == p) & (ev.per == "2024–25") & (ev.zona == "margem")]
            if c0.empty or c1.empty:
                continue
            g0, g1 = c0.granulo.unique(), c1.granulo.unique()
            reps = []
            for k in range(501):
                s0 = g0 if k == 0 else RNG.choice(g0, len(g0))
                s1 = g1 if k == 0 else RNG.choice(g1, len(g1))
                cc0 = pd.concat([c0[c0.granulo == g] for g in s0]); ee0 = pd.concat([e0[e0.granulo == g] for g in s0])
                cc1 = pd.concat([c1[c1.granulo == g] for g in s1]); ee1 = pd.concat([e1[e1.granulo == g] for g in s1])
                km0 = (np.floor(cc0.s0 / BIN)).value_counts() * 0.02
                n0 = (np.floor(ee0.s0 / BIN)).value_counts()
                dens0 = (n0.reindex(km0.index, fill_value=0) / km0)[km0 >= 0.1]
                b1 = np.floor(cc1.s0 / BIN)
                ok = b1.isin(dens0.index).to_numpy()
                km_ok = ok.sum() * 0.02
                if km_ok < 1:
                    reps.append((np.nan, np.nan, 0, 0)); continue
                prev = dens0.reindex(b1[ok]).sum() * 0.02 / km_ok
                eb = np.floor(ee1.s0 / BIN)
                obs = eb.isin(dens0.index).sum() / km_ok
                reps.append((prev, obs, km_ok, ok.mean()))
            reps = np.array(reps, float)
            d = reps[1:, 1] - reps[1:, 0]
            lin.append(dict(rgt=r, par=p, km_2425_vistos=reps[0, 2], frac_vista=reps[0, 3],
                            previsto=reps[0, 0], observado=reps[0, 1], obs_menos_prev=reps[0, 1] - reps[0, 0],
                            lo=np.nanpercentile(d, 2.5), hi=np.nanpercentile(d, 97.5),
                            pass_1821=len(g0), pass_2425=len(g1)))
    C = pd.DataFrame(lin)
    C.to_csv(OUT / "advecao_previsao.csv", index=False)
    print("\nC. 2024–25 na margem: previsto por advecção pura do campo de 2018–21 × observado (fendas/km)")
    print(C.round(2).to_string())
    with pd.option_context("display.width", 220, "display.max_rows", 300):
        for (r, ref), g in B.groupby(["rgt", "ref"]):
            print(f"\nB. RGT {r} — {ref} (bloco = início, km ao longo do fluxo; fendas/km)")
            print(g.pivot_table(index="bloco_km", columns="periodo", values="dens").round(2).to_string())


if __name__ == "__main__":
    main()
