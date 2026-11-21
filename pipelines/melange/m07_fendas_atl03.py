"""Mélange e fraturas da Thwaites, passo 7: fendas no ATL03 (margem de cisalhamento × interior da TEIS).

Entrada: data/interim/melange/atl03/*.parquet (passo 6; fótons de sinal, conf ≥ 2), trilhas RGT 1079 e 5.
Só feixes fortes (densidade de fótons suficiente para células de 5 m). Por passagem e feixe:
  1. projeta em EPSG:3031 e ordena pelo tempo; s = distância acumulada ao longo da trilha;
  2. perfil da superfície em células de 5 m: percentil 90 do h dos fótons (topo; fótons das paredes e do
     fundo de fendas ficam abaixo);
  3. resíduo r = perfil − mediana móvel de 300 m (61 células);
  4. σ = 1,4826·MAD de r no interior da TEIS (todas as passagens), calculado antes da detecção;
  Controle de qualidade: passagem×feixe entra só com mediana ≥ 10 fótons/célula e ≤ 20% de células com < 5
  fótons (nuvem); células com < 5 fótons ficam sem superfície; lacuna só conta com ≥ 10 fótons nas duas bordas.
  fenda = (a) sequência de células com r < −max(4σ, 1,5 m) e largura 10–300 m, ou (b) lacuna sem fótons
  de sinal de 10–300 m entre células com superfície (paredes íngremes não devolvem fótons). Lacunas maiores
  (nuvem, sombra) ficam fora.
Zonas (domínio do produto basal v2, 2 km): margem = TEIS a ≤ 5 km da TWGT; interior = TEIS a > 10 km;
meio; TWGT. Densidade = fendas por km de perfil com superfície, por zona e ano; IC 95% por bootstrap das
passagens (unidade independente = passagem, não a célula).
Saídas: outputs/melange/fendas_atl03_eventos.parquet, fendas_atl03_serie.csv
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "data" / "interim" / "melange" / "atl03"
OUT = ROOT / "outputs" / "melange"
CEL, JAN, K, PROF_MIN = 5.0, 61, 4.0, 1.5
N_CEL, N_BORDA, N_MED_PASS, VAZIO_MAX = 5, 10, 10, 0.20
RNG = np.random.default_rng(20261118)


def zonas():
    dom = pd.read_csv(ROOT / "manuscrito" / "figuras" / "dados" / "R_fig1_dominio_basal_2km.csv")
    dom = dom[dom.unidade.str.startswith("Thwaites")]
    tw = dom[dom.unidade == "Thwaites_TWGT"][["x_km", "y_km"]].to_numpy() * 1000
    te = dom[dom.unidade == "Thwaites_TEIS"][["x_km", "y_km"]].to_numpy() * 1000
    d_tw = cKDTree(tw).query(te)[0] / 1000
    lab = np.concatenate([np.where(d_tw <= 5, "margem", np.where(d_tw > 10, "interior", "meio")),
                         np.full(len(tw), "TWGT")])
    return cKDTree(np.vstack([te, tw])), lab


def perfis():
    tree, lab = zonas()
    tr = Transformer.from_crs(4326, 3031, always_xy=True)
    for f in sorted(glob.glob(str(DIR / "*.parquet"))):
        nome = Path(f).stem
        data = nome.split("_")[1]
        d = pd.read_parquet(f, columns=["lat", "lon", "h", "dt", "beam", "forte"])
        d = d[d.forte]
        if d.empty:
            continue
        x, y = tr.transform(d.lon.to_numpy(), d.lat.to_numpy())
        dist, ii = tree.query(np.c_[x, y])
        d = d.assign(x=x, y=y, zona=np.where(dist <= 1500, lab[ii], ""))
        d = d[d.zona != ""]
        for b, p in d.groupby("beam"):
            p = p.sort_values("dt")
            s = np.r_[0, np.cumsum(np.hypot(np.diff(p.x.to_numpy()), np.diff(p.y.to_numpy())))]
            p = p.assign(c=np.floor(s / CEL).astype(int))
            sup = p.groupby("c").agg(h=("h", lambda v: np.percentile(v, 90)), x=("x", "mean"), y=("y", "mean"),
                                     zona=("zona", "first"), n=("h", "size"))
            sup = sup.reindex(np.arange(sup.index.min(), sup.index.max() + 1))
            sup["n"] = sup.n.fillna(0)
            # controle de qualidade da passagem (nuvem): mediana de fótons por célula e fração vazia
            if sup.n[sup.n > 0].median() < N_MED_PASS or (sup.n < N_CEL).mean() > VAZIO_MAX:
                continue
            sup.loc[sup.n < N_CEL, "h"] = np.nan
            sup["zona"] = sup.zona.ffill().bfill()
            med = sup.h.rolling(JAN, center=True, min_periods=20).median()
            sup["r"] = sup.h - med
            yield nome, b, int(data[:4]), int(data[4:6]), sup


def main():
    perf = list(perfis())
    rint = np.concatenate([s.loc[s.zona == "interior", "r"].dropna().to_numpy() for *_, s in perf])
    sig = 1.4826 * np.median(np.abs(rint - np.median(rint)))
    lim = max(K * sig, PROF_MIN)
    print(f"{len(perf)} perfis (passagem × feixe forte); σ no interior da TEIS {sig:.2f} m -> limiar {lim:.2f} m")
    eventos, comp = [], []
    for nome, b, ano, mes, sup in perf:
        hh, r = sup.h.to_numpy(), sup.r.to_numpy()
        low = np.isfinite(r) & (r < -lim)
        idx = np.flatnonzero(low)
        if idx.size:
            for gi in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
                w = (len(gi) + 1) * CEL
                if 10 <= w <= 300:
                    c = gi[np.argmin(r[gi])]
                    eventos.append((nome, b, ano, mes, "depressão", sup.x.iat[c], sup.y.iat[c], sup.zona.iat[c], -r[c], w))
        idx = np.flatnonzero(~np.isfinite(hh))
        if idx.size:
            for gi in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
                w = len(gi) * CEL
                if 10 <= w <= 300 and gi[0] > 0 and gi[-1] < len(hh) - 1:
                    c0, c1 = gi[0] - 1, gi[-1] + 1
                    if min(sup.n.iat[c0], sup.n.iat[c1]) < N_BORDA:
                        continue
                    eventos.append((nome, b, ano, mes, "lacuna", 0.5 * (sup.x.iat[c0] + sup.x.iat[c1]),
                                    0.5 * (sup.y.iat[c0] + sup.y.iat[c1]), sup.zona.iat[c0], np.nan, w))
        km = sup[np.isfinite(sup.h)].groupby("zona").size() * CEL / 1000
        for z, k in km.items():
            comp.append((nome, b, ano, mes, z, k))
    ev = pd.DataFrame(eventos, columns=["granulo", "beam", "ano", "mes", "tipo", "x", "y", "zona", "prof", "largura"])
    cp = pd.DataFrame(comp, columns=["granulo", "beam", "ano", "mes", "zona", "km"])
    ev.to_parquet(OUT / "fendas_atl03_eventos.parquet", index=False)
    cp.to_parquet(OUT / "fendas_atl03_comprimento.parquet", index=False)
    print("limiar de profundidade:", round(lim, 2), "m")
    lin = []
    for (z, a), c in cp.groupby(["zona", "ano"]):
        e = ev[(ev.zona == z) & (ev.ano == a)]
        for tipo in ["depressão", "lacuna", "ambos"]:
            ee = e if tipo == "ambos" else e[e.tipo == tipo]
            nr = ee.groupby("granulo").size().reindex(c.granulo.unique(), fill_value=0)
            kr = c.groupby("granulo").km.sum()
            u = kr.index.to_numpy()
            bt = [nr.loc[s].sum() / kr.loc[s].sum() for s in (RNG.choice(u, len(u)) for _ in range(500))]
            lin.append(dict(zona=z, ano=a, tipo=tipo, km=kr.sum(), n=int(nr.sum()), dens=nr.sum() / kr.sum(),
                            lo=np.percentile(bt, 2.5), hi=np.percentile(bt, 97.5), n_passagens=len(u)))
    s = pd.DataFrame(lin)
    s.to_csv(OUT / "fendas_atl03_serie.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(s[s.tipo != "lacuna"].round(3).to_string())


if __name__ == "__main__":
    main()
