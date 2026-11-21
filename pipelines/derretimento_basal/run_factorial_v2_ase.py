"""Fatorial ampliado da incerteza estrutural (substitui ``run_factorial_uncertainty_ase.py``).

Usa as componentes por pixel no domínio união (``run_union_components_ase.py``). Para cada
combinação de escolhas e cada unidade:

    m_b(pixel) = SMB [+ (RACMO − GEMB) da unidade] − DH/Dt − H_firn·∇·u_σ − Δ_firn
    média por área no domínio da combinação

Fatores (fatores estruturais de definição de domínio):
- máscara f: f080, f090, f095, IC (d\\* pela resposta à maré);
- distância à frente: 1,0 / 1,5 / 2,0 km;
- perda de gelo máxima: 5% / 10% / 20%;
- corte pelo radar MCoRDS normalizado por voo: desvio aceito 2% / 3% / 5% → d\\*_radar por
  plataforma, derivado do perfil ``perfil_razao_normalizado_por_voo.csv``;
- escala de ∇·u: 1, 2, 3, 5 km;
- SMB: GEMB × RACMO (diferença de 2019–2022 por unidade);
- firn: GEMB × IMAU. H·∇·u usa o H do próprio firn em cada pixel (não aditivo). O DH/Dt do IMAU
  só existe em 2019–2023: Δ_firn = (tendência IMAU − tendência GEMB) em 2019–2023, medida **na
  série da própria variante de máscara**. É a única parcela transferida entre períodos.

Fixo e declarado: d\\*_recente (linhas de 2019–2025), a agregação (termo separado) e a velocidade
(termo separado, MEaSUREs × ITS_LIVE).

Total: soma das unidades na **mesma** combinação (escolhas comuns a todas as plataformas).
Saída: ``outputs/basal_melt_ase/fatorial_v2_*.csv``.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
O = ROOT / "outputs"
COMP = ROOT / "data" / "interim" / "basal_melt" / "componentes_uniao_500m.nc"
RADAR = O / "mcords_hydro_test" / "perfil_razao_normalizado_por_voo.csv"
UNITS = ["Pine_Island", "Thwaites_TEIS", "Crosson", "Dotson", "Cosgrove"]
BASE = {u: u.replace("_TEIS", "").replace("_TWGT", "") for u in UNITS}
D_REC = {"Pine_Island": 3.0, "Thwaites_TEIS": 1.5, "Crosson": 3.0, "Dotson": 2.5, "Cosgrove": 2.0}
VARS = {"f090": "classe", "f080": "classe_f080", "f095": "classe_f095", "ic": "classe_ic"}
LEVELS = {"mascara": list(VARS), "frente_km": [1.0, 1.5, 2.0], "perda_max": [0.05, 0.10, 0.20],
          "radar_tol": [0.02, 0.03, 0.05], "escala_km": [1, 2, 3, 5], "smb": ["GEMB", "RACMO"],
          "firn": ["GEMB", "IMAU"]}
CENTRAL = {"mascara": "f090", "frente_km": 1.5, "perda_max": 0.10, "radar_tol": 0.03,
           "escala_km": None, "smb": "GEMB", "firn": "GEMB"}
OPT = {"Pine_Island": 3, "Thwaites_TEIS": 2, "Crosson": 1, "Dotson": 1, "Cosgrove": 1}
RHO_I, PIX = 917.0, 0.25e6


def radar_cut(prof, plat, tol):
    """Início da 1ª faixa a partir da qual todas as faixas até 12 km ficam dentro de 1 ± tol."""
    g = prof[prof.plataforma == plat].copy()
    g["lo"] = g.dist_gl_km.str.extract(r"\(([-\d.]+),").astype(float)
    g = g[g.lo >= 0].sort_values("lo")
    ok = (g.razao_norm_med - 1).abs().to_numpy() <= tol
    for i in range(len(ok)):
        if ok[i:].all():
            return float(g.lo.to_numpy()[i])
    return float(g.lo.max())


def trend(g, col, period):
    g = g[(g.ano >= period[0]) & (g.ano <= period[1])]
    y = g[col].to_numpy()
    ok = np.isfinite(y)
    return np.polyfit(g.t_meio.to_numpy()[ok], y[ok], 1)[0]


def main() -> int:
    c = xr.open_dataset(COMP)
    u = c.unidade.values
    smb_diff = pd.read_csv(O / "basal_melt_ase" / "smb_gemb_racmo.csv").set_index("unidade")["diff"]
    prof = pd.read_csv(RADAR)
    rcut = {(p, tol): radar_cut(prof, p, tol) for p in set(BASE.values())
            for tol in LEVELS["radar_tol"]}
    ser = {k: pd.read_csv(O / "thickness_ase" / f"serie_espessura_{v}.csv") for k, v in VARS.items()}
    dfirn = {}
    for k, s in ser.items():
        for un in UNITS:
            g = s[s.plataforma == un]
            dfirn[(k, un)] = (trend(g, "dH_lagr_imau_m", (2019, 2023))
                              - trend(g, "dH_lagr_m", (2019, 2023)))
    rows = []
    for un in UNITS:
        s = u == un
        P = {k: c[k].values[s] for k in ("DHDt", "H_gemb", "H_imau", "SMB", "perda_gelo_frac",
                                         "frente_dist_km", "dist_LA_BedMachine_km",
                                         "dist_LA_recente_km")}
        div = {sc: c[f"div_{sc}km"].values[s] for sc in LEVELS["escala_km"]}
        cls = {k: c[v].values[s] == 1 for k, v in VARS.items()}
        base_ok = (P["dist_LA_recente_km"] >= D_REC[un]) & np.isfinite(P["DHDt"]) & \
            np.isfinite(P["SMB"])
        for combo in itertools.product(*LEVELS.values()):
            f = dict(zip(LEVELS, combo))
            H = P["H_imau"] if f["firn"] == "IMAU" else P["H_gemb"]
            front_ok = (P["frente_dist_km"] > f["frente_km"]) & ~(P["perda_gelo_frac"] > f["perda_max"])
            dom = cls[f["mascara"]] & base_ok & front_ok & \
                (P["dist_LA_BedMachine_km"] >= rcut[(BASE[un], f["radar_tol"])]) & np.isfinite(H) & \
                np.isfinite(div[f["escala_km"]])
            if dom.sum() < 20:
                continue
            mb = P["SMB"][dom] - P["DHDt"][dom] - H[dom] * div[f["escala_km"]][dom]
            if f["smb"] == "RACMO":
                mb = mb + smb_diff[un]
            if f["firn"] == "IMAU":
                mb = mb - dfirn[(f["mascara"], un)]
            mean = float(np.mean(mb))
            rows.append({"unidade": un, **f, "mb_mpa": mean, "area_km2": dom.sum() * PIX / 1e6,
                         "Gt": mean * dom.sum() * PIX * RHO_I / 1e12})
    t = pd.DataFrame(rows)
    t.to_csv(O / "basal_melt_ase" / "fatorial_v2_combinacoes.csv", index=False)

    # resumo por unidade e efeito principal de cada fator
    summ = t.groupby("unidade").agg(mb_mediana=("mb_mpa", "median"), mb_min=("mb_mpa", "min"),
                                    mb_max=("mb_mpa", "max"), mb_p05=("mb_mpa", lambda v: v.quantile(.05)),
                                    mb_p95=("mb_mpa", lambda v: v.quantile(.95)), n=("mb_mpa", "size"))
    for fac in LEVELS:
        summ[f"efeito_{fac}"] = t.groupby(["unidade", fac]).mb_mpa.mean().groupby("unidade").agg(
            lambda v: v.max() - v.min())
    cen = t
    for k, v in CENTRAL.items():
        if v is not None:
            cen = cen[cen[k] == v]
    cen = cen[cen.apply(lambda r: r.escala_km == OPT[r.unidade], axis=1)].set_index("unidade")
    summ["mb_central"] = cen.mb_mpa
    summ["semiamplitude_p05_p95"] = (summ.mb_p95 - summ.mb_p05) / 2
    summ.to_csv(O / "basal_melt_ase" / "fatorial_v2_resumo.csv")
    tot = t.groupby(list(LEVELS)).Gt.sum()
    tot = tot[t.groupby(list(LEVELS)).unidade.nunique() == len(UNITS)]
    pd.set_option("display.width", 260)
    print("cortes de radar (km):", {k: v for k, v in sorted(rcut.items())})
    print(summ.round(2).to_string())
    print(f"\nTOTAL domínio medido: mediana {tot.median():.1f} Gt/a; p05–p95 {tot.quantile(.05):.1f}–"
          f"{tot.quantile(.95):.1f}; mín–máx {tot.min():.1f}–{tot.max():.1f}; n = {len(tot)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
