"""Incerteza completa do derretimento basal no domínio do produto central v2 (frente por data).

Substitui, para a tabela final, ``run_factorial_v2_ase.py`` + ``envelope_por_unidade.csv``, que
foram calculados no domínio anterior (frente pela união das frentes; Thwaites cortada pelo radar).

1. **Fatorial estrutural** (m_b por pixel, média por área no domínio de cada combinação):

       m_b = SMB [+ (RACMO − GEMB)] − DH/Dt − H_firn·∇·u_σ − Δ_firn [+ Δ_Thwaites]

   Fatores:
   - máscara f: f080, f090, f095, IC;
   - distância à frente da mesma data (km): 1,0 / 1,5 / 2,0. Cada nível tem o seu produto
     (``run_front_epoch_ase.py --d-front``): muda o filtro dos segmentos, logo DH/Dt, H e a
     distância à célula observada, e também o corte estático da frente;
   - perda de gelo máxima no pixel: 5 / 10 / 20%;
   - célula observada a ≤ 0,5 / 1,0 / 1,5 km libera a faixa da frente;
   - tolerância do radar MCoRDS (fora da Thwaites): 2 / 3 / 5%;
   - escala de ∇·u: 1, 2, 3, 5 km;
   - SMB: GEMB × RACMO; firn: GEMB × IMAU (H × razão H_IMAU/H_GEMB; Δ_firn = diferença das
     tendências IMAU − GEMB em 2019–2023, da série da variante de máscara — único termo
     transferido entre períodos);
   - Thwaites: correção β·maré + radar (produto) × corte pelo radar na tolerância do nível.
   Efeito estrutural = semiamplitude p05–p95 sobre as combinações.
2. **Agregação**: média do mapa × média só das células com observação própria × m_b com o DH/Dt da
   série de espessura v2 (regressão sobre as 28 estações). Efeito = maior |diferença| ao central.
3. **Velocidade**: MEaSUREs − ITS_LIVE em H·∇·u (``run_velocity_crosscheck_ase.py``; domínio
   anterior, declarado).
4. **Estatística**: σ da tendência da série v2 (σ trimestrais do jackknife por grupos de passagens).

σ_estrutural = √(fatorial² + agregação² + velocidade²); σ_total = √(σ_estr² + σ_estat²).
O total em Gt/ano também sai do fatorial na mesma combinação para todas as plataformas.

Saídas: ``outputs/basal_melt_ase/incerteza_v2_*.csv``.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipelines" / "derretimento_basal"))
from run_factorial_v2_ase import radar_cut, trend  # noqa: E402

I = ROOT / "data" / "interim" / "basal_melt"
O = ROOT / "outputs"
COMP = I / "componentes_uniao_500m.nc"
THR = I / "thwaites_recuperada_500m.nc"
PROD = {1.0: I / "mb_final_v2_dfront1.nc", 1.5: I / "mb_final_v2_500m.nc", 2.0: I / "mb_final_v2_dfront2.nc"}
UNITS = ["Pine_Island", "Thwaites_TEIS", "Crosson", "Dotson", "Cosgrove"]
BASE = {u: u.replace("_TEIS", "") for u in UNITS}
D_REC = {"Pine_Island": 3.0, "Thwaites_TEIS": 1.5, "Crosson": 3.0, "Dotson": 2.5, "Cosgrove": 2.0}
OPT = {"Pine_Island": 3, "Thwaites_TEIS": 2, "Crosson": 1, "Dotson": 1, "Cosgrove": 1}
VARS = {"f090": "classe", "f080": "classe_f080", "f095": "classe_f095", "ic": "classe_ic"}
LEVELS = {"mascara": list(VARS), "frente_km": [1.0, 1.5, 2.0], "perda_max": [0.05, 0.10, 0.20],
          "obs_km": [0.5, 1.0, 1.5], "radar_tol": [0.02, 0.03, 0.05], "escala_km": [1, 2, 3, 5],
          "smb": ["GEMB", "RACMO"], "firn": ["GEMB", "IMAU"], "thwaites": ["correcao", "corte_radar"]}
CENTRAL = {"mascara": "f090", "frente_km": 1.5, "perda_max": 0.10, "obs_km": 1.0, "radar_tol": 0.03,
           "smb": "GEMB", "firn": "GEMB", "thwaites": "correcao"}
RHO_I, PIX = 917.0, 0.25e6


def wslope(t, y, e):
    ok = np.isfinite(y) & np.isfinite(e)
    p, c = np.polyfit(t[ok], y[ok], 1, w=1 / np.maximum(e[ok], 1e-3), cov=True)
    return float(p[0]), float(np.sqrt(c[0, 0]))


def main() -> int:
    for k, p in PROD.items():
        if not p.is_file():
            raise FileNotFoundError(f"falta o produto da frente {k} km: {p} (rode run_front_epoch_ase.py --d-front {k})")
    c = xr.open_dataset(COMP)
    unit = xr.open_dataset(PROD[1.5]).unidade.values
    prod = {k: xr.open_dataset(p) for k, p in PROD.items()}
    delta = np.nan_to_num(xr.open_dataset(THR).delta_correcao.values)
    smb_diff = pd.read_csv(O / "basal_melt_ase" / "smb_gemb_racmo.csv").set_index("unidade")["diff"]
    prof = pd.read_csv(O / "mcords_hydro_test" / "perfil_razao_normalizado_por_voo.csv")
    rcut = {(p, tol): radar_cut(prof, p, tol) for p in set(BASE.values()) for tol in LEVELS["radar_tol"]}
    print("corte do radar (km):", {k: round(v, 2) for k, v in sorted(rcut.items())})
    dfirn = {}
    for k, v in VARS.items():
        s = pd.read_csv(O / "thickness_ase" / f"serie_espessura_{v}.csv")
        for un in UNITS:
            g = s[s.plataforma == un]
            dfirn[(k, un)] = trend(g, "dH_lagr_imau_m", (2019, 2023)) - trend(g, "dH_lagr_m", (2019, 2023))
    rows = []
    for un in UNITS:
        s = unit == un
        P = {k: c[k].values[s] for k in ("SMB", "perda_gelo_frac", "frente_dist_km", "dist_LA_BedMachine_km",
                                         "dist_LA_recente_km", "H_gemb", "H_imau")}
        ratio = np.where(P["H_gemb"] > 0, P["H_imau"] / P["H_gemb"], np.nan)
        div = {sc: c[f"div_{sc}km"].values[s] for sc in LEVELS["escala_km"]}
        cls = {k: c[v].values[s] == 1 for k, v in VARS.items()}
        pr = {k: {"DHDt": d.DHDt.values[s], "H": d.H.values[s], "obs": d.dist_obs_m.values[s] / 1e3}
              for k, d in prod.items()}
        dl = delta[s]
        base_ok = (P["dist_LA_recente_km"] >= D_REC[un]) & np.isfinite(P["SMB"])
        is_thw = un.startswith("Thwaites")
        for combo in itertools.product(*LEVELS.values()):
            f = dict(zip(LEVELS, combo))
            if not is_thw and f["thwaites"] != "correcao":
                continue                                      # fator só existe na Thwaites
            if is_thw and f["thwaites"] == "correcao" and f["radar_tol"] != 0.03:
                continue                                      # com correção, não há corte de radar
            q = pr[f["frente_km"]]
            H = q["H"] * ratio if f["firn"] == "IMAU" else q["H"]
            front_ok = ((P["frente_dist_km"] > f["frente_km"]) & ~(P["perda_gelo_frac"] > f["perda_max"])) \
                | (q["obs"] <= f["obs_km"])
            if is_thw and f["thwaites"] == "correcao":
                radar_ok = np.ones(s.sum(), bool)
            else:
                radar_ok = P["dist_LA_BedMachine_km"] >= rcut[(BASE[un], f["radar_tol"])]
            dv = div[f["escala_km"]]
            dom = cls[f["mascara"]] & base_ok & front_ok & radar_ok & np.isfinite(H) & np.isfinite(dv) \
                & np.isfinite(q["DHDt"])
            if dom.sum() < 20:
                continue
            mb = P["SMB"][dom] - q["DHDt"][dom] - H[dom] * dv[dom]
            if is_thw and f["thwaites"] == "correcao":
                mb = mb + dl[dom]
            if f["smb"] == "RACMO":
                mb = mb + smb_diff[un]
            if f["firn"] == "IMAU":
                mb = mb - dfirn[(f["mascara"], un)]
            mean = float(np.mean(mb))
            rows.append({"unidade": un, **f, "mb_mpa": mean, "area_km2": dom.sum() * PIX / 1e6,
                         "Gt": mean * dom.sum() * PIX * RHO_I / 1e12})
        print(f"{un}: {sum(r['unidade'] == un for r in rows)} combinações", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(O / "basal_melt_ase" / "incerteza_v2_fatorial_combinacoes.csv", index=False)

    # central (deve reproduzir o produto v2)
    def is_central(r):
        return all(r[k] == v for k, v in CENTRAL.items()) and r.escala_km == OPT[r.unidade]
    cen = t[t.apply(is_central, axis=1)].set_index("unidade")

    # agregação e estatística, no domínio v2
    v2 = prod[1.5]
    m, dom, obs = v2.mb.values, v2.dominio.values == 1, v2.dist_obs_m.values
    ser = pd.read_csv(O / "series_v2" / "espessura_trimestral_v2.csv")
    q = pd.read_csv(O / "series_v2" / "mb_trimestral_v2.csv")
    vel = pd.read_csv(O / "velocity_crosscheck" / "Hdiv_ITS_LIVE_vs_MEaSUREs.csv")
    summ = []
    for un in UNITS:
        g = t[t.unidade == un]
        k = dom & (unit == un)
        central_map = float(np.mean(m[k]))
        cells = float(np.mean(m[k & (obs == 0)]))
        gs = ser[ser.unidade == un].sort_values("t_meio")
        dh, se_stat = wslope(gs.t_meio.to_numpy(), gs.dH_m.to_numpy(), gs.se_m.to_numpy())
        gq = q[q.unidade == un]
        serie = float(gq.SMB_mpa.mean() - dh - gq.Hdiv_mpa.mean() + gq.correcao_mpa.mean())
        alts = {"celulas_observadas": cells - central_map, "serie": serie - central_map}
        d_agg = max(abs(v) for v in alts.values())
        dv = float(vel[(vel.unidade == un) & (vel.escala_km == max(OPT[un], 2))].diferenca_mpa.iloc[0])
        semi = float((g.mb_mpa.quantile(.95) - g.mb_mpa.quantile(.05)) / 2)
        # efeito principal; na Thwaites o radar só varia sem a correção (desenho desbalanceado)
        eff = {}
        for fac in LEVELS:
            gg = g[g.thwaites == "corte_radar"] if (fac == "radar_tol" and un.startswith("Thwaites")) else g
            if gg[fac].nunique() > 1:
                eff[fac] = float(gg.groupby(fac).mb_mpa.mean().agg(lambda v: v.max() - v.min()))
        s_struct = float(np.sqrt(semi ** 2 + d_agg ** 2 + dv ** 2))
        s_tot = float(np.hypot(s_struct, se_stat))
        comps = {"fatorial": semi, "agregacao": d_agg, "velocidade": abs(dv), "estatistica": se_stat}
        summ.append({"unidade": un, "mb_mapa_mpa": central_map, "mb_central_fatorial": float(cen.loc[un, "mb_mpa"]),
                     "fatorial_p05": g.mb_mpa.quantile(.05), "fatorial_p95": g.mb_mpa.quantile(.95),
                     "semiamplitude_fatorial": semi, "agregacao": d_agg,
                     "agregacao_celulas_obs": alts["celulas_observadas"], "agregacao_serie": alts["serie"],
                     "velocidade": dv, "sigma_estat": se_stat, "sigma_estrutural": s_struct,
                     "sigma_total": s_tot, "fator_dominante": max(comps, key=comps.get), "n_comb": len(g),
                     **{f"efeito_{k}": v for k, v in eff.items()}})
    sm = pd.DataFrame(summ)
    sm.to_csv(O / "basal_melt_ase" / "incerteza_v2_resumo.csv", index=False)

    # total em Gt/ano: mesma combinação para todas as plataformas (fator Thwaites fixo no central
    # para as demais; combinações com o corte de radar da Thwaites usam o tol do mesmo nível)
    key = [k for k in LEVELS if k not in ("thwaites",)]
    tt = t.copy()
    other = tt.unidade != "Thwaites_TEIS"
    tt_thw = tt[~other]
    tot = []
    for thw_lvl in LEVELS["thwaites"]:
        a = tt_thw[tt_thw.thwaites == thw_lvl]
        if thw_lvl == "correcao":
            a = pd.concat([a.assign(radar_tol=r) for r in LEVELS["radar_tol"]])
        tot.append(pd.concat([tt[other], a]).groupby(key).Gt.agg(["sum", "size"]))
    tot = pd.concat(tot)
    tot = tot[tot["size"] == len(UNITS)]["sum"]
    pd.set_option("display.width", 280)
    print(sm.round(2).to_string(index=False))
    print(f"\nTOTAL medido (fatorial): mediana {tot.median():.1f}; p05–p95 {tot.quantile(.05):.1f}–"
          f"{tot.quantile(.95):.1f} Gt/a; n = {len(tot)}")
    pd.DataFrame({"Gt_total": tot.values}).describe(percentiles=[.05, .5, .95]).to_csv(
        O / "basal_melt_ase" / "incerteza_v2_total_Gt.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
