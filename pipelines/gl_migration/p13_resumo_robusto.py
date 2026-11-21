"""InSAR + ICESat-2, passo D: resumo robusto da comparação observado × previsto.

1. Bootstrap em blocos espaciais: linhas de fluxo a 1,5–2 km não são independentes. Blocos de 10 km
   (pela posição da linha de aterramento) são reamostrados (2.000 réplicas); estatísticas por
   geleira: mediana de v_obs, de v_prev e da razão das medianas.
2. Fração de linhas com recuo observado significativamente mais rápido que o previsto
   (v_obs_hi < v_prev_lo, IC de 95% sem sobreposição) e mais lento (v_obs_lo > v_prev_hi).
3. Sensibilidade à janela do gradiente de HAF (2, 3, 5 km): recalcula v_prev com dh/dt fixo.
Saída: outputs/gl_migration/resumo_previsao_observado.csv e sensibilidade_janela.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "gl_migration"
RNG = np.random.default_rng(20261117)
RW, RI = 1027.0, 917.0


def main():
    d = pd.read_parquet(OUT / "previsao_vs_observado.parquet")
    d = d[np.isfinite(d.v_prev) & np.isfinite(d.v_obs)]
    d["bloco"] = (d.x // 10000).astype(int).astype(str) + "_" + (d.y // 10000).astype(int).astype(str)
    linhas = []
    for gl, g in d.groupby("geleira"):
        blocos = g.bloco.unique()
        reps = []
        for _ in range(2000):
            bb = RNG.choice(blocos, len(blocos))
            s = pd.concat([g[g.bloco == b] for b in bb])
            reps.append((s.v_obs.median(), s.v_prev.median(), s.v_obs.median() / s.v_prev.median()))
        reps = np.array(reps)
        rapido = (g.v_obs_hi < g.v_prev_lo).mean()
        lento = (g.v_obs_lo > g.v_prev_hi).mean()
        linhas.append(dict(geleira=gl, n_linhas=len(g), n_blocos=len(blocos),
                           v_obs=g.v_obs.median(), v_obs_lo=np.percentile(reps[:, 0], 2.5), v_obs_hi=np.percentile(reps[:, 0], 97.5),
                           v_prev=g.v_prev.median(), v_prev_lo=np.percentile(reps[:, 1], 2.5), v_prev_hi=np.percentile(reps[:, 1], 97.5),
                           razao=g.v_obs.median() / g.v_prev.median(), razao_lo=np.percentile(reps[:, 2], 2.5),
                           razao_hi=np.percentile(reps[:, 2], 97.5), frac_mais_rapido=rapido, frac_mais_lento=lento,
                           haf_gl=g.haf_gl.median(), dhdt=g.dhdt.median()))
    r = pd.DataFrame(linhas)
    r.to_csv(OUT / "resumo_previsao_observado.csv", index=False)
    with pd.option_context("display.width", 220):
        print(r.round(2).to_string())

    # sensibilidade à janela
    vert = pd.read_parquet(OUT / "fluxo_linhas.parquet")
    import xarray as xr
    bm = xr.open_dataset(ROOT / "data" / "NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc")
    sub = bm[["thickness", "bed"]].sel(x=slice(vert.x.min() - 2000, vert.x.max() + 2000),
                                       y=slice(vert.y.max() + 2000, vert.y.min() - 2000)).load()
    X, Y = xr.DataArray(vert.x.values, dims="p"), xr.DataArray(vert.y.values, dims="p")
    vert["haf"] = sub.thickness.interp(x=X, y=Y).values + RW / RI * sub.bed.interp(x=X, y=Y).values
    sens = []
    for jan in (2000, 3000, 5000):
        vp = []
        for _, rr in d.iterrows():
            v = vert[vert.fid == rr.fid]
            tr = v[(v.s >= rr.s_gl - jan) & (v.s <= rr.s_gl)]
            if len(tr) < 6 or tr.haf.isna().any():
                vp.append(np.nan); continue
            G = np.polyfit(tr.s, tr.haf, 1)[0]
            vp.append(-rr.dhdt / G if G < 0 else np.nan)
        d[f"vp_{jan}"] = vp
        for gl, g in d.groupby("geleira"):
            sens.append(dict(geleira=gl, janela_km=jan / 1000, frac_G_neg=g[f"vp_{jan}"].notna().mean(),
                             v_prev=g[f"vp_{jan}"].median(), razao=g.v_obs.median() / g[f"vp_{jan}"].median()))
    s = pd.DataFrame(sens)
    s.to_csv(OUT / "sensibilidade_janela.csv", index=False)
    print(s.round(2).to_string())


if __name__ == "__main__":
    main()
