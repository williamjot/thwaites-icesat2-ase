"""Comparação com o produto ITS_LIVE de derretimento basal (NSIDC-0792; Paolo et al. 2023).

NSIDC-0792: trimestral, 1.920 m, 1992–2017. Lê-se remotamente só o recorte da ASE (melt_mean e
melt de 2010–2017), sem baixar o arquivo continental. As épocas não se sobrepõem (2019–2025 aqui),
então a comparação é de **padrão espacial e ordem de grandeza**, não de valores contemporâneos.

Comparações, no domínio final do nosso produto (Thwaites com a faixa recuperada):
- média por unidade nos mesmos pixels (o nosso mapa é reamostrado para a grade de 1.920 m);
- correlação espacial (Spearman) pixel a pixel;
- perfil de m_b × distância à linha de aterramento.

Saída: ``data/validation/paolo2023_ase_subset.nc`` e ``outputs/compare_paolo2023/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
SUB = ROOT / "data" / "validation" / "paolo2023_ase_subset.nc"
COMP = ROOT / "data" / "interim" / "basal_melt" / "componentes_uniao_500m.nc"
DOM = ROOT / "data" / "interim" / "basal_melt" / "dominio_final_500m.nc"
THR = ROOT / "data" / "interim" / "basal_melt" / "thwaites_recuperada_500m.nc"
MB = ROOT / "data" / "interim" / "basal_melt" / "mb_continuo_500m.nc"
OUT = ROOT / "outputs" / "compare_paolo2023"
UNITS = ["Pine_Island", "Thwaites_TEIS", "Crosson", "Dotson", "Cosgrove"]
MENG = {"Pine_Island": 16.28, "Thwaites_TEIS": 7.13, "Crosson": 7.79, "Dotson": 7.15, "Cosgrove": 1.22}


def fetch(x0, x1, y0, y1):
    if SUB.is_file():
        return xr.open_dataset(SUB)
    import earthaccess
    earthaccess.login(strategy="netrc")
    g = earthaccess.search_data(short_name="NSIDC-0792", count=1)
    ds = xr.open_dataset(earthaccess.open(g)[0], engine="h5netcdf")
    s = ds[["melt_mean", "melt", "ID"]].sel(x=slice(x0, x1), y=slice(y1, y0))
    s = s.sel(time=slice("2010-01-01", None)).load()
    s.attrs = {"fonte": "NSIDC-0792 v1 (Paolo et al. 2023), recorte ASE, 2010-2017"}  # attrs originais têm bytes inválidos
    s.to_netcdf(SUB)
    return s


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    mb = xr.open_dataset(MB)
    u = mb.unidade.values
    m = mb.mb.values.copy()
    dom = xr.open_dataset(DOM).dominio_final.values == 1
    thr = xr.open_dataset(THR).mb_corrigido.values
    th = np.isin(u, ["Thwaites_TEIS", "Thwaites_TWGT"])
    # Thwaites: domínio e valores da versão recuperada
    m = np.where(th, thr, np.where(dom, m, np.nan))
    unit = xr.open_dataset(COMP).unidade.values
    xs, ys = mb.x.values, mb.y.values
    ii, jj = np.nonzero(np.isfinite(m))
    pad = 10e3
    p = fetch(xs[jj].min() - pad, xs[jj].max() + pad, ys[ii].min() - pad, ys[ii].max() + pad)
    # convenção do NSIDC-0792: perda de massa basal é NEGATIVA (medido: média negativa nas cinco
    # plataformas e Spearman de −0,14 a −0,78 contra o nosso m_b > 0). Inverte-se para m_b > 0.
    pm = -p.melt_mean.values
    px, py = p.x.values, p.y.values
    # reamostra o nosso mapa (500 m) para a grade de 1.920 m: média dos pixels dentro de cada célula
    cx = np.round((xs - px[0]) / (px[1] - px[0])).astype(int)
    cy = np.round((ys - py[0]) / (py[1] - py[0])).astype(int)
    CX, CY = np.meshgrid(cx, cy)
    ok = np.isfinite(m) & (CX >= 0) & (CX < len(px)) & (CY >= 0) & (CY < len(py))
    key = CY[ok] * len(px) + CX[ok]
    our = pd.Series(m[ok]).groupby(key).mean()
    lab = pd.Series(unit[ok]).groupby(key).agg(lambda v: v.value_counts().index[0])
    cnt = pd.Series(m[ok]).groupby(key).size()
    flat_p = pm.ravel()[our.index.to_numpy()]
    d = pd.DataFrame({"nosso": our.to_numpy(), "paolo": flat_p, "unidade": lab.to_numpy(),
                      "n500": cnt.to_numpy()})
    d = d[np.isfinite(d.paolo) & (d.n500 >= 8)]                 # célula de 1,92 km bem coberta
    rows = []
    for un in UNITS:
        g = d[d.unidade == un]
        rows.append({"unidade": un, "n_celulas_1p92km": len(g),
                     "nosso_media_mpa": g.nosso.mean(), "paolo_media_mpa": g.paolo.mean(),
                     "dif_mpa": g.nosso.mean() - g.paolo.mean(),
                     "spearman": g[["nosso", "paolo"]].corr("spearman").iloc[0, 1],
                     "meng_2019_2023": MENG.get(un)})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "comparacao_por_unidade.csv", index=False)
    d.to_csv(OUT / "pares_celulas.csv", index=False)
    pd.set_option("display.width", 200)
    print(t.round(2).to_string(index=False))
    # série recente de Paolo (média por unidade nas mesmas células) para referência de tendência
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
