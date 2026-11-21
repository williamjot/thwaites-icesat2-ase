"""
Exporta em shapefile as camadas do mapa da área de estudo.

Por que existe
--------------
A figura (`pipelines/figures/run_mapa_area_estudo.py`) é raster na origem:
BedMachine e REMA são grades. Este script vetoriza EXATAMENTE as mesmas
camadas, importando aquele módulo — assim o shapefile e a figura não podem
divergir. Se a figura mudar de fonte ou de critério, o vetor muda junto.

Duas resoluções, de propósito
-----------------------------
* Camadas continentais: vetorizadas da grade decimada que a figura desenha
  (~4,2 km/px). Servem para REDESENHAR o mapa, não para medir área: a
  decimação por vizinho mais próximo desloca a margem em até meio pixel.
* Camadas regionais do ASE: vetorizadas na resolução NATIVA do BedMachine
  (500 m). São as que suportam análise.

Essa diferença está registrada no campo `res_m` de cada camada e no README
gerado junto — sem isso alguém mede área de plataforma na camada continental e
erra por quilômetros.

Limitações do formato (impostas pelo shapefile, não por escolha)
----------------------------------------------------------------
* Nome de campo com no máximo 10 caracteres.
* Codificação declarada em .cpg como UTF-8; alguns SIG antigos ignoram o .cpg e
  leem como cp1252, o que embaralha acento. Os valores de atributo aqui são
  ASCII por esse motivo — os acentos ficam só no README.
* Uma geometria por camada (sem mistura de ponto/linha/polígono).

Saída: outputs/shapefiles_area_estudo/ (+ .zip)
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.transform import from_bounds
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipelines" / "gelo_apoiado" / "figures"))

import run_mapa_area_estudo as figura   # noqa: E402  (precisa do sys.path acima)

OUTDIR = ROOT / "outputs" / "shapefiles_area_estudo"
CRS = "EPSG:3031"

# Nome legível de cada classe BedMachine. ASCII: ver nota de codificação.
CLASSE = {
    figura.BM_OCEAN: "oceano",
    figura.BM_ROCK: "rocha_exposta",
    figura.BM_GROUNDED: "gelo_aterrado",
    figura.BM_FLOATING: "plataforma_flutuante",
    figura.BM_VOSTOK: "lago_vostok",
}
FONTE_BM = "BedMachine Antarctica v4 (~2019)"


# ================================================================ util =======
def transform_for(shape_hw, extent):
    """Transform afim da grade lida (extent = x0, x1, y0, y1)."""
    x0, x1, y0, y1 = extent
    return from_bounds(x0, y0, x1, y1, shape_hw[1], shape_hw[0])


def polygonize(mask_bool, extent, min_area_m2=0.0, rotulo=""):
    """Polígonos de uma máscara booleana, com conferência de área.

    `connectivity=4` não é detalhe: com 8 o GDAL une pixels que só se tocam na
    diagonal e devolve polígonos que se auto-tocam, portanto inválidos. Medido
    nesta máscara: 38 dos 146 polígonos saíam inválidos e neles estava 99 % da
    área (440 km² válidos de 40.857 km² reais). Descartar inválido é o que
    transforma esse detalhe em dado truncado sem aviso.

    `min_area_m2` existe mas nenhuma camada o usa: descartar polígono pequeno
    altera o dado. Um pixel isolado de rocha exposta É rocha exposta, e filtrar
    por área removia 62 % dessa classe no continente — uma classe legitimamente
    fragmentada vira um shapefile mutilado. Quem quiser generalizar que o faça
    no SIG, com o dado completo em mãos.

    A área total vetorizada é comparada com a do raster e a diferença é
    impressa: é o controle que faz um truncamento aparecer em vez de passar.
    """
    tr = transform_for(mask_bool.shape, extent)
    res_m = (extent[1] - extent[0]) / mask_bool.shape[1]
    geoms, descartados, reparados = [], 0, 0
    for geom, val in features.shapes(mask_bool.astype(np.uint8),
                                     mask=mask_bool, transform=tr,
                                     connectivity=4):
        if val != 1:
            continue
        g = shape(geom)
        if not g.is_valid:
            g = g.buffer(0)                  # repara em vez de jogar fora
            reparados += 1
            if not g.is_valid:
                descartados += 1
                continue
        if g.area >= min_area_m2:
            geoms.append(g)

    area_vet = sum(g.area for g in geoms)
    area_ras = float(mask_bool.sum()) * res_m ** 2
    perda = 100.0 * (1.0 - area_vet / area_ras) if area_ras else 0.0
    nota = f"    {rotulo or 'mascara':26s} {len(geoms):5d} poligonos  " \
           f"area {area_vet / 1e6:9.0f} km2  perda {perda:5.2f} %"
    if reparados or descartados:
        nota += f"  (reparados {reparados}, descartados {descartados})"
    print(nota)
    if perda > 1.0:
        print(f"    !! ATENCAO: {rotulo} perdeu {perda:.2f} % da area do "
              f"raster — confira antes de usar")
    return geoms


def write(gdf, name, descricao):
    """Grava um shapefile e devolve a linha de inventário."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / f"{name}.shp"
    gdf = gdf.set_crs(CRS, allow_override=True)
    gdf.to_file(path, driver="ESRI Shapefile", encoding="utf-8")
    (OUTDIR / f"{name}.cpg").write_text("UTF-8", encoding="ascii")
    kinds = sorted({g.geom_type for g in gdf.geometry})
    print(f"  {name:32s} {len(gdf):6d} feicoes  {','.join(kinds)}")
    return {"camada": name, "n": len(gdf), "geom": ",".join(kinds),
            "descricao": descricao}


def boundary_lines(geoms):
    """Fronteira de um conjunto de polígonos, como linhas."""
    merged = unary_union(geoms)
    bnd = merged.boundary
    parts = list(bnd.geoms) if hasattr(bnd, "geoms") else [bnd]
    return [p for p in parts if isinstance(p, LineString) and len(p.coords) > 1]


# ========================================================= continentais ======
def export_continental(inventario):
    """Camadas do mapa continental, na grade que a figura desenha."""
    cont = figura.continent_layers(target_px=1600)
    mask, extent = cont["mask"], cont["extent"]
    res_m = (extent[1] - extent[0]) / mask.shape[1]
    print(f"[continental] grade {mask.shape}  {res_m:.0f} m/px")

    # Uma feicao por classe, dissolvida: o shapefile fica pequeno e o campo
    # `classe` faz o papel do colormap da figura.
    registros = []
    for code in (figura.BM_ROCK, figura.BM_GROUNDED, figura.BM_FLOATING,
                 figura.BM_VOSTOK):
        sel = mask == code
        if not sel.any():
            continue
        geoms = polygonize(sel, extent, rotulo=CLASSE[code])
        if not geoms:
            continue
        registros.append({"classe": CLASSE[code], "cod_bm": int(code),
                          "res_m": round(res_m), "fonte": FONTE_BM,
                          "geometry": unary_union(geoms)})
    gelo = gpd.GeoDataFrame(registros, crs=CRS)
    inventario.append(write(
        gelo, "ant_superficie_classes",
        "Classes de superficie do continente (dissolvidas por classe)"))

    # Margem de gelo do continente: fronteira do gelo com o oceano.
    ice_geoms = polygonize(mask != figura.BM_OCEAN, extent, rotulo="gelo (margem)")
    linhas = boundary_lines(ice_geoms)
    inventario.append(write(
        gpd.GeoDataFrame({"tipo": "margem_de_gelo", "res_m": round(res_m),
                          "fonte": FONTE_BM,
                          "geometry": linhas}, crs=CRS),
        "ant_margem_gelo", "Limite gelo/oceano do continente (linha)"))

    return cont


def export_dominio(cont, inventario):
    """Domínio de processamento e a mancha destacada na figura."""
    rx, ry = figura.roi_outline_xy()
    roi = Polygon(np.column_stack([rx, ry]))
    inventario.append(write(
        gpd.GeoDataFrame(
            {"nome": ["dominio_processamento"],
             "lon_min": [figura.ROI_LON[0]], "lon_max": [figura.ROI_LON[1]],
             "lat_min": [figura.ROI_LAT[0]], "lat_max": [figura.ROI_LAT[1]],
             "fonte": ["config/default.yaml roi"],
             "geometry": [roi]}, crs=CRS),
        "dominio_processamento",
        "Caixa lon/lat do dominio, densificada em EPSG:3031"))

    mask, extent = cont["mask"], cont["extent"]
    res_m = (extent[1] - extent[0]) / mask.shape[1]
    from matplotlib.path import Path as MplPath

    xs, ys = cont["xs"], cont["ys"]
    X, Y = np.meshgrid(xs, ys)
    inside = MplPath(np.column_stack([rx, ry])).contains_points(
        np.column_stack([X.ravel(), Y.ravel()])).reshape(mask.shape)
    hi = inside & (mask != figura.BM_OCEAN)
    geoms = polygonize(hi, extent, rotulo="area de estudo")
    inventario.append(write(
        gpd.GeoDataFrame({"nome": "area_estudo_destaque", "res_m": round(res_m),
                          "fonte": FONTE_BM,
                          "geometry": [unary_union(geoms)]}, crs=CRS),
        "area_estudo_destaque",
        "Gelo contido no dominio: a mancha azul da figura. "
        "NAO e a bacia de drenagem da Thwaites"))


# ============================================================== regionais ====
def export_regional(inventario):
    """Camadas do ASE na resolução nativa do BedMachine."""
    rx, ry = figura.roi_outline_xy()
    pad = 24_000.0
    bounds = (rx.min() - pad, ry.min() - pad, rx.max() + pad, ry.max() + pad)

    # Limite alto de propósito: `read_bedmachine` só decima quando a janela
    # excede o limite, então isto lê na resolução nativa de 500 m.
    mask, extent = figura.read_bedmachine(bounds, target_px=100_000)
    res_m = (extent[1] - extent[0]) / mask.shape[1]
    xs, ys = figura.grid_centers(mask.shape, extent)
    print(f"[regional] grade {mask.shape}  {res_m:.0f} m/px")

    registros = []
    for code in (figura.BM_ROCK, figura.BM_GROUNDED, figura.BM_FLOATING):
        sel = mask == code
        if not sel.any():
            continue
        geoms = polygonize(sel, extent, rotulo=CLASSE[code])
        if geoms:
            registros.append({"classe": CLASSE[code], "cod_bm": int(code),
                              "res_m": round(res_m), "fonte": FONTE_BM,
                              "geometry": unary_union(geoms)})
    inventario.append(write(
        gpd.GeoDataFrame(registros, crs=CRS), "ase_superficie_classes",
        "Classes de superficie no ASE, resolucao nativa 500 m"))

    grounded = polygonize(
        np.isin(mask, (figura.BM_ROCK, figura.BM_GROUNDED, figura.BM_VOSTOK)),
        extent, rotulo="aterrado (GL)")
    inventario.append(write(
        gpd.GeoDataFrame({"tipo": "linha_aterramento", "res_m": round(res_m),
                          "fonte": FONTE_BM,
                          "geometry": boundary_lines(grounded)}, crs=CRS),
        "ase_linha_aterramento",
        "Limite do gelo aterrado: a linha preta da figura. Epoca ~2019; "
        "a GL da Thwaites migra"))

    ice = polygonize(mask != figura.BM_OCEAN, extent, rotulo="gelo (margem ASE)")
    inventario.append(write(
        gpd.GeoDataFrame({"tipo": "margem_de_gelo", "res_m": round(res_m),
                          "fonte": FONTE_BM,
                          "geometry": boundary_lines(ice)}, crs=CRS),
        "ase_margem_gelo", "Limite gelo/oceano no ASE (frente de calving)"))

    # Plataformas nomeadas: poligono do BedMachine, nome do NSIDC-0794.
    shelf_id, code_map = figura.sample_iceshelf_id(xs, ys)
    floating = mask == figura.BM_FLOATING
    registros = []
    for nome in ("Thwaites", "Pine_Island", "Crosson", "Dotson", "Cosgrove",
                 "Getz", "Abbot"):
        if nome not in code_map:
            continue
        sel = floating & (shelf_id == code_map[nome])
        if sel.sum() < 30:
            continue
        geoms = polygonize(figura.largest_component(sel), extent, rotulo=nome)
        if not geoms:
            continue
        registros.append({
            "nome": nome.replace("_", " "), "id_nsidc": int(code_map[nome]),
            "res_m": round(res_m), "fonte": FONTE_BM + " + NSIDC-0794 v1",
            "geometry": unary_union(geoms)})
    inventario.append(write(
        gpd.GeoDataFrame(registros, crs=CRS), "ase_plataformas_nomeadas",
        "Plataformas: geometria do BedMachine, nome do NSIDC-0794 "
        "(mascara dilatada, usada so para nomear)"))

    return mask, extent, xs, ys, shelf_id, code_map


def export_isolinhas(mask, extent, xs, ys, inventario):
    """Isolinhas REMA de 400 m, só sobre gelo aterrado — mesma regra da figura:
    sobre plataforma a superficie responde a flutuacao e mare, e a isolinha ali
    nao tem o mesmo significado."""
    import matplotlib.pyplot as plt

    elev = figura.read_rema(mask.shape, extent)
    z = np.where(np.isin(mask, (figura.BM_GROUNDED, figura.BM_VOSTOK)), elev,
                 np.nan)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cs = ax.contour(xs, ys, z, levels=figura.CONTOUR_LEVELS)
    registros = []
    for nivel, segs in zip(cs.levels, cs.allsegs):
        for seg in segs:
            if len(seg) > 1:
                registros.append({"elev_m": float(nivel), "passo_m": 400,
                                  "fonte": "REMA",
                                  "geometry": LineString(seg)})
    plt.close(fig)
    inventario.append(write(
        gpd.GeoDataFrame(registros, crs=CRS), "ase_isolinhas_rema",
        "Isolinhas de elevacao REMA (400 m), so sobre gelo aterrado"))


def export_toponimos(mask, extent, xs, ys, shelf_id, code_map, inventario):
    """Pontos de rótulo: as MESMAS âncoras que a figura usa."""
    shelves = figura.shelf_label_points(
        mask, shelf_id, code_map, xs, ys,
        {"Thwaites": "Thwaites", "Pine Island": "Pine_Island",
         "Crosson": "Crosson", "Dotson": "Dotson", "Cosgrove": "Cosgrove",
         "Getz": "Getz"})
    glaciers = figura.glacier_label_points(
        shelves, {"Thwaites": ("Thwaites", 38.0),
                  "Pine Island": ("Pine Island", 45.0),
                  "Pope Kohler Smith": ("Crosson", 42.0)})

    registros = []
    for nome, (x, y, _) in shelves.items():
        registros.append({
            "nome": f"Plataforma {nome}", "tipo": "plataforma",
            "metodo": "ponto interno do poligono (dist. transform)",
            "geometry": Point(x, y)})
    for nome, (x, y, n) in glaciers.items():
        rotulo = {"Pope Kohler Smith": "Geleiras Pope/Smith/Kohler"}.get(
            nome, f"Geleira {nome}")
        registros.append({
            "nome": rotulo, "tipo": "geleira", "n_gl": int(n),
            "metodo": "centroide da GL InSAR projetado a montante",
            "geometry": Point(x, y)})
    inventario.append(write(
        gpd.GeoDataFrame(registros, crs=CRS), "ase_toponimos",
        "Ancoras de texto da figura (plataformas e geleiras)"))


def export_grounding_lines(inventario):
    """GL InSAR observada, recortada no ASE — no mapa ela só posiciona
    topônimos, mas é a linha OBSERVADA e datada, útil por si."""
    gl = gpd.read_file(figura.INSAR_GL).to_crs(CRS)
    cols = [c for c in ("Glac_Name", "Sensor", "Year", "geometry")
            if c in gl.columns]
    gl = gl[cols].rename(columns={"Glac_Name": "glaciar", "Sensor": "sensor",
                                  "Year": "ano"})
    inventario.append(write(
        gl, "ase_gl_insar_observada",
        "MEaSUREs InSAR GL v2.1 no ASE (1992-2025). Glac_Name identifica o "
        "par de aquisicao, nao necessariamente uma unica geleira"))


# ================================================================ README =====
def write_readme(inventario, res_cont, res_ase):
    linhas = [
        "# Shapefiles do mapa da área de estudo",
        "",
        "Gerado por `pipelines/export_mapa_shapefiles.py`, que importa",
        "`pipelines/figures/run_mapa_area_estudo.py` — as geometrias aqui são",
        "as mesmas que a figura desenha.",
        "",
        "**CRS de todas as camadas: EPSG:3031** (WGS84 / Antarctic Polar",
        "Stereographic, escala verdadeira em 71°S).",
        "",
        "## Camadas",
        "",
        "| Camada | Feições | Geometria | Descrição |",
        "|---|---:|---|---|",
    ]
    for item in inventario:
        linhas.append(f"| `{item['camada']}` | {item['n']} | {item['geom']} | "
                      f"{item['descricao']} |")
    linhas += [
        "",
        "## Duas resoluções — não misture os usos",
        "",
        "* `ant_*` e `area_estudo_destaque`: vetorizadas da grade decimada que",
        f"  a figura desenha ({res_cont / 1000:.1f} km/px). Servem para "
        "**redesenhar o mapa**.",
        "  Não meça área com elas: a reamostragem por vizinho mais próximo",
        "  desloca a margem em até meio pixel.",
        f"* `ase_*`: resolução nativa do BedMachine ({res_ase:.0f} m). São as",
        "  camadas que suportam análise.",
        "",
        "O campo `res_m` registra a resolução de origem de cada feição.",
        "",
        "## Ressalvas que viajam com os dados",
        "",
        "* **`area_estudo_destaque` não é a bacia de drenagem da Thwaites.** É",
        "  o gelo contido no domínio de processamento, que inclui Pine Island,",
        "  Smith/Pope/Kohler e parte da Getz. Este projeto não tem polígono de",
        "  bacia em disco.",
        "* **`ase_linha_aterramento` é o limite grounded/floating do BedMachine",
        "  v4, época ~2019**, e não uma GL instantânea. A linha da Thwaites",
        "  migra. Para a linha observada e datada, use",
        "  `ase_gl_insar_observada`, que traz o campo `ano`.",
        "* **`ase_plataformas_nomeadas` tem geometria do BedMachine e nome do",
        "  NSIDC-0794.** A máscara do NSIDC é dilatada e extrapolada, então ela",
        "  entra apenas para atribuir nome — nunca como fronteira.",
        "* **`ase_gl_insar_observada`: `glaciar` identifica o par de aquisição**",
        "  (há valores como \"Abbot Thwaites\"), não necessariamente uma única",
        "  geleira. Não agregue por esse campo sem conferir.",
        "* **Isolinhas REMA só sobre gelo aterrado.** Sobre plataforma a",
        "  superfície responde a flutuação e maré; a isolinha ali não teria o",
        "  mesmo significado.",
        "",
        "## Codificação",
        "",
        "Os `.cpg` declaram UTF-8, mas SIG antigos ignoram o `.cpg` e leem em",
        "cp1252. Por isso os valores de atributo são ASCII — os acentos ficam",
        "só neste README.",
        "",
    ]
    (OUTDIR / "README.md").write_text("\n".join(linhas), encoding="utf-8")
    print(f"  README.md ({len(inventario)} camadas)")


def main():
    if OUTDIR.exists():
        shutil.rmtree(OUTDIR)
    OUTDIR.mkdir(parents=True)
    inventario = []

    print("[1/5] camadas continentais")
    cont = export_continental(inventario)
    export_dominio(cont, inventario)

    print("[2/5] camadas regionais (ASE, 500 m)")
    mask, extent, xs, ys, shelf_id, code_map = export_regional(inventario)

    print("[3/5] isolinhas REMA")
    export_isolinhas(mask, extent, xs, ys, inventario)

    print("[4/5] toponimos e GL observada")
    export_toponimos(mask, extent, xs, ys, shelf_id, code_map, inventario)
    export_grounding_lines(inventario)

    print("[5/5] README e pacote")
    res_cont = ((cont["extent"][1] - cont["extent"][0])
                / cont["mask"].shape[1])
    res_ase = (extent[1] - extent[0]) / mask.shape[1]
    write_readme(inventario, res_cont, res_ase)
    zip_path = shutil.make_archive(str(OUTDIR), "zip", root_dir=str(OUTDIR))
    print(f"-> {OUTDIR}")
    print(f"-> {zip_path}")


if __name__ == "__main__":
    main()
