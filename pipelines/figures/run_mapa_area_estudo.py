"""
Figura 1 — Mapa da área de estudo (Amundsen Sea Embayment / Geleira Thwaites).

Escopo
------
Figura de LOCALIZAÇÃO: geometria observada + topônimos. Nenhum resultado do
pipeline (dh/dt, taxa basal, massa) entra aqui — o objetivo é situar o leitor,
não apresentar medida.

Saídas
------
`--layout continental` (padrão): figura única — a Antártica inteira com o
domínio de estudo destacado. É o mapa de localização pedido.

`--layout duplo`: acrescenta um segundo painel com o recorte do Amundsen Sea
Embayment (geometria de superfície e topônimos). Fica disponível para quando a
figura precisar mostrar o detalhe regional, e grava com outro nome para não
sobrescrever a figura principal.

Fontes (todas já em disco; nada é baixado em tempo de execução)
---------------------------------------------------------------
* BedMachine Antarctica v4 (`data/BedMachineAntarctica.tif`, 500 m, EPSG:3031)
  — máscara de superfície (oceano / rocha exposta / gelo aterrado / plataforma
  flutuante). É a MESMA fonte da máscara científica do pipeline, então a linha
  de aterramento desenhada é exatamente a que separa `grounded` de `floating`
  nos produtos. Usar um contorno externo produziria uma figura que discorda do
  cálculo. Época nominal do produto: ~2019 — declarada no rodapé, porque a
  linha de aterramento da Thwaites migra e uma linha sem data seria enganosa.
* REMA (`data/REMA_ase_32m.tif`) — sombreamento de relevo e isolinhas de
  elevação, lidos decimados. Servem de referência topográfica (forma da bacia,
  divisores de gelo); não codificam nenhuma grandeza medida por este trabalho.
  O mosaico cobre exatamente a ROI.
* NSIDC-0794 v1 (`iceshelf_id`) — usado SOMENTE para ATRIBUIR NOME aos polígonos
  de plataforma do BedMachine. A máscara do NSIDC é dilatada e extrapolada (ver
  data/external_geometry/README.md), então seu contorno não é desenhado e seu
  centróide não é usado como âncora: o rótulo é posicionado no ponto mais
  interno do polígono do BedMachine (transformada de distância), garantindo que
  o texto caia dentro da geometria efetivamente desenhada.
* MEaSUREs InSAR GL v2.1 recortada no ASE — usada SOMENTE para posicionar
  topônimos de geleira aterrada. Só os nomes de geleira única entram; os nomes
  compostos do atributo `Glac_Name` (ex. "Abbot Thwaites") identificam o par de
  aquisição, não a geleira, e seus centróides caem fora do alvo.

Projeção
--------
Dados em EPSG:3031. O mapa continental usa a projeção nativa (meridiano central
0°), a vista polar canônica. O painel regional do layout duplo usa a MESMA
projeção girada para meridiano central 105°W (centro do ASE), de modo que o
norte da região aponte para cima — sem isso a região aparece girada ~105° e
desperdiça metade da moldura.

Não há rosa dos ventos: em estereográfica polar o norte é radial (aponta para
fora do polo), logo uma seta "N" única seria falsa em todo ponto fora do
meridiano onde fosse desenhada. A orientação é lida nos meridianos rotulados.

Arquivos: outputs/fig1_area_estudo.{pdf,png} e, no layout duplo,
outputs/fig1_area_estudo_duplo.{pdf,png}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import rasterio
from matplotlib.colors import LightSource, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.path import Path as MplPath
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BEDMACHINE = ROOT / "data" / "BedMachineAntarctica.tif"
BEDMACHINE_NC = (ROOT / "data"
                 / "NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc")
REMA = ROOT / "data" / "REMA_ase_32m.tif"
ICESHELF_ID = (ROOT / "data" / "external_geometry" / "NSIDC-0794_v1"
               / "NSIDC-0794_19971001-20210314_V01.0.nc")
INSAR_GL = ROOT / "data" / "grounding" / "processed" / "InSAR_GL_ASE_v02.1.gpkg"
OUTDIR = ROOT / "outputs"

# Codificação BedMachine v4 (flag_meanings da variável `mask`).
BM_OCEAN, BM_ROCK, BM_GROUNDED, BM_FLOATING, BM_VOSTOK = 0, 1, 2, 3, 4

# ROI de processamento do pipeline (config/default.yaml -> roi).
ROI_LON = (-115.0, -95.0)
ROI_LAT = (-77.5, -73.0)

TRUE_SCALE_LAT = -71.0          # escala verdadeira do EPSG:3031
CENTRAL_LON_ASE = -105.0        # centro do ASE: gira o eixo, não o dado

CRS_DATA = ccrs.SouthPolarStereo(central_longitude=0.0,
                                 true_scale_latitude=TRUE_SCALE_LAT)
CRS_ASE = ccrs.SouthPolarStereo(central_longitude=CENTRAL_LON_ASE,
                                true_scale_latitude=TRUE_SCALE_LAT)
GEO = ccrs.PlateCarree()

# ---------------------------------------------------------------- paleta -----
# A cor codifica classe de superfície, não grandeza: baixa saturação, sem
# gradiente, legível em impressão P&B.
C_OCEAN = "#cfdfeb"
C_SHELF = "#9dc8e4"
C_GROUNDED = "#fafbfc"
C_ROCK = "#9a9189"
C_GL = "#15171a"
C_COAST = "#2f6f99"
C_BOX = "#b3302a"
C_TEXT = "#15171a"
C_SHELF_TEXT = "#14506f"
C_SEA_TEXT = "#4a7d9e"
C_CONTOUR = "#b9c5ce"
C_FOOT = "#6b7883"
C_REGION = "#8a94a0"

# Painel a (continente)
C_CONT_ICE = "#e4ebf1"
C_CONT_SHELF = "#cfe0ec"
C_CONT_EDGE = "#8ba3b5"
C_HIGHLIGHT = "#17527a"

# Isolinhas de elevação REMA (m). Passo de 400 m: com 250 m o interior fica
# rajado e as linhas competem com a linha de aterramento.
CONTOUR_LEVELS = np.arange(400, 3200, 400)

EPOCA_NOTA = ("Geometria: BedMachine Antarctica v4 (~2019); relevo e isolinhas: "
              "REMA. A linha de aterramento da Thwaites migra — a época vale "
              "para as duas linhas desenhadas.")


# ============================================================== leitura ======
def read_bedmachine(bounds=None, target_px=1600):
    """Máscara BedMachine decimada. Reamostragem NEAREST — o dado é categórico;
    qualquer média inventaria classes inexistentes."""
    with rasterio.open(BEDMACHINE) as src:
        if bounds is None:
            win = Window(0, 0, src.width, src.height)
        else:
            x0, y0, x1, y1 = bounds
            win = from_bounds(x0, y0, x1, y1, transform=src.transform)
            win = win.intersection(Window(0, 0, src.width, src.height))
        # Um único fator para as duas dimensões. Limitar altura e largura
        # separadamente a `target_px` comprime a janela sempre que ela não é
        # quadrada e o limite cai entre os dois lados — o mapa sai com a
        # escala errada em um dos eixos.
        scale = min(1.0, target_px / max(win.height, win.width))
        h = max(int(round(win.height * scale)), 1)
        w = max(int(round(win.width * scale)), 1)
        arr = src.read(1, window=win, out_shape=(h, w),
                       resampling=Resampling.nearest)
        b = rasterio.windows.bounds(win, src.transform)
    return arr, (b[0], b[2], b[1], b[3])       # x0, x1, y0, y1


def read_rema(shape, extent):
    """Elevação REMA na MESMA grade da máscara.

    Lida por janela com `out_shape` — o mosaico tem 22.729 x 21.173 px a 32 m e
    não cabe em memória.
    """
    x0, x1, y0, y1 = extent
    with rasterio.open(REMA) as src:
        win = from_bounds(x0, y0, x1, y1, transform=src.transform)
        win = win.intersection(Window(0, 0, src.width, src.height))
        elev = src.read(1, window=win, out_shape=shape,
                        resampling=Resampling.average).astype(float)
        nodata = src.nodata
    if nodata is not None:
        elev[elev == nodata] = np.nan
    elev[elev < -100] = np.nan
    return elev


def hillshade(elev, res_m, azdeg=315.0, altdeg=45.0, vert_exag=60.0):
    """Sombreamento com exagero vertical alto — POR QUÊ:

    a declividade mediana do interior do ASE, medida neste recorte, é ~0,4 %
    (0,24°). Com `vert_exag` de ordem 1 o sombreamento fica plano e o interior
    do manto vira um vazio branco. O exagero é escolha ESTÉTICA e está
    declarado: o relevo aqui é referência de forma, não medida — nenhuma
    grandeza é lida desta camada.

    O NaN do oceano é preenchido só para o cálculo do gradiente e restaurado
    depois: sem isso o oceano é renderizado como terreno e a figura inventa
    relevo onde há água.
    """
    valid = np.isfinite(elev)
    if not valid.any():
        return None
    idx = ndimage.distance_transform_edt(~valid, return_distances=False,
                                         return_indices=True)
    filled = elev[tuple(idx)]
    shade = LightSource(azdeg=azdeg, altdeg=altdeg).hillshade(
        filled, vert_exag=vert_exag, dx=res_m, dy=res_m)
    shade[~valid] = np.nan
    return shade


def read_bedmachine_surface(xs, ys):
    """Elevação de superfície do BedMachine (continente inteiro) na grade dada.

    O REMA cobre só o recorte do ASE; para dar relevo ao continente a única
    fonte em disco é a variável `surface` do grânulo NetCDF do BedMachine —
    13.333 x 13.333 px a 500 m, ou 711 MB em float32. Por isso a leitura é
    decimada por passo no próprio NetCDF (`[::step, ::step]`), nunca inteira.

    O passo é escolhido para ficar igual ou mais fino que a grade de destino, e
    o resultado é amostrado por vizinho mais próximo em `xs`/`ys`. A grade do
    NetCDF e a do GeoTIFF da máscara são a mesma (centros de 500 m coincidentes),
    então o resíduo de alinhamento fica abaixo de meio pixel.
    """
    alvo = abs(float(xs[1] - xs[0]))
    with nc.Dataset(BEDMACHINE_NC) as ds:
        gx = np.asarray(ds.variables["x"][:]).astype(float)
        gy = np.asarray(ds.variables["y"][:]).astype(float)
        nativo = abs(gx[1] - gx[0])
        step = max(1, int(alvo // nativo))
        sx, sy = gx[::step], gy[::step]
        sup = np.asarray(ds.variables["surface"][::step, ::step]).astype(float)
    ix = np.clip(np.searchsorted(sx, xs), 0, sx.size - 1)
    iy = np.clip(np.searchsorted(-sy, -ys), 0, sy.size - 1)
    return sup[np.ix_(iy, ix)]


def grid_centers(shape, extent):
    """Centros de pixel da imagem lida (origin='upper')."""
    h, w = shape
    x0, x1, y0, y1 = extent
    dx, dy = (x1 - x0) / w, (y1 - y0) / h
    return x0 + dx * (np.arange(w) + 0.5), y1 - dy * (np.arange(h) + 0.5)


def sample_iceshelf_id(xs, ys):
    """`iceshelf_id` do NSIDC-0794 amostrado (nearest) na grade do BedMachine.
    Lê apenas a janela necessária: o grânulo tem 22.896 x 18.392 px."""
    with nc.Dataset(ICESHELF_ID) as ds:
        var = ds.variables["iceshelf_id"]
        gx = np.asarray(ds.variables["x"][:])
        gy = np.asarray(ds.variables["y"][:])
        code = {n: int(v) for v, n in zip(np.asarray(var.flag_values),
                                          str(var.flag_meanings).split())}
        i0 = max(int(np.searchsorted(gx, xs.min())) - 2, 0)
        i1 = min(int(np.searchsorted(gx, xs.max())) + 2, gx.size - 1)
        j0 = max(int(np.searchsorted(-gy, -ys.max())) - 2, 0)   # y decrescente
        j1 = min(int(np.searchsorted(-gy, -ys.min())) + 2, gy.size - 1)
        sub = np.asarray(var[j0:j1 + 1, i0:i1 + 1])
        sx, sy = gx[i0:i1 + 1], gy[j0:j1 + 1]
    ix = np.clip(np.searchsorted(sx, xs), 0, sx.size - 1)
    iy = np.clip(np.searchsorted(-sy, -ys), 0, sy.size - 1)
    return sub[np.ix_(iy, ix)], code


# ============================================================== rótulos ======
def interior_point(binary, xs, ys):
    """Ponto mais interno de uma máscara binária (máximo da transformada de
    distância). Diferente do centróide, cai DENTRO do polígono mesmo quando ele
    é côncavo — caso de praticamente toda plataforma."""
    if not binary.any():
        return None
    dist = ndimage.distance_transform_edt(binary)
    j, i = np.unravel_index(np.argmax(dist), dist.shape)
    return float(xs[i]), float(ys[j]), float(dist[j, i])


def largest_component(sel):
    """Descarta fragmentos soltos, que puxariam a âncora de texto para um
    pedaço irrelevante do polígono."""
    lab, n = ndimage.label(sel)
    if n <= 1:
        return sel
    sizes = ndimage.sum(sel, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def shelf_label_points(mask, shelf_id, code, xs, ys, wanted):
    """Âncora de cada plataforma: polígono = BedMachine (o que é desenhado),
    nome = NSIDC-0794 (o que identifica)."""
    floating = mask == BM_FLOATING
    out = {}
    for label, key in wanted.items():
        if key not in code:
            continue
        sel = floating & (shelf_id == code[key])
        if sel.sum() < 30:
            continue
        pt = interior_point(largest_component(sel), xs, ys)
        if pt is not None:
            out[label] = pt
    return out


def upstream_point(cx, cy, shelf_xy, distance_km):
    """Ponto a montante da linha de aterramento, sobre o tronco da geleira.

    O centróide da GL fica SOBRE a linha: um rótulo ali pousa em cima do
    contorno e fica ambíguo entre os dois lados. O sentido do escoamento local
    é dado pelo par (plataforma alimentada -> GL da geleira); seguir esse vetor
    além da GL cai no gelo aterrado que drena para aquela plataforma. Fica
    determinado pelos dados — nenhuma coordenada é digitada à mão.
    """
    vx, vy = cx - shelf_xy[0], cy - shelf_xy[1]
    norm = float(np.hypot(vx, vy))
    if norm == 0:
        return cx, cy
    d = distance_km * 1e3
    return cx + d * vx / norm, cy + d * vy / norm


def glacier_label_points(shelves, names, year_min=2015):
    """Topônimo de geleira aterrada, ancorado a montante e não sobre a GL."""
    gl = gpd.read_file(INSAR_GL)
    gl = gl[gl["Year"] >= year_min]
    out = {}
    for name, (shelf_key, distance_km) in names.items():
        sub = gl[gl["Glac_Name"] == name]
        if sub.empty or shelf_key not in shelves:
            continue
        c = sub.union_all().centroid
        sx, sy, _ = shelves[shelf_key]
        px, py = upstream_point(c.x, c.y, (sx, sy), distance_km)
        out[name] = (px, py, len(sub))
    return out


# =============================================================== desenho =====
def class_rgb(mask, shade=None, strength=0.55):
    """Máscara categórica -> RGB, sem colormap (que interpolaria classes
    vizinhas na reamostragem de tela). O sombreamento modula apenas o gelo; o
    oceano fica plano, para não sugerir batimetria que não está sendo
    mostrada."""
    rgb = np.zeros(mask.shape + (3,), dtype=float)
    for c, col in [(BM_OCEAN, C_OCEAN), (BM_ROCK, C_ROCK),
                   (BM_GROUNDED, C_GROUNDED), (BM_FLOATING, C_SHELF),
                   (BM_VOSTOK, C_SHELF)]:
        rgb[mask == c] = to_rgb(col)
    if shade is not None:
        s = np.where(np.isfinite(shade), shade, 0.5)
        factor = np.clip(1.0 + strength * (2.0 * s - 1.0), 0.6, 1.10)
        ice = (mask != BM_OCEAN)[..., None]
        rgb = np.where(ice, np.clip(rgb * factor[..., None], 0, 1), rgb)
    return rgb


def roi_outline_xy(n=300):
    """Contorno da ROI em EPSG:3031, densificado: em projeção polar os lados de
    uma caixa lon/lat são curvas, não retas."""
    lon = np.concatenate([
        np.linspace(ROI_LON[0], ROI_LON[1], n), np.full(n, ROI_LON[1]),
        np.linspace(ROI_LON[1], ROI_LON[0], n), np.full(n, ROI_LON[0])])
    lat = np.concatenate([
        np.full(n, ROI_LAT[0]), np.linspace(ROI_LAT[0], ROI_LAT[1], n),
        np.full(n, ROI_LAT[1]), np.linspace(ROI_LAT[1], ROI_LAT[0], n)])
    xy = CRS_DATA.transform_points(GEO, lon, lat)
    return xy[:, 0], xy[:, 1]


def draw_contours(ax, mask, xs, ys, gl_lw=0.6, coast_lw=0.4):
    """Linha de aterramento e margem de gelo, derivadas da MESMA máscara que
    colore o mapa — desenhar um shapefile externo aqui faria a linha discordar
    da classificação mostrada."""
    grounded = np.isin(mask, (BM_ROCK, BM_GROUNDED, BM_VOSTOK)).astype(float)
    ice = (mask != BM_OCEAN).astype(float)
    ax.contour(xs, ys, ice, levels=[0.5], colors=[C_COAST],
               linewidths=coast_lw, transform=CRS_DATA, zorder=4)
    ax.contour(xs, ys, grounded, levels=[0.5], colors=[C_GL],
               linewidths=gl_lw, transform=CRS_DATA, zorder=5)


def draw_elevation_contours(ax, elev, mask, xs, ys):
    """Isolinhas de elevação só sobre gelo aterrado: sobre plataforma a
    superfície é controlada por flutuação e maré, e a isolinha ali sugeriria
    uma topografia sem o mesmo significado."""
    z = np.where(np.isin(mask, (BM_GROUNDED, BM_VOSTOK)), elev, np.nan)
    cs = ax.contour(xs, ys, z, levels=CONTOUR_LEVELS, colors=[C_CONTOUR],
                    linewidths=0.3, transform=CRS_DATA, zorder=3)
    # Só as isolinhas baixas recebem cota. Acima de 1200 m o relevo é o maciço
    # costeiro, onde as curvas se apertam e as cotas se sobrepõem — viram
    # mancha, não informação.
    ax.clabel(cs, levels=[lv for lv in CONTOUR_LEVELS if lv <= 1200],
              fmt="%d", fontsize=4.6, colors=C_FOOT, inline=True,
              inline_spacing=14)


def scale_factor(lat):
    """Fator de escala pontual da estereográfica polar com escala verdadeira em
    71°S. A barra é desenhada em metros de PROJEÇÃO; sem este fator ela erra
    ~1 % na latitude da Thwaites."""
    return ((1 + np.sin(np.deg2rad(abs(TRUE_SCALE_LAT))))
            / (1 + np.sin(np.deg2rad(abs(lat)))))


def add_scalebar(ax, x, y, length_km, lat_ref, n_seg=2, fontsize=5.8,
                 color=C_TEXT, height_frac=0.008):
    """Barra de escala em km reais na latitude `lat_ref`. `x`, `y` em fração do
    eixo."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    length_m = length_km * 1e3 * scale_factor(lat_ref)
    px, py = x0 + x * (x1 - x0), y0 + y * (y1 - y0)
    h = (y1 - y0) * height_frac
    seg = length_m / n_seg
    for k in range(n_seg):
        ax.add_patch(Rectangle(
            (px + k * seg, py), seg, h, zorder=14,
            facecolor=(color if k % 2 == 0 else "white"),
            edgecolor=color, linewidth=0.45))
    for k, val in enumerate(np.linspace(0, length_km, n_seg + 1)):
        ax.text(px + k * seg, py + h * 1.4, f"{val:g}", ha="center",
                va="bottom", fontsize=fontsize, color=color, zorder=14,
                path_effects=halo())
    ax.text(px + length_m / 2, py - h * 1.0, "km", ha="center", va="top",
            fontsize=fontsize, color=color, zorder=14, path_effects=halo())


def add_north_arrow(ax, x_frac, y_frac, length_m, crs_axis, fontsize=7.0,
                    color=C_TEXT):
    """Seta de norte calculada, não desenhada de memória.

    Em estereográfica polar o norte é radial: aponta para fora do polo sul, que
    é a origem do EPSG:3031. A direção é medida no ponto de ancoragem e a seta
    é traçada em coordenadas de DADOS — em coordenadas de eixo o ângulo sairia
    torto assim que a moldura não fosse quadrada.

    Ela só é honesta em recorte pequeno: através dos 20° de longitude deste
    mapa a direção do norte gira ~20°, então a seta vale para o entorno do
    ponto onde está, não para a folha inteira.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    px, py = x0 + x_frac * (x1 - x0), y0 + y_frac * (y1 - y0)

    nat = CRS_DATA.transform_points(crs_axis, np.array([px]), np.array([py]))[0]
    versor = np.array([nat[0], nat[1]], dtype=float)
    versor /= np.linalg.norm(versor)
    ponta = CRS_DATA.transform_points(
        crs_axis,
        np.array([nat[0] + versor[0] * length_m]),
        np.array([nat[1] + versor[1] * length_m]))[0]
    rotulo = CRS_DATA.transform_points(
        crs_axis,
        np.array([nat[0] + versor[0] * length_m * 1.32]),
        np.array([nat[1] + versor[1] * length_m * 1.32]))[0]

    ax.annotate("", xy=(ponta[0], ponta[1]), xytext=(px, py),
                xycoords="data", textcoords="data", zorder=13,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9,
                                mutation_scale=7, shrinkA=0, shrinkB=0))
    ax.text(rotulo[0], rotulo[1], "N", ha="center", va="center",
            fontsize=fontsize, weight="bold", color=color, zorder=13,
            path_effects=halo())


def _borda(px, py, x0, x1, y0, y1, tol):
    """Em qual borda da moldura este ponto encosta."""
    if abs(px - x0) <= tol:
        return "esquerda"
    if abs(px - x1) <= tol:
        return "direita"
    if abs(py - y0) <= tol:
        return "inferior"
    if abs(py - y1) <= tol:
        return "superior"
    return None


def label_graticule(ax, crs_axis, lons=(), lats=(), fontsize=6.2,
                    color="#55606a", pad_frac=0.012, evitar=("superior",
                                                             "direita"),
                    min_sep_frac=0.055, tick_frac=0.0):
    """Rotula meridianos e paralelos onde cruzam a moldura.

    O seletor do cartopy não serve em nenhuma das duas projeções usadas aqui:
    na girada não produz rótulo de latitude algum, e na nativa do EPSG:3031
    emite dois de cinco meridianos e nenhum paralelo — as linhas cruzam a
    moldura em diagonal e o algoritmo desiste. Aqui cada linha é percorrida, o
    ponto em que sai da moldura é localizado e o rótulo vai do lado de fora
    daquela borda.

    `evitar` lista bordas ocupadas por outros elementos (inset, legenda): se a
    linha sai por uma delas, usa-se a outra extremidade. `min_sep_frac` suprime
    rótulos que cairiam em cima de um já posto — perto do polo os meridianos
    convergem e se amontoam.

    `tick_frac` desenha um tique curto para dentro da moldura, no lugar da
    linha de grade: sem ele o rótulo fica solto e não se sabe a que ponto da
    borda corresponde, já que aqui as linhas chegam em diagonal.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    tol = 0.004 * max(x1 - x0, y1 - y0)
    pad = pad_frac * (x1 - x0)
    postos = []

    def texto(px, py, borda, label):
        for qx, qy in postos:
            if (abs(px - qx) < min_sep_frac * (x1 - x0)
                    and abs(py - qy) < min_sep_frac * (y1 - y0)):
                return
        dx, dy, ha, va = 0.0, 0.0, "center", "center"
        if borda == "esquerda":
            dx, ha = -pad, "right"
        elif borda == "direita":
            dx, ha = pad, "left"
        elif borda == "inferior":
            dy, va = -pad, "top"
        else:
            dy, va = pad, "bottom"
        if tick_frac:
            t = tick_frac * (x1 - x0)
            tx = -t if borda == "direita" else (t if borda == "esquerda" else 0)
            ty = -t if borda == "superior" else (t if borda == "inferior" else 0)
            ax.plot([px, px + tx], [py, py + ty], color=color, lw=0.6,
                    solid_capstyle="butt", zorder=12, clip_on=False)
        ax.text(px + dx, py + dy, label, ha=ha, va=va, fontsize=fontsize,
                color=color, clip_on=False, zorder=12)
        postos.append((px, py))

    def percorre(lon_arr, lat_arr, label):
        pts = crs_axis.transform_points(GEO, lon_arr, lat_arr)
        px, py = pts[:, 0], pts[:, 1]
        vis = (px >= x0 - tol) & (px <= x1 + tol) & \
              (py >= y0 - tol) & (py <= y1 + tol)
        if not vis.any():
            return
        idx = np.where(vis)[0]
        candidatos = []
        for k in (idx[0], idx[-1]):
            b = _borda(px[k], py[k], x0, x1, y0, y1, tol * 3)
            if b is not None:
                candidatos.append((b, px[k], py[k]))
        if not candidatos:
            return
        preferidos = [c for c in candidatos if c[0] not in evitar]
        if not preferidos:
            # A linha só sai por borda ocupada. Desenhar o tique mesmo assim
            # deixaria uma marca sem cota — um traço órfão na moldura.
            return
        b, qx, qy = preferidos[0]
        texto(qx, qy, b, label)

    n = 4000
    for lon in lons:
        lat_arr = np.linspace(-79.5, -71.0, n)
        percorre(np.full(n, float(lon)), lat_arr,
                 f"{abs(lon):g}°{'W' if lon < 0 else 'E'}")
    for lat in lats:
        lon_arr = np.linspace(-132.0, -84.0, n)
        percorre(lon_arr, np.full(n, float(lat)), f"{abs(lat):g}°S")


def label_parallels(ax, lats, fontsize=6.0, color="#4a555f", pad_frac=0.008,
                    crs_axis=None):
    """Rótulo dos paralelos na borda esquerda.

    Na projeção girada do painel b o cartopy não gera rótulo de latitude
    nenhum (`left_label_artists` volta vazio): os paralelos cruzam a moldura
    em diagonal e o seletor automático os descarta. Aqui o ponto de rótulo é
    calculado — a interseção de cada paralelo com a borda esquerda visível.
    """
    crs_axis = crs_axis or CRS_ASE
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lons = np.linspace(-130.0, -85.0, 4000)
    for lat in lats:
        pts = crs_axis.transform_points(GEO, lons, np.full_like(lons, lat))
        px, py = pts[:, 0], pts[:, 1]
        vis = (py > y0) & (py < y1) & (px >= x0) & (px <= x1)
        if not vis.any():
            continue
        i = int(np.argmin(np.where(vis, px, np.inf)))
        ax.text(x0 - pad_frac * (x1 - x0), py[i], f"{abs(lat):g}°S",
                ha="right", va="center", fontsize=fontsize, color=color,
                clip_on=False, zorder=12)


def halo(size=1.8, color="white"):
    return [pe.withStroke(linewidth=size, foreground=color)]


def place(ax, x, y, text, size=7.0, color=C_TEXT, style="normal",
          weight="normal", ha="center", va="center", rotation=0.0,
          crs=CRS_DATA, zorder=11):
    """Topônimo com halo branco: sobre o sombreamento, texto sem halo some."""
    return ax.text(x, y, text, transform=crs, ha=ha, va=va, fontsize=size,
                   color=color, style=style, weight=weight, rotation=rotation,
                   zorder=zorder, linespacing=1.15, path_effects=halo())


# ========================================================== painel a =========
def continent_layers(target_px=1100, margin_m=90_000.0):
    """Máscara continental e a moldura que a cinge.

    O extent é medido no próprio dado e devolvido ANTES de a página ser
    montada: é ele que fixa a proporção do painel a. Calculá-lo só na hora de
    desenhar deixaria o eixo com uma proporção que não é a do mapa, e o
    cartopy preencheria a diferença com faixas brancas.
    """
    mask, extent = read_bedmachine(None, target_px=target_px)
    xs, ys = grid_centers(mask.shape, extent)
    ice = mask != BM_OCEAN
    # A moldura vem do maior componente conexo, não de todo pixel de gelo: o
    # BedMachine classifica ilhas subantárticas isoladas até y = +3,31e6 m
    # (~62°S), e o bounding box de tudo estica o mapa em ~800 km de oceano
    # vazio. As ilhas continuam desenhadas — só não mandam no enquadramento.
    jj, ii = np.where(largest_component(ice))
    frame = [float(xs[ii.min()] - margin_m), float(xs[ii.max()] + margin_m),
             float(ys[jj.max()] - margin_m), float(ys[jj.min()] + margin_m)]
    return {"mask": mask, "extent": extent, "xs": xs, "ys": ys, "ice": ice,
            "frame": frame}


def label_meridians(ax, lons, fontsize=5.6, color="#7b8896", pad_frac=0.018):
    """Rótulo dos meridianos junto à moldura.

    Substitui a rosa dos ventos: em estereográfica polar o norte é radial, e
    uma seta "N" única seria falsa em todo ponto fora do meridiano em que
    fosse desenhada. Os meridianos dão a orientação sem essa mentira.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lats = np.linspace(-89.5, -55.0, 3000)
    for lon in lons:
        pts = CRS_DATA.transform_points(GEO, np.full_like(lats, lon), lats)
        px, py = pts[:, 0], pts[:, 1]
        vis = (px > x0) & (px < x1) & (py > y0) & (py < y1)
        if not vis.any():
            continue
        k = int(np.where(vis)[0][-1])          # saída da moldura
        ux, uy = px[k] - px[k - 1], py[k] - py[k - 1]
        n = float(np.hypot(ux, uy)) or 1.0
        pad = pad_frac * (x1 - x0)
        label = ("0°" if lon == 0 else "180°" if abs(lon) == 180 else
                 f"{abs(lon):g}°{'W' if lon < 0 else 'E'}")
        ax.text(px[k] + pad * ux / n, py[k] + pad * uy / n, label,
                ha="center", va="center", fontsize=fontsize, color=color,
                clip_on=False, zorder=8)


def draw_continent(ax, layers, rx, ry, arrow_target=None, fs=1.0,
                   meridian_labels=True, frame_box=True, grid=True,
                   destaque="mancha", rotulos=True, escala=True,
                   relevo=False, isolinhas_m=None):
    """Antártica inteira, com o domínio de estudo destacado.

    O destaque em azul é o GELO (aterrado ou flutuante) contido no domínio de
    processamento — não uma bacia de drenagem: este projeto não carrega um
    polígono de bacia, e desenhar um contorno de bacia "de memória" inventaria
    uma fronteira que nenhum dado em disco sustenta.

    `fs` escala a tipografia: a mesma função serve a figura de página inteira e
    ao painel pequeno do layout duplo.
    """
    mask, extent = layers["mask"], layers["extent"]
    xs, ys, ice = layers["xs"], layers["ys"], layers["ice"]

    ax.set_extent(layers["frame"], crs=CRS_DATA)
    ax.patch.set_facecolor("white")

    rgb = np.ones(mask.shape + (3,), dtype=float)
    rgb[np.isin(mask, (BM_ROCK, BM_GROUNDED, BM_VOSTOK))] = to_rgb(C_CONT_ICE)
    rgb[mask == BM_FLOATING] = to_rgb(C_CONT_SHELF)

    elev = None
    if relevo or isolinhas_m:
        elev = read_bedmachine_surface(xs, ys)
    if relevo:
        res_m = abs(float(xs[1] - xs[0]))
        # Exagero alto pela mesma razão do mapa regional, e mais ainda: a
        # declividade mediana do manto nesta grade de 4 km é 0,29 %. O relevo
        # aqui é textura declarada, não medida.
        shade = hillshade(np.where(ice, elev, np.nan), res_m, vert_exag=70.0)
        if shade is not None:
            s = np.where(np.isfinite(shade), shade, 0.5)
            fator = np.clip(1.0 + 0.38 * (2.0 * s - 1.0), 0.70, 1.06)
            rgb = np.where(ice[..., None],
                           np.clip(rgb * fator[..., None], 0, 1), rgb)

    X, Y = np.meshgrid(xs, ys)
    inside = MplPath(np.column_stack([rx, ry])).contains_points(
        np.column_stack([X.ravel(), Y.ravel()])).reshape(mask.shape)
    hi = inside & ice
    if destaque == "mancha":
        rgb[hi] = to_rgb(C_HIGHLIGHT)
    hj, hii = np.where(hi)
    hx, hy = float(xs[hii].mean()), float(ys[hj].mean())
    print(f"[mapa] gelo no domínio: {int(hi.sum())} px "
          f"(centro x={hx:.0f} y={hy:.0f})  destaque={destaque}")

    ax.imshow(rgb, extent=extent, origin="upper", transform=CRS_DATA,
              interpolation="nearest", zorder=1)
    if isolinhas_m:
        # Só sobre gelo aterrado, pela mesma razão do mapa regional: sobre
        # plataforma a superfície é controlada por flutuação e maré.
        z = np.where(np.isin(mask, (BM_GROUNDED, BM_VOSTOK)), elev, np.nan)
        ax.contour(xs, ys, z,
                   levels=np.arange(isolinhas_m, 4500, isolinhas_m),
                   colors=[C_CONTOUR], linewidths=0.25 * fs,
                   transform=CRS_DATA, zorder=2)

    ax.contour(xs, ys, ice.astype(float), levels=[0.5], colors=[C_CONT_EDGE],
               linewidths=0.35 * fs, transform=CRS_DATA, zorder=3)
    # Linha de aterramento: separa o que flutua do que se apoia no leito, e é
    # a mesma fronteira que define a máscara científica do pipeline.
    ax.contour(xs, ys,
               np.isin(mask, (BM_ROCK, BM_GROUNDED, BM_VOSTOK)).astype(float),
               levels=[0.5], colors=["#7e8b96"], linewidths=0.3 * fs,
               transform=CRS_DATA, zorder=3)

    if grid:
        gl = ax.gridlines(linewidth=0.25, color="#aebcc7", alpha=0.8,
                          linestyle=(0, (1, 3)),
                          xlocs=np.arange(-180, 180, 30),
                          ylocs=[-80, -70, -60], zorder=3)
        gl.top_labels = gl.bottom_labels = False
        gl.left_labels = gl.right_labels = False
    if meridian_labels:
        label_meridians(ax, [0, -30, -60, -90, -120, -150, 180, 150, 120,
                             90, 60, 30], fontsize=4.2 * fs)

    if destaque == "poligono":
        # A área entra como polígono, não como mancha de gelo pintada: o que se
        # localiza aqui é o RECORTE do estudo. Um preenchimento opaco esconderia
        # a geometria contida nele, que é o assunto da outra figura.
        ax.fill(rx, ry, facecolor=C_BOX, alpha=0.15, edgecolor="none",
                transform=CRS_DATA, zorder=4)
        ax.plot(rx, ry, color=C_BOX, lw=1.3 * fs, transform=CRS_DATA,
                zorder=5, solid_joinstyle="round")
    else:
        ax.plot(rx, ry, color=C_BOX, lw=0.9 * fs, ls=(0, (4, 2)),
                transform=CRS_DATA, zorder=5)

    if rotulos:
        ax.text(0.60, 0.55, "ANTÁRTICA", transform=ax.transAxes, ha="center",
                va="center", fontsize=7.6 * fs, color="#5f6f7c", weight="bold")
        ax.text(0.085, 0.545, "Mar de\nAmundsen", transform=ax.transAxes,
                ha="center", va="center", fontsize=5.6 * fs, color=C_SEA_TEXT,
                style="italic", linespacing=1.15)
    # A seta aponta para a PRÓPRIA Thwaites, não para o centro da mancha: o
    # destaque cobre todo o domínio (Thwaites, Pine Island, Getz), e uma seta
    # ao centro dele rotularia de "Thwaites" o conjunto inteiro. O alvo é a
    # plataforma da Thwaites, localizada na máscara; sem ela, cai no centro da
    # mancha, que ao menos não aponta para fora da área.
    if rotulos:
        tx, ty = arrow_target if arrow_target is not None else (hx, hy)
        ax.annotate("Geleira\nThwaites", xy=(tx, ty),
                    xycoords=CRS_DATA._as_mpl_transform(ax),
                    xytext=(0.105, 0.215), textcoords=ax.transAxes,
                    ha="center", va="center", fontsize=6.8 * fs, color=C_TEXT,
                    weight="bold", linespacing=1.15,
                    arrowprops=dict(arrowstyle="-|>", color=C_TEXT,
                                    lw=0.65 * fs, mutation_scale=6 * fs,
                                    shrinkA=2, shrinkB=2),
                    zorder=6)
    if escala:
        add_scalebar(ax, 0.62, 0.055, 2000, lat_ref=TRUE_SCALE_LAT, n_seg=2,
                     fontsize=5.4 * fs, height_frac=0.012)
    for spine in ax.spines.values():
        spine.set_visible(frame_box)
        spine.set_linewidth(0.5)
        spine.set_edgecolor("#9aa8b4")


def locate_thwaites_shelf():
    """Centro da plataforma da Thwaites, para o alvo da seta.

    Lê um recorte pequeno — a figura continental não precisa do mosaico
    regional inteiro só para posicionar uma seta.
    """
    rx, ry = roi_outline_xy()
    bounds = (rx.min(), ry.min(), rx.max(), ry.max())
    mask, extent = read_bedmachine(bounds, target_px=700)
    xs, ys = grid_centers(mask.shape, extent)
    shelf_id, code = sample_iceshelf_id(xs, ys)
    found = shelf_label_points(mask, shelf_id, code, xs, ys,
                               {"Thwaites": "Thwaites"})
    pt = found.get("Thwaites")
    return None if pt is None else (pt[0], pt[1])


# =================================================== figura: continental =====
def build_continental():
    """Figura única: a Antártica com o domínio de estudo destacado."""
    setup_style()
    rx, ry = roi_outline_xy()
    cont = continent_layers(target_px=1600)
    target = locate_thwaites_shelf()
    print(f"[mapa] alvo da seta (plataforma Thwaites): {target}")

    frame = cont["frame"]
    aspect = (frame[3] - frame[2]) / (frame[1] - frame[0])
    print(f"[mapa] moldura {[round(v) for v in frame]}  aspecto {aspect:.3f}")

    # 150 mm, a mesma largura da figura regional: as duas formam par e devem
    # sair na mesma escala tipográfica.
    W = 5.9
    fs = 1.45
    left, right, top, bottom = 0.14, 0.14, 0.12, 0.40
    w_map = W - left - right
    h_map = w_map * aspect
    H = top + h_map + bottom

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([left / W, bottom / H, w_map / W, h_map / H],
                      projection=CRS_DATA)
    # Oceano branco, sem grade e sem rótulo de meridiano: o mapa existe para
    # situar o recorte, e cada traço a mais compete com o polígono.
    # Sem curvas de nível: o relevo fica por conta do sombreado, e as linhas
    # que sobram são as de contorno (margem de gelo e linha de aterramento),
    # que delimitam feição e não cota.
    draw_continent(ax, cont, rx, ry, arrow_target=target, fs=fs,
                   meridian_labels=False, grid=False, destaque="poligono",
                   escala=False, relevo=True, isolinhas_m=None)

    fig.text(0.5, 0.030,
             "Superfície e máscara: BedMachine Antarctica v4 (~2019); relevo "
             "sombreado derivado da mesma superfície.  "
             "Projeção estereográfica polar sul (EPSG:3031).",
             ha="center", va="bottom", fontsize=5.0, color=C_FOOT)

    print(f"[fig] {W:.2f} x {H:.2f} in")
    save(fig, "fig_area_estudo_continental")


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.6, "savefig.facecolor": "white",
    })


def save(fig, stem):
    OUTDIR.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        out = OUTDIR / f"{stem}.{ext}"
        fig.savefig(out, dpi=600 if ext == "png" else None)
        print("->", out)


# ====================================================== figura: regional =====
# Topônimos de REGIÃO. Não delimitam nada e não saem de nenhum arquivo: são
# âncoras de orientação, dadas em lon/lat (não em coordenada de tela) para que
# a posição seja conferível num atlas. Marie Byrd Land e Ellsworth Land são
# províncias amplas; o ponto marca onde o nome cabe no mapa, não um centróide.
REGIOES = {
    "Terra de\nMarie Byrd": (-112.0, -76.8),
    "Terra de\nEllsworth": (-96.5, -76.2),
}


def draw_inset_antartica(fig, rect, rx, ry):
    """Inset de localização: silhueta do continente e a caixa do domínio."""
    cont = continent_layers(target_px=700)
    mask, extent = cont["mask"], cont["extent"]
    xs, ys, ice = cont["xs"], cont["ys"], cont["ice"]

    ax = fig.add_axes(rect, projection=CRS_DATA)
    ax.set_extent(cont["frame"], crs=CRS_DATA)
    ax.patch.set_facecolor("white")
    ax.patch.set_alpha(0.92)

    rgb = np.ones(mask.shape + (3,), dtype=float)
    rgb[ice] = to_rgb(C_CONT_ICE)
    rgb[mask == BM_FLOATING] = to_rgb(C_CONT_SHELF)
    ax.imshow(rgb, extent=extent, origin="upper", transform=CRS_DATA,
              interpolation="nearest", zorder=1)
    ax.contour(xs, ys, ice.astype(float), levels=[0.5], colors=[C_CONT_EDGE],
               linewidths=0.3, transform=CRS_DATA, zorder=2)
    ax.plot(rx, ry, color=C_BOX, lw=0.8, transform=CRS_DATA, zorder=3)
    ax.fill(rx, ry, color=C_BOX, alpha=0.35, transform=CRS_DATA, zorder=3)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.4)
        spine.set_edgecolor("#b3bec8")
    return ax


def build_regional():
    """Figura única do setor do Mar de Amundsen, em chave de periódico.

    Sem caixa de legenda: cada feição é identificada pelo próprio topônimo, e
    a distinção aterrado/flutuante fica na linha de aterramento — o que a
    legenda diria, a caption diz melhor.
    """
    setup_style()
    rx, ry = roi_outline_xy()
    pad = 24_000.0
    bounds = (rx.min() - pad, ry.min() - pad, rx.max() + pad, ry.max() + pad)

    mask, extent = read_bedmachine(bounds, target_px=1700)
    xs, ys = grid_centers(mask.shape, extent)
    res_m = (extent[1] - extent[0]) / mask.shape[1]
    elev = read_rema(mask.shape, extent)
    shade = hillshade(elev, res_m)
    shelf_id, code = sample_iceshelf_id(xs, ys)

    shelves = shelf_label_points(
        mask, shelf_id, code, xs, ys,
        {"Thwaites": "Thwaites", "Pine Island": "Pine_Island",
         "Crosson": "Crosson", "Dotson": "Dotson", "Cosgrove": "Cosgrove",
         "Getz": "Getz"})
    glaciers = glacier_label_points(
        shelves, {"Thwaites": ("Thwaites", 38.0),
                  "Pine Island": ("Pine Island", 45.0),
                  "Pope Kohler Smith": ("Crosson", 42.0)})

    # A Getz entra no recorte só por uma ponta: a âncora cai colada na borda e
    # o rótulo sai cortado. A plataforma segue desenhada, sem nome.
    rotulados = {k: v for k, v in shelves.items() if k != "Getz"}

    print(f"[regional] grade {mask.shape}  {res_m:.0f} m/px  "
          f"REMA valido {np.isfinite(elev).mean():.1%}")

    # Projeção NATIVA do EPSG:3031 (meridiano central 0°), não a girada.
    # Nesta orientação o norte não aponta para cima: ele é radial e, no centro
    # deste recorte, cai a ~165° (esquerda, levemente abaixo). Por isso a
    # figura leva seta de norte — sem ela, o leitor assume "cima = norte" e
    # erra o sentido do escoamento das geleiras.
    crs_axis = CRS_DATA
    m = 0.015 * (rx.max() - rx.min())
    ext = [rx.min() - m, rx.max() + m, ry.min() - m, ry.max() + m]
    aspect = (ext[3] - ext[2]) / (ext[1] - ext[0])

    W = 5.9                      # 150 mm: a moldura é retrato, não paisagem
    left, right, top, bottom = 0.40, 0.10, 0.10, 0.44
    w_map = W - left - right
    h_map = w_map * aspect
    H = top + h_map + bottom
    print(f"[fig] {W:.2f} x {H:.2f} in   aspecto {aspect:.3f}")

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([left / W, bottom / H, w_map / W, h_map / H],
                      projection=crs_axis)
    ax.set_extent(ext, crs=crs_axis)

    ax.imshow(class_rgb(mask, shade), extent=extent, origin="upper",
              transform=CRS_DATA, interpolation="nearest", zorder=1)
    draw_elevation_contours(ax, elev, mask, xs, ys)
    draw_contours(ax, mask, xs, ys, gl_lw=0.7, coast_lw=0.45)

    # Sem linhas de grade sobre o mapa: só o tique na moldura e a cota. A
    # referência geográfica continua auditável sem riscar o relevo.
    # Bordas superior e direita ficam livres — lá estão o inset e o respiro do
    # interior do manto.
    label_graticule(ax, crs_axis,
                    lons=np.arange(-115, -94, 5), lats=[-74, -75, -76, -77],
                    evitar=("superior", "direita"), tick_frac=0.009)

    # --- topônimos ---------------------------------------------------------
    for nome, (lx, ly, _) in rotulados.items():
        place(ax, lx, ly, f"Plataforma\n{nome}", size=6.2,
              color=C_SHELF_TEXT, style="italic")
    for nome, (lx, ly, _) in glaciers.items():
        rotulo = {"Pope Kohler Smith": "Geleiras\nPope/Smith/Kohler"}.get(
            nome, f"Geleira\n{nome}")
        place(ax, lx, ly, rotulo, size=7.0, weight="bold")

    for nome, (lon, lat) in REGIOES.items():
        xy = CRS_DATA.transform_points(GEO, np.array([lon]), np.array([lat]))
        place(ax, float(xy[0, 0]), float(xy[0, 1]), nome, size=6.6,
              color=C_REGION)

    place(ax, 0.245, 0.47, "MAR DE\nAMUNDSEN", size=8.5, color=C_SEA_TEXT,
          style="italic", crs=ax.transAxes, zorder=10)

    add_scalebar(ax, 0.055, 0.075, 100, lat_ref=-75.0, fontsize=6.0)
    draw_inset_antartica(fig, [(left + w_map - 1.62) / W,
                               (bottom + h_map - 1.34) / H,
                               1.52 / W, 1.24 / H], rx, ry)

    fig.text(left / W, 0.012,
             "Geometria de superficie: BedMachine Antarctica v4 (~2019); "
             "relevo e isolinhas (400 m): REMA. "
             "Projecao estereografica polar sul (EPSG:3031).",
             ha="left", va="bottom", fontsize=5.2, color=C_FOOT)

    save(fig, "fig_area_estudo_regional")


# ========================================================= figura: duplo =====
def build_dual(diagnostic=False):
    setup_style()

    rx, ry = roi_outline_xy()
    pad = 24_000.0
    bounds = (rx.min() - pad, ry.min() - pad, rx.max() + pad, ry.max() + pad)

    mask, extent = read_bedmachine(bounds, target_px=1700)
    xs, ys = grid_centers(mask.shape, extent)
    res_m = (extent[1] - extent[0]) / mask.shape[1]
    elev = read_rema(mask.shape, extent)
    shade = hillshade(elev, res_m)
    shelf_id, code = sample_iceshelf_id(xs, ys)

    shelves = shelf_label_points(
        mask, shelf_id, code, xs, ys,
        {"Thwaites": "Thwaites", "Pine Island": "Pine_Island",
         "Crosson": "Crosson", "Dotson": "Dotson", "Cosgrove": "Cosgrove",
         "Getz": "Getz"})
    # (plataforma alimentada, afastamento do rótulo a montante da GL, km).
    # Smith/Pope/Kohler drenam para a Crosson.
    glaciers = glacier_label_points(
        shelves, {"Thwaites": ("Thwaites", 38.0),
                  "Pine Island": ("Pine Island", 45.0),
                  "Pope Kohler Smith": ("Crosson", 42.0)})

    print(f"[b] mask {mask.shape}  extent {[round(v) for v in extent]}  "
          f"{res_m:.0f} m/px")
    vals, counts = np.unique(mask, return_counts=True)
    print(f"[b] classes {dict(zip(vals.tolist(), counts.tolist()))}")
    print(f"[b] REMA válido: {np.isfinite(elev).mean():.1%}  "
          f"hillshade: {'ok' if shade is not None else 'INDISPONÍVEL'}")
    for k, v in sorted(shelves.items()):
        print(f"  plataforma {k:12s} x={v[0]:10.0f} y={v[1]:10.0f}")
    for k, v in sorted(glaciers.items()):
        print(f"  geleira    {k:18s} x={v[0]:10.0f} y={v[1]:10.0f} n={v[2]}")

    # ---- geometria da página ------------------------------------------------
    pts = CRS_ASE.transform_points(CRS_DATA, rx, ry)
    bx, by = pts[:, 0], pts[:, 1]
    m = 0.018 * (bx.max() - bx.min())
    ext_b = [bx.min() - m, bx.max() + m, by.min() - m, by.max() + m]
    aspect_b = (ext_b[3] - ext_b[2]) / (ext_b[1] - ext_b[0])

    cont = continent_layers()
    ext_a = cont["frame"]
    aspect_a = (ext_a[3] - ext_a[2]) / (ext_a[1] - ext_a[0])

    # Largura de coluna dupla da Nature = 183 mm.
    W = 7.2
    left, right, top, bottom, gap = 0.44, 0.10, 0.26, 0.42, 0.10
    w_a = 2.60
    h_a = w_a * aspect_a
    w_b = W - left - right
    h_b = w_b * aspect_b
    H = top + h_a + gap + h_b + bottom
    print(f"[fig] {W:.2f} x {H:.2f} in  (a {w_a:.2f}x{h_a:.2f}, "
          f"b {w_b:.2f}x{h_b:.2f})")

    fig = plt.figure(figsize=(W, H))
    ax_a = fig.add_axes([left / W, (bottom + h_b + gap) / H, w_a / W, h_a / H],
                        projection=CRS_DATA)
    ax_b = fig.add_axes([left / W, bottom / H, w_b / W, h_b / H],
                        projection=CRS_ASE)

    thw = shelves.get("Thwaites")
    draw_continent(ax_a, cont, rx, ry, fs=0.95, meridian_labels=False,
                   frame_box=False,
                   arrow_target=None if thw is None else (thw[0], thw[1]))

    ax_b.set_extent(ext_b, crs=CRS_ASE)
    ax_b.imshow(class_rgb(mask, shade), extent=extent, origin="upper",
                transform=CRS_DATA, interpolation="nearest", zorder=1)
    draw_elevation_contours(ax_b, elev, mask, xs, ys)
    draw_contours(ax_b, mask, xs, ys)
    ax_b.plot(rx, ry, color=C_BOX, lw=0.9, ls=(0, (5, 2.5)),
              transform=CRS_DATA, zorder=7)

    # Dois gridliners: o cartopy só rotula bem os meridianos nesta projeção
    # girada, e pedir rótulo de latitude no mesmo objeto deixa escapar um
    # rótulo solto, fora de estilo, na borda curva. Os paralelos são
    # desenhados sem rótulo e rotulados por `label_parallels`.
    grid_kw = dict(linewidth=0.3, color="#93a0aa", alpha=0.6,
                   linestyle=(0, (1, 3)), zorder=6)
    gl = ax_b.gridlines(draw_labels=True, xlocs=np.arange(-115, -94, 5),
                        ylocs=[], **grid_kw)
    gl.top_labels = gl.right_labels = gl.left_labels = False
    gl.rotate_labels = False
    gl.xlabel_style = {"size": 6.0, "color": "#4a555f"}
    ax_b.gridlines(draw_labels=False, xlocs=[],
                   ylocs=np.arange(-77, -72, 1), **grid_kw)
    label_parallels(ax_b, [-74, -75, -76, -77])

    if diagnostic:
        for k, v in shelves.items():
            ax_b.plot(v[0], v[1], "o", ms=3, color="red", transform=CRS_DATA,
                      zorder=10)
            ax_b.text(v[0], v[1], k, fontsize=7, color="red",
                      transform=CRS_DATA, zorder=10)
        for k, v in glaciers.items():
            ax_b.plot(v[0], v[1], "s", ms=3, color="blue", transform=CRS_DATA,
                      zorder=10)
            ax_b.text(v[0], v[1], k, fontsize=7, color="blue",
                      transform=CRS_DATA, zorder=10)
        OUTDIR.mkdir(exist_ok=True)
        out = OUTDIR / "fig1_area_estudo_DIAG.png"
        fig.savefig(out, dpi=130)
        print("->", out)
        return

    # ---- topônimos do painel b ---------------------------------------------
    for name, (lx, ly, _) in shelves.items():
        place(ax_b, lx, ly, f"Plataforma\n{name}", size=6.0,
              color=C_SHELF_TEXT, style="italic")
    for name, (lx, ly, _) in glaciers.items():
        label = {"Pope Kohler Smith": "Geleiras\nPope/Smith/Kohler"}.get(
            name, f"Geleira\n{name}")
        place(ax_b, lx, ly, label, size=6.8, weight="bold")

    place(ax_b, 0.50, 0.87, "MAR DE AMUNDSEN", size=7.4, color=C_SEA_TEXT,
          style="italic", crs=ax_b.transAxes, zorder=10)

    add_scalebar(ax_b, 0.815, 0.055, 100, lat_ref=-75.0)

    handles = [
        Patch(facecolor=C_GROUNDED, edgecolor="#b9c3cb", lw=0.4,
              label="Gelo aterrado"),
        Patch(facecolor=C_SHELF, edgecolor="#7fb2d4", lw=0.4,
              label="Plataforma flutuante"),
        Patch(facecolor=C_OCEAN, edgecolor="#b9c3cb", lw=0.4, label="Oceano"),
        Patch(facecolor=C_ROCK, edgecolor="#7d756d", lw=0.4,
              label="Rocha exposta"),
        Line2D([], [], color=C_GL, lw=0.8, label="Linha de aterramento"),
        Line2D([], [], color=C_COAST, lw=0.8, label="Margem de gelo"),
        Line2D([], [], color=C_CONTOUR, lw=0.6, label="Isolinha REMA (400 m)"),
        Line2D([], [], color=C_BOX, lw=1.0, ls=(0, (5, 2.5)),
               label="Domínio de processamento"),
    ]
    # A legenda ocupa o vazio à direita do painel a, em vez de cobrir o mapa.
    leg = ax_a.legend(handles=handles, loc="center left", fontsize=6.6,
                      frameon=False, labelspacing=0.72, handlelength=1.7,
                      handletextpad=0.75, bbox_to_anchor=(1.10, 0.50))
    leg.set_zorder(15)

    for axis, tag in ((ax_a, "a"), (ax_b, "b")):
        axis.text(-0.012, 1.012, tag, transform=axis.transAxes, ha="left",
                  va="bottom", fontsize=9, weight="bold", zorder=16)
    fig.text(left / W, 0.010, EPOCA_NOTA, ha="left", va="bottom",
             fontsize=5.2, color=C_FOOT, wrap=True)

    save(fig, "fig1_area_estudo_duplo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=("continental", "regional", "duplo"),
                    default="continental",
                    help="continental: só o mapa da Antártica (padrão); "
                         "regional: só o setor do Mar de Amundsen; "
                         "duplo: os dois painéis numa figura")
    ap.add_argument("--diagnostic", action="store_true",
                    help="layout duplo: marca as âncoras de rótulo e pula o "
                         "acabamento")
    args = ap.parse_args()
    if args.layout == "continental":
        build_continental()
    elif args.layout == "regional":
        build_regional()
    else:
        build_dual(diagnostic=args.diagnostic)


if __name__ == "__main__":
    main()
