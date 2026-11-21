"""Processamento de ApRES (radar FMCW de fase) para taxa de derretimento basal.

Método padrão (Brennan et al. 2014, IET RSN, doi:10.1049/iet-rsn.2013.0053; Nicholls et al. 2015,
J. Glaciol., doi:10.3189/2015JoG15J073):

1. perfil complexo de alcance: chirp médio do disparo, sem média, janela de Blackman,
   preenchimento ×p, rotação de meia janela, FFT e correção de fase de referência
   φ_ref = 2π·fc·n/(B·p) − K·n²/(2·B²·p²);
2. deslocamento entre duas datas em janelas de profundidade: correlação complexa com busca de
   defasagem inteira (grosseira) e fase no máximo (fina): Δr = lag·dR + φ·λc/(4π);
3. deformação vertical: ajuste linear Δr(z) = a + b·z nas camadas internas coerentes;
4. derretimento no período: m = (a + b·z_base) − Δr_base (o fundo sobe além do previsto pela
   deformação quando há derretimento).

Convenção de sinal verificada com chirp sintético: Δr = −angle(s1·conj(s2))·λc/(4π).

Parâmetros dos arquivos do ITGC MELT (atributos do disparo): 200–400 MHz, chirp de 1 s, 40.000
amostras, ε = 3,18.
"""

from __future__ import annotations

import numpy as np

C = 299_792_458.0


class Config:
    def __init__(self, f0=2e8, f1=4e8, T=1.0, n=40000, er=3.18, pad=2):
        self.B = f1 - f0
        self.fc = 0.5 * (f0 + f1)
        self.K = self.B / T
        self.n = n
        self.er = er
        self.ci = C / np.sqrt(er)
        self.p = pad
        self.lam_c = self.ci / self.fc                       # comprimento de onda no gelo
        self.dR = self.ci / (2 * self.B * pad)               # tamanho do bin de alcance


def range_profile(chirp, cfg: Config):
    v = np.asarray(chirp, float)
    v = v - v.mean()
    win = np.blackman(len(v))
    v = v * win
    pad = np.zeros(cfg.p * len(v))
    pad[: len(v)] = v
    pad = np.roll(pad, -len(v) // 2)
    spec = np.sqrt(2 * cfg.p) / len(pad) * np.fft.fft(pad) / np.sqrt(np.mean(win ** 2))
    nf = len(pad) // 2
    spec = spec[:nf]
    nn = np.arange(nf)
    phiref = 2 * np.pi * cfg.fc * nn / (cfg.B * cfg.p) - cfg.K * nn ** 2 / (2 * cfg.B ** 2 * cfg.p ** 2)
    return nn * cfg.dR, spec * np.exp(-1j * phiref)


def file_profile(nc_path, cfg: Config):
    """Perfil complexo médio de todos os disparos de um arquivo (média coerente)."""
    import netCDF4
    profs, times = [], []
    with netCDF4.Dataset(nc_path) as d:
        for g in d.groups.values():
            data = np.asarray(g["data"][:], float)
            R, s = range_profile(data.mean(0), cfg)
            profs.append(s)
            times.append(np.datetime64(getattr(g, "Time stamp").replace(" ", "T")))
    return R, np.mean(profs, 0), times[len(times) // 2]


def displacement(s1, s2, R, cfg: Config, win_m=4.0, maxlag=4):
    """Δr por janela de profundidade (centro, Δr, coerência), em duas etapas.

    Grosseira: correlação das AMPLITUDES com interpolação parabólica do pico (resolve
    deslocamentos maiores que λc/4, sem ambiguidade de fase). Fina: fase da correlação complexa
    na defasagem inteira mais próxima; entre os ramos Δr_fino + k·λc/2, escolhe-se o mais próximo
    do grosseiro. Escolher a defasagem pela coerência complexa máxima é ambíguo (fase periódica
    em λc/2 = 0,28 m e passo de 0,21 m): numa 1ª versão isso deu −554 m/a.
    """
    nb = int(round(win_m / cfg.dR))
    half = cfg.lam_c / 2
    centres, dr, coh = [], [], []
    for a in range(maxlag + 1, len(R) - nb - maxlag - 1, nb):
        x = s1[a:a + nb]
        ax = abs(x) - abs(x).mean()
        cc = []
        for lag in range(-maxlag, maxlag + 1):
            ay = abs(s2[a + lag:a + lag + nb])
            ay = ay - ay.mean()
            cc.append(np.sum(ax * ay) / np.sqrt(np.sum(ax ** 2) * np.sum(ay ** 2) + 1e-30))
        cc = np.array(cc)
        i = int(np.argmax(cc))
        frac = 0.0
        if 0 < i < len(cc) - 1:
            den = cc[i - 1] - 2 * cc[i] + cc[i + 1]
            frac = 0.5 * (cc[i - 1] - cc[i + 1]) / den if den != 0 else 0.0
        lag_c = i - maxlag
        coarse = (lag_c + frac) * cfg.dR
        y = s2[a + lag_c:a + lag_c + nb]
        c = np.sum(x * np.conj(y)) / np.sqrt(np.sum(abs(x) ** 2) * np.sum(abs(y) ** 2))
        # sinal: angle(s1·conj(s2)) é NEGATIVO quando o refletor se afasta (teste sintético:
        # +5 cm → −1,121 rad); por isso Δr = −φ·λc/(4π). Uma versão com +φ inverteu m_b.
        fine = lag_c * cfg.dR - np.angle(c) * cfg.lam_c / (4 * np.pi)
        k = np.round((coarse - fine) / half)
        centres.append(R[a + nb // 2])
        dr.append(fine + k * half)
        coh.append(abs(c))
    return np.array(centres), np.array(dr), np.array(coh)


def find_bed(R, s, zmin=100.0, zmax=900.0):
    amp = 20 * np.log10(abs(s) + 1e-30)
    sel = (R >= zmin) & (R <= zmax)
    return R[sel][np.argmax(amp[sel])]


def internal_phase_displacement(s1, s2, R, cfg, zmin, zmax, win_m=4.0, coh_min=0.95):
    """Camadas internas: fase com defasagem zero, desenrolada ao longo da profundidade.

    Em 5 dias o deslocamento por deformação é ≪ λc/4 (0,14 m), então não há ambiguidade de ramo
    entre janelas vizinhas; a correlação de amplitude nas camadas fracas é ruidosa demais
    (resíduo de 0,5 a 13 m numa 1ª versão).
    """
    nb = int(round(win_m / cfg.dR))
    z, ph, coh = [], [], []
    for a in range(0, len(R) - nb, nb):
        zc = R[a + nb // 2]
        if zc < zmin or zc > zmax:
            continue
        x, y = s1[a:a + nb], s2[a:a + nb]
        c = np.sum(x * np.conj(y)) / np.sqrt(np.sum(abs(x) ** 2) * np.sum(abs(y) ** 2))
        z.append(zc)
        ph.append(-np.angle(c))                       # mesma convenção de sinal
        coh.append(abs(c))
    z, ph, coh = map(np.array, (z, ph, coh))
    ok = coh > coh_min
    if ok.sum() < 5:
        return z[ok], np.full(ok.sum(), np.nan)
    phu = np.unwrap(ph[ok])
    return z[ok], phu * cfg.lam_c / (4 * np.pi)


def robust_line(z, d, nsig=3.0, it=5):
    keep = np.isfinite(d)
    for _ in range(it):
        if keep.sum() < 5:
            return np.nan, np.nan, keep
        a_b = np.polyfit(z[keep], d[keep], 1)
        r = d - np.polyval(a_b, z)
        s = 1.4826 * np.median(abs(r[keep] - np.median(r[keep])))
        new = np.isfinite(d) & (abs(r) < nsig * max(s, 1e-4))
        if (new == keep).all():
            break
        keep = new
    b, a = a_b
    return a, b, keep


def melt_between(R, s1, s2, cfg, z_bed, zmin=40.0, zfrac=0.8, win_m=4.0):
    """Derretimento no intervalo (m), deformação (Δr por m de profundidade) e nº de camadas.

    Ancoragem: o deslocamento interno desenrolado tem uma constante 2π arbitrária; a do fundo vem
    do método de duas etapas (absoluto). Para compará-los, a reta das camadas é ancorada pelo
    deslocamento absoluto (duas etapas) mediano das camadas mais coerentes.
    """
    zi, di = internal_phase_displacement(s1, s2, R, cfg, zmin, zfrac * z_bed, win_m)
    a, b, keep = robust_line(zi, di)
    if not np.isfinite(b):
        return np.nan, np.nan, 0
    z, dr, coh = displacement(s1, s2, R, cfg, win_m=win_m)
    sel = (z > zmin) & (z < zfrac * z_bed) & (coh > 0.97)
    if sel.sum() >= 3:
        # constante de ancoragem: múltiplo de λc/2 que aproxima a reta do absoluto
        off = np.median(dr[sel] - (a + b * z[sel]))
        a = a + np.round(off / (cfg.lam_c / 2)) * (cfg.lam_c / 2)
    k = np.argmin(abs(z - z_bed))
    m = (a + b * z_bed) - dr[k]
    return m, b, int(keep.sum())
