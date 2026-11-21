"""Orquestrador da cadeia de derretimento basal da ASE (ICESat-2 ATL06 r007, 2019–2025).

Roda as etapas em ordem, cada uma como processo separado (memória liberada entre elas).
Etapas com cache caro (download, maré CATS, advecção) pulam o que já existe por conta própria.

    python pipelines/run_basal_ase_all.py --list
    python pipelines/run_basal_ase_all.py --from espessura
    python pipelines/run_basal_ase_all.py --only validacao_apres

O download (etapa ``download``) e as leituras remotas (``velocidade_measures``, ``radar``) exigem
login Earthdata em ``~/.netrc``. Tempo total sem cache: da ordem de um dia, dominado pelo
download (~9 h) e pelas advecções Lagrangianas.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = "pipelines/"

STEPS: list[tuple[str, list[str], str]] = [
    ("download", [P + "fetch_atl06_son_mam_ase.py", "--seasons", "son", "mam", "jja", "djf"],
     "ATL06 r007 das 4 estações na ROI, um grânulo por vez"),
    ("mascara_plataformas", [P + "build_shelf_mask_ase.py"], "plataformas (NSIDC-0794 ∩ BedMachine)"),
    ("cobertura", [P + "run_coverage_audit_ase.py"], "auditoria de cobertura"),
    ("mare_faixa", [P + "run_tide_response_ase.py"], "faixa ±15 km, maré CATS e β euleriano"),
    ("mare_lagrangiana", [P + "run_tide_response_lagrangian.py", "Cosgrove", "Thwaites", "Crosson",
                          "Dotson", "Pine_Island"], "β com topografia Lagrangiana"),
    ("mascara_hidrostatica", [P + "build_hydrostatic_mask.py"], "d* por plataforma e variantes"),
    ("espessura", [P + "run_thickness_ase.py", "--stage", "ABCD"], "H e série por efeitos fixos"),
    ("espessura_variantes_f080", [P + "run_thickness_ase.py", "--stage", "D", "--variant",
                                  "classe_f080", "--no-jack"], "série na variante f080"),
    ("espessura_variantes_f095", [P + "run_thickness_ase.py", "--stage", "D", "--variant",
                                  "classe_f095", "--no-jack"], "série na variante f095"),
    ("espessura_variantes_ic", [P + "run_thickness_ase.py", "--stage", "D", "--variant",
                                "classe_ic", "--no-jack"], "série na variante IC"),
    ("divergencia", [P + "run_divergence_ase.py", "--save", "1", "2", "3", "5"],
     "∇·u e escala por validação cruzada"),
    ("termo_fluxo", [P + "run_flux_term_ase.py"], "H·∇·u"),
    ("derretimento", [P + "run_basal_melt_ase.py"], "m_b por célula e série trimestral"),
    ("mapa_continuo", [P + "run_mb_continuous_ase.py"], "krigagem escolhida por CV"),
    ("radar", [P + "fetch_validation_radar.py", "mcords"], "IceBridge MCoRDS L2"),
    ("apres_download", [P + "fetch_validation_radar.py", "apres"], "ApRES ITGC MELT"),
    ("teste_radar", [P + "run_mcords_hydro_test.py"], "erro hidrostático × distância"),
    ("frente", [P + "run_front_zone_ase.py"], "faixa da frente e perda de gelo"),
    ("velocidade_measures", [P + "run_velocity_crosscheck_ase.py"], "MEaSUREs × ITS_LIVE"),
    ("verificacoes", [P + "run_open_checks_ase.py"], "folga do leito, recuo, recongelamento"),
    ("dominio_final", [P + "run_final_domain_ase.py"], "domínio final"),
    ("envelope", [P + "run_uncertainty_mb_ase.py"], "envelope (agregação) e σ por pixel"),
    ("componentes_uniao", [P + "run_union_components_ase.py"], "componentes no domínio união"),
    ("fatorial", [P + "run_factorial_v2_ase.py"], "fatorial de 1.728 combinações"),
    ("resumo", [P + "run_final_summary_ase.py"], "tabela final"),
    ("thwaites_correcao", [P + "run_gz_recovery_ase.py", "Thwaites"], "correção β·maré + radar"),
    ("thwaites_recuperada", [P + "run_thwaites_recovered.py"], "faixa recuperada na Thwaites"),
    ("massa", [P + "run_mass_flux_ase.py"], "Gt/ano (versão por domínio)"),
    ("validacao_apres", [P + "run_apres_validation.py"], "ApRES G1–G4 × produto"),
    ("coerencia_enso", [P + "run_coherence_test_ase.py"], "teste da hipótese de Meng"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--from", dest="start")
    ap.add_argument("--only", nargs="+")
    a = ap.parse_args()
    names = [s[0] for s in STEPS]
    if a.list:
        for n, _, d in STEPS:
            print(f"{n:28s} {d}")
        return 0
    todo = STEPS
    if a.start:
        todo = STEPS[names.index(a.start):]
    if a.only:
        todo = [s for s in STEPS if s[0] in a.only]
    for n, cmd, d in todo:
        t0 = time.time()
        print(f"=== {n}: {d}", flush=True)
        r = subprocess.run([sys.executable, *cmd], cwd=ROOT)
        if r.returncode != 0:
            print(f"*** {n} falhou (código {r.returncode}); parando.", flush=True)
            return r.returncode
        print(f"=== {n} ok ({time.time() - t0:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
