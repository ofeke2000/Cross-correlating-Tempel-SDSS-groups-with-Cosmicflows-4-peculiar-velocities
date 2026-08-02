"""
plot_velocity_centered_dipole.py
----------------------------------
Runnable script: measure the velocity-centered zeta_1 dipole
(`dvcorr.estimators.shell_dipole.velocity_centered_shell_dipole`) on real
MDPL2 halos and plot it, alongside its monopole companion (CLAUDE.md hard
rule 6) and a velocity-shuffle null.

Thin driver only: the actual stage functions (`load_and_carve`,
`draw_candidates`, `run_estimator`, `global_number_density`,
`normalize_result`, `make_figure`) live in
`dvcorr.pipeline.velocity_centered`, the single source of truth also
consumed by notebooks/05_velocity_centered_dipole.ipynb -- per CLAUDE.md's
"notebooks are exploration only ... never reimplement" convention, neither
this script nor that notebook reimplements any of this pipeline. The MDPL2
catalog load is long; this script is not run as part of task verification --
only imported / byte-compiled to check it is well-formed. Which catalog it
reads comes from `RunConfig().catalog` (`dvcorr.config.catalog.CatalogConfig`).

Matplotlib backend
------------------
`matplotlib.use("Agg")` is called HERE, at module import time, BEFORE
`dvcorr.pipeline.velocity_centered` (which imports `matplotlib.pyplot` at
module level for `make_figure`) is imported. This is the only place in the
project that selects a backend -- a notebook that imports the pipeline
module directly never goes through this script, so it keeps whatever backend
(e.g. an inline one) it already has.

Usage
-----
    .venv/bin/python -m scripts.plot_velocity_centered_dipole
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")  # headless: write straight to file, no display

import numpy as np

from dvcorr import conventions
from dvcorr.config import PathsConfig, add_catalog_arguments, catalog_from_args
from dvcorr.pipeline.mass_diagnostics import mass_funnel, print_mass_funnel
from dvcorr.pipeline.velocity_centered import (
    RunConfig,
    draw_candidates,
    global_number_density,
    load_and_carve,
    make_figure,
    normalize_result,
    run_estimator,
    select_shared_centers,
)


def main() -> None:
    """Chain the pipeline stages and write the PNG."""
    args = add_catalog_arguments(argparse.ArgumentParser(description=__doc__)).parse_args()
    cfg = RunConfig(catalog=catalog_from_args(args))
    paths = PathsConfig()
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)

    carved = load_and_carve(cfg, paths)
    candidates = draw_candidates(cfg, carved)
    result = run_estimator(cfg, candidates.s, candidates.v, carved.pos, observer)

    # Same two cuts the estimator applies internally (idempotent), re-run here
    # only to recover WHICH candidates survived, so the funnel can report the
    # centers by mass. The estimator returns counts, not identities.
    centers = select_shared_centers(
        cfg,
        candidates.s,
        candidates.v,
        observer,
        mvir_candidates=candidates.mvir,
        is_distinct_candidates=candidates.is_distinct,
    )
    print_mass_funnel(
        mass_funnel(
            carved.catalog_mvir,
            carved.mvir,
            candidates.mvir,
            centers.mvir_centers,
            centers.is_distinct_centers,
            cfg.catalog.name,
            cfg.catalog.describe_cuts(),
        )
    )

    n_bar = global_number_density(carved.n_carved, cfg.sub_volume_radius)
    normalized = normalize_result(result, n_bar, cfg.shuffle_seed)

    fig = make_figure(cfg, result, normalized)

    paths.ensure_output_dir()
    out_path = paths.output_dir / cfg.output_name
    fig.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
