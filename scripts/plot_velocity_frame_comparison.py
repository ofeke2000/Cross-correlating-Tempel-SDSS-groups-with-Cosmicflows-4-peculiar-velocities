"""
plot_velocity_frame_comparison.py
------------------------------------
Runnable script: compare the observer-frame velocity-centered zeta_1 dipole
(`dvcorr.estimators.shell_dipole.velocity_centered_shell_dipole`) against the
observer-free velocity-frame dipole
(`dvcorr.estimators.velocity_frame_dipole.velocity_frame_shell_dipole`) on
the SAME shared set of MDPL2 halo centers, and plot it: the comparison
dipole figure and its separate monopole companion (both frames overlaid on
each, CLAUDE.md hard rule 6 -- `make_comparison_figure` returns a
`DipoleFigures`, two PNGs, never one merged image) plus the
axis-rotation-angle diagnostic figure.

Thin driver only: the actual stage functions (`select_shared_centers`,
`run_both_frames`, `normalize_comparison`, `make_comparison_figure`,
`make_angle_diagnostic_figure`) live in
`dvcorr.pipeline.velocity_frame_comparison`, the single source of truth;
`load_and_carve`, `draw_candidates`, and `box_number_density` are reused
unchanged from `dvcorr.pipeline.velocity_centered` -- per CLAUDE.md's "one
library, two thin consumers" model, this script reimplements none of the
pipeline. The MDPL2 catalog load is long (~4M rows); this script is not run
as part of task verification -- only imported / byte-compiled to check it is
well-formed.

Matplotlib backend
------------------
`matplotlib.use("Agg")` is called HERE, at module import time, BEFORE
`dvcorr.pipeline.velocity_frame_comparison` (which imports
`dvcorr.pipeline.velocity_centered`, which imports `matplotlib.pyplot` at
module level for `make_figure`) is imported. This is the only place in this
script that selects a backend -- a notebook that imports the pipeline
modules directly never goes through this script, so it keeps whatever
backend (e.g. an inline one) it already has.

Usage
-----
    .venv/bin/python -m scripts.plot_velocity_frame_comparison
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
    dipole_and_monopole_paths,
    draw_candidates,
    box_number_density,
    load_and_carve,
)
from dvcorr.pipeline.velocity_frame_comparison import (
    ComparisonRunConfig,
    make_angle_diagnostic_figure,
    make_comparison_figure,
    normalize_comparison,
    run_both_frames,
    select_shared_centers,
)


def main() -> None:
    """Chain the pipeline stages and write both PNGs."""
    args = add_catalog_arguments(argparse.ArgumentParser(description=__doc__)).parse_args()
    cfg = ComparisonRunConfig(catalog=catalog_from_args(args))
    paths = PathsConfig()
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)

    carved = load_and_carve(cfg, paths)
    candidates = draw_candidates(cfg, carved)
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

    results = run_both_frames(cfg, centers, carved.pos, observer)

    n_bar = box_number_density(carved.n_total)
    comparison = normalize_comparison(cfg, results, n_bar)

    comparison_figs = make_comparison_figure(cfg, results, comparison)
    angle_fig = make_angle_diagnostic_figure(cfg, comparison)

    paths.ensure_output_dir()
    dipole_path, monopole_path = dipole_and_monopole_paths(
        paths.output_dir, cfg.comparison_output_name
    )
    angle_path = paths.output_dir / cfg.angle_diagnostic_output_name
    comparison_figs.dipole.savefig(dipole_path, dpi=cfg.dpi, bbox_inches="tight")
    comparison_figs.monopole.savefig(monopole_path, dpi=cfg.dpi, bbox_inches="tight")
    angle_fig.savefig(angle_path, dpi=cfg.dpi, bbox_inches="tight")
    print(f"wrote {dipole_path}")
    print(f"wrote {monopole_path}")
    print(f"wrote {angle_path}")


if __name__ == "__main__":
    main()
