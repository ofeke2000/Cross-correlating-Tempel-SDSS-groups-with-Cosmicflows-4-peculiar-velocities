"""
plot_redshift_space_comparison.py
------------------------------------
Runnable script: compare the real-space velocity-centered zeta_1 dipole
against the redshift-space run of the SAME estimator
(`dvcorr.estimators.shell_dipole.velocity_centered_shell_dipole`, unchanged --
only the tracer/center positions differ, via
`dvcorr.redshift_space.to_redshift_space`) on a SHARED set of MDPL2 halo
centers, and plot both: the center-averaged comparison (dipole and its
separate monopole companion, CLAUDE.md hard rule 6 --
`make_redshift_comparison_figure` returns a `DipoleFigures`, two PNGs) and
the single-example-center figure (likewise a `DipoleFigures`).

Thin driver only: the actual stage functions (`load_and_carve_buffered`,
`build_tracer_spaces`, `select_redshift_shared_centers`, `run_both_spaces`,
`normalize_redshift_comparison`, `make_redshift_comparison_figure`,
`make_single_center_figure`) live in
`dvcorr.pipeline.redshift_space_comparison`, the single source of truth;
`draw_candidates` and `box_number_density` are reused unchanged from
`dvcorr.pipeline.velocity_centered` -- per CLAUDE.md's "one library, two thin
consumers" model, this script reimplements none of the pipeline. The MDPL2
catalog load is long (~4M rows); this script is not run as part of task
verification -- only imported / byte-compiled to check it is well-formed
(mirrors `scripts/plot_velocity_centered_dipole.py:15-17`).

Matplotlib backend
------------------
`matplotlib.use("Agg")` is called HERE, at module import time, BEFORE
`dvcorr.pipeline.redshift_space_comparison` (which imports
`dvcorr.pipeline.velocity_centered`, which imports `matplotlib.pyplot` at
module level for `make_figure`) is imported. This is the only place in this
script that selects a backend -- a notebook that imports the pipeline
modules directly never goes through this script, so it keeps whatever
backend (e.g. an inline one) it already has.

Usage
-----
    .venv/bin/python -m scripts.plot_redshift_space_comparison
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
    box_number_density,
    dipole_and_monopole_paths,
    draw_candidates_from_arrays,
)
from dvcorr.pipeline.redshift_space_comparison import (
    RedshiftSpaceRunConfig,
    build_tracer_spaces,
    load_and_carve_buffered,
    make_redshift_comparison_figure,
    make_single_center_figure,
    normalize_redshift_comparison,
    run_both_spaces,
    select_redshift_shared_centers,
)


def main() -> None:
    """Chain the pipeline stages and write both PNGs."""
    args = add_catalog_arguments(argparse.ArgumentParser(description=__doc__)).parse_args()
    cfg = RedshiftSpaceRunConfig(catalog=catalog_from_args(args))
    paths = PathsConfig()
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)

    buffer = load_and_carve_buffered(cfg, paths)
    tracers = build_tracer_spaces(cfg, buffer, observer)

    candidates = draw_candidates_from_arrays(
        cfg,
        buffer.pos_core,
        buffer.vel_core,
        buffer.mvir_core,
        buffer.is_distinct_core,
    )

    centers = select_redshift_shared_centers(
        cfg,
        candidates.s,
        candidates.v,
        observer,
        buffer.v_margin_kms,
        buffer.v_margin_mpc,
        mvir_candidates=candidates.mvir,
        is_distinct_candidates=candidates.is_distinct,
    )
    print_mass_funnel(
        mass_funnel(
            buffer.catalog_mvir,
            buffer.mvir_core,
            candidates.mvir,
            centers.mvir_centers,
            centers.is_distinct_centers,
            cfg.catalog.name,
            cfg.catalog.describe_cuts(),
        )
    )

    results = run_both_spaces(cfg, centers, tracers, observer)

    n_bar = box_number_density(buffer.n_total)
    comparison = normalize_redshift_comparison(cfg, results, n_bar)

    comparison_figs = make_redshift_comparison_figure(cfg, results, comparison)
    single_center_figs = make_single_center_figure(cfg, results, comparison.n_bar)

    paths.ensure_output_dir()
    comparison_dipole_path, comparison_monopole_path = dipole_and_monopole_paths(
        paths.output_dir, cfg.comparison_output_name
    )
    single_center_dipole_path, single_center_monopole_path = dipole_and_monopole_paths(
        paths.output_dir, cfg.single_center_output_name
    )
    comparison_figs.dipole.savefig(comparison_dipole_path, dpi=cfg.dpi, bbox_inches="tight")
    comparison_figs.monopole.savefig(comparison_monopole_path, dpi=cfg.dpi, bbox_inches="tight")
    single_center_figs.dipole.savefig(single_center_dipole_path, dpi=cfg.dpi, bbox_inches="tight")
    single_center_figs.monopole.savefig(
        single_center_monopole_path, dpi=cfg.dpi, bbox_inches="tight"
    )
    print(f"wrote {comparison_dipole_path}")
    print(f"wrote {comparison_monopole_path}")
    print(f"wrote {single_center_dipole_path}")
    print(f"wrote {single_center_monopole_path}")


if __name__ == "__main__":
    main()
