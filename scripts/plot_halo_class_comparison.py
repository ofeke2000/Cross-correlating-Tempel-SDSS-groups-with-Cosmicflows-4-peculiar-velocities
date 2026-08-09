"""
plot_halo_class_comparison.py
--------------------------------
Runnable script: run BOTH existing comparisons -- observer frame vs. velocity
frame, and real space vs. redshift space -- twice over, once with the centers
restricted to SUBHALOS and once with them restricted to PARENT halos, and write
every figure.

Seven PNGs: the two per-class figures each comparison already knows how to draw
(`make_comparison_figure`, `make_redshift_comparison_figure`, called once per
class, nulls and all), the two class-CONTRAST figures that are new here
(`make_frame_class_contrast_figure`, `make_redshift_class_contrast_figure`),
and the sub-h^-1-Mpc occupancy histogram (`make_class_occupancy_figure`) that
says how much collapsed structure each class's centers sit in. Per-class
filenames are prefixed with the class so the two runs never overwrite each
other.

Thin driver only: every stage function lives in
`dvcorr.pipeline.halo_class_comparison`, the single source of truth, which in
turn calls `dvcorr.pipeline.velocity_frame_comparison` and
`dvcorr.pipeline.redshift_space_comparison` unchanged. This script owns no
pipeline logic -- it resolves a config, chains stages, and saves.

ONE CATALOG LOAD for all four runs: `load_and_carve_buffered` reads the catalog
once, `carved_from_buffer` re-presents its core arrays as the plain carve the
frame comparison wants, and `build_tracer_spaces` builds the redshift-space
tracer field once for both classes. Requires the `num_of_subhalos` column, so
the catalog must have been converted by a version of
`dvcorr.pipeline.catalog_conversion` that writes it -- if it was not, the load
says so and names the command that fixes it.

The full MDPL2 catalog load is long and four estimator-heavy runs follow it;
this script is not run as part of task verification -- only imported /
byte-compiled to check it is well-formed (mirrors
`scripts/plot_redshift_space_comparison.py`).

Matplotlib backend
------------------
`matplotlib.use("Agg")` is called HERE, at module import time, BEFORE any
`dvcorr.pipeline` module is imported. This is the only place in this script
that selects a backend -- a notebook importing the pipeline modules directly
never goes through this script and keeps its own inline backend.

Usage
-----
    .venv/bin/python -m scripts.plot_halo_class_comparison
    .venv/bin/python -m scripts.plot_halo_class_comparison --mass-min 1e12
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")  # headless: write straight to file, no display

import numpy as np

from dvcorr import conventions
from dvcorr.config import PathsConfig, add_catalog_arguments, catalog_from_args
from dvcorr.pipeline.halo_class_comparison import (
    HaloClassRunConfig,
    carved_from_buffer,
    class_output_name,
    class_shell_occupancy_for_runs,
    make_class_occupancy_figure,
    make_frame_class_contrast_figure,
    make_redshift_class_contrast_figure,
    require_classifiable_catalog,
    run_frame_comparison_for_class,
    run_redshift_comparison_for_class,
)
from dvcorr.pipeline.redshift_space_comparison import (
    build_tracer_spaces,
    load_and_carve_buffered,
    make_redshift_comparison_figure,
)
from dvcorr.pipeline.velocity_centered import box_number_density
from dvcorr.pipeline.velocity_frame_comparison import make_comparison_figure


def main() -> None:
    """Run both comparisons for both center classes and write all seven PNGs."""
    args = add_catalog_arguments(argparse.ArgumentParser(description=__doc__)).parse_args()
    cfg = HaloClassRunConfig(catalog=catalog_from_args(args))
    paths = PathsConfig()
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)

    # Fail before the multi-minute catalog load, not after it: a catalog with
    # no subhalos in it cannot answer either half of this comparison.
    require_classifiable_catalog(cfg)

    buffer = load_and_carve_buffered(cfg, paths, with_num_of_subhalos=True)
    carved = carved_from_buffer(buffer)
    tracers = build_tracer_spaces(cfg, buffer, observer)
    n_bar = box_number_density(buffer.n_total)

    frame_runs = {
        center_class: run_frame_comparison_for_class(
            cfg, center_class, carved, observer, n_bar
        )
        for center_class in cfg.center_classes
    }
    redshift_runs = {
        center_class: run_redshift_comparison_for_class(
            cfg, center_class, buffer, tracers, observer, n_bar
        )
        for center_class in cfg.center_classes
    }

    paths.ensure_output_dir()

    def _save(fig, name: str) -> None:
        path = paths.output_dir / name
        fig.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
        print(f"wrote {path}")

    for center_class, run in frame_runs.items():
        _save(
            make_comparison_figure(cfg, run.results, run.comparison),
            class_output_name(center_class, cfg.frame_output_name),
        )
    for center_class, run in redshift_runs.items():
        _save(
            make_redshift_comparison_figure(cfg, run.results, run.comparison),
            class_output_name(center_class, cfg.redshift_output_name),
        )

    _save(make_frame_class_contrast_figure(cfg, frame_runs), cfg.frame_contrast_output_name)
    _save(
        make_redshift_class_contrast_figure(cfg, redshift_runs),
        cfg.redshift_contrast_output_name,
    )

    # Counted on the FRAME runs' centers (the redshift runs' are a subset, cut
    # by the wider r_max + v_margin margin), against the same full-carve
    # tracer field the estimators used.
    _save(
        make_class_occupancy_figure(
            cfg, class_shell_occupancy_for_runs(cfg, frame_runs, carved.pos, n_bar)
        ),
        cfg.occupancy_output_name,
    )


if __name__ == "__main__":
    main()
