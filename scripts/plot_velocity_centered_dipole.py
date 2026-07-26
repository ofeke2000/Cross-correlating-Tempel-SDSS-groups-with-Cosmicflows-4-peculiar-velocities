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
catalog load is long (~4M rows); this script is not run as part of task
verification -- only imported / byte-compiled to check it is well-formed.

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

import matplotlib

matplotlib.use("Agg")  # headless: write straight to file, no display

import numpy as np

from dvcorr import conventions
from dvcorr.config import PathsConfig
from dvcorr.pipeline.velocity_centered import (
    RunConfig,
    draw_candidates,
    global_number_density,
    load_and_carve,
    make_figure,
    normalize_result,
    run_estimator,
)


def main() -> None:
    """Chain the pipeline stages and write the PNG."""
    cfg = RunConfig()
    paths = PathsConfig()
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)

    pos, vel = load_and_carve(cfg, paths)
    s_candidates, v_candidates = draw_candidates(cfg, pos, vel)
    result = run_estimator(cfg, s_candidates, v_candidates, pos, observer)

    n_bar = global_number_density(pos.shape[0], cfg.sub_volume_radius)
    normalized = normalize_result(result, n_bar, cfg.shuffle_seed)

    fig = make_figure(cfg, result, normalized)

    paths.ensure_output_dir()
    out_path = paths.output_dir / cfg.output_name
    fig.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
