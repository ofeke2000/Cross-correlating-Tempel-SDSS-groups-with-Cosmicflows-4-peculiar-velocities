"""
test_mass_diagnostics.py
------------------------
Unit tests for `dvcorr.pipeline.mass_diagnostics`: the mass funnel's
construction guards, the exactness of the particle-count conversion, and that
the figure builds from a synthetic funnel.

Synthetic arrays only -- never the real catalogs (see
`tests/test_catalog_equivalence.py` for the tests that do need them).
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from dvcorr import conventions
from dvcorr.pipeline.mass_diagnostics import (
    MassFunnel,
    make_mass_histogram_figure,
    mass_funnel,
    particle_counts,
    print_mass_funnel,
)


def _synthetic_funnel(seed: int = 0) -> MassFunnel:
    """A nested four-stage funnel with a plausible mass spread."""
    rng = np.random.default_rng(seed)
    catalog = conventions.PARTICLE_MASS * rng.integers(2, 100_000, size=5000).astype(float)
    carved = catalog[rng.choice(catalog.size, size=1200, replace=False)]
    candidates = carved[rng.choice(carved.size, size=300, replace=False)]
    centers = candidates[rng.choice(candidates.size, size=140, replace=False)]
    return mass_funnel(
        catalog,
        carved,
        candidates,
        centers,
        rng.random(centers.size) > 0.12,
        "full",
        "full, no mass cut, subhalos included",
    )


class TestParticleCounts:
    def test_conversion_is_exact_on_quantized_masses(self) -> None:
        """mvir is an integer multiple of PARTICLE_MASS, so the count comes back
        as that exact integer -- the property the figure's top axis rests on."""
        counts = np.array([2, 20, 100, 664, 2_300_000], dtype=float)
        recovered = particle_counts(counts * conventions.PARTICLE_MASS)
        np.testing.assert_allclose(recovered, counts, rtol=1e-12)

    def test_shape_is_preserved_for_one_element(self) -> None:
        # The project's shape contract: an (N,) function returns (N,) at N = 1.
        assert particle_counts(np.array([conventions.PARTICLE_MASS])).shape == (1,)


class TestMassFunnelConstruction:
    def test_stages_are_in_funnel_order_and_nested(self) -> None:
        funnel = _synthetic_funnel()
        sizes = [stage.size for stage in funnel.stages]
        assert sizes == sorted(sizes, reverse=True)
        assert len(funnel.stages) == 4

    def test_missing_centers_raise(self) -> None:
        """A funnel without its final stage is not a funnel: the centers are the
        population the dipole is measured from, and the diagnostic exists to
        describe them."""
        with pytest.raises(ValueError):
            mass_funnel(
                np.ones(10), np.ones(5), np.ones(3), None, None, "full", "cuts"
            )

    def test_mismatched_center_labels_raise(self) -> None:
        """A length mismatch means the two arrays survived different cuts, which
        would mislabel every subhalo in the figure rather than fail loudly."""
        with pytest.raises(ValueError):
            mass_funnel(
                np.ones(10),
                np.ones(5),
                np.ones(3),
                np.ones(3),
                np.ones(2, dtype=bool),
                "full",
                "cuts",
            )


class TestReporting:
    def test_print_mass_funnel_reports_every_stage(self, capsys) -> None:
        print_mass_funnel(_synthetic_funnel())
        out = capsys.readouterr().out
        for label in ("catalog", "carved", "candidates", "centers"):
            assert label in out
        # The resolution columns are the point of the table.
        assert f"< {conventions.RESOLVED_PARTICLE_COUNT}p" in out
        assert f"< {conventions.CONVERGED_PARTICLE_COUNT}p" in out

    def test_figure_builds_with_both_panels(self) -> None:
        fig = make_mass_histogram_figure(_synthetic_funnel())
        assert len(fig.axes) >= 2
        # Cartesian x-axis even though the BINS are logarithmic: the axis shows
        # log10(mvir) linearly, per this project's plotting convention.
        assert fig.axes[0].get_xscale() == "linear"
        assert fig.axes[0].get_yscale() == "log"
        matplotlib.pyplot.close(fig)

    def test_figure_survives_centers_that_are_all_distinct(self) -> None:
        """The subhalo curve is simply absent, rather than the figure failing --
        the mvir12 catalog contains no subhalos at all, so this is the ordinary
        case there, not an edge case."""
        rng = np.random.default_rng(3)
        catalog = conventions.PARTICLE_MASS * rng.integers(2, 5000, size=800).astype(float)
        centers = catalog[:50]
        funnel = mass_funnel(
            catalog, catalog[:400], catalog[:100], centers,
            np.ones(centers.size, dtype=bool), "mvir12", "mvir12, distinct halos only",
        )
        fig = make_mass_histogram_figure(funnel)
        assert len(fig.axes) >= 2
        matplotlib.pyplot.close(fig)
