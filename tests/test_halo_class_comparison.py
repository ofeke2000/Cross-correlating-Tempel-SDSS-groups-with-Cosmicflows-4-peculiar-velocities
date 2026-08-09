"""
test_halo_class_comparison.py
-----------------------------
Unit and wiring tests for `dvcorr.pipeline.halo_class_comparison`, on synthetic
arrays only -- never the real catalog.

Three things are worth pinning here, in descending order of how expensive a
silent regression would be:

1. THE TWO CLASSES ARE DISJOINT, and a sub-subhalo lands in the subhalo class
   rather than in both. `center_class_mask`'s `is_distinct` conjunct is the
   only thing enforcing that, and dropping it produces no error -- just a
   quietly overlapping comparison whose "difference between classes" is partly
   a comparison of shared halos with themselves.
2. THE SENTINEL IS NEVER READ AS A COUNT. A catalog that cannot say which
   halos host subhalos (`NUM_OF_SUBHALOS_UNKNOWN`) must raise, not classify
   every halo as "not a parent" -- those are different statements and only one
   of them is true.
3. CLASS FILTERING PRECEDES THE CANDIDATE DRAW, and the tracer field is NOT
   filtered. Both are the difference between comparing two center populations
   against one density field and accidentally comparing two different density
   fields.

Backend discipline: this module calls `matplotlib.use("Agg")` itself, before
importing any `dvcorr.pipeline` module -- the same division of responsibility
`test_plot_wiring.py` follows and every `scripts/plot_*.py` driver follows: the
CONSUMER selects the backend, the library never does.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from dvcorr import conventions
from dvcorr.config import CATALOG_MVIR12, CatalogConfig, ShellConfig
from dvcorr.pipeline.catalog_conversion import NUM_OF_SUBHALOS_UNKNOWN
from dvcorr.pipeline.halo_class_comparison import (
    CENTER_CLASS_PARENT,
    CENTER_CLASS_SUBHALO,
    VALID_CENTER_CLASSES,
    ClassFrameRun,
    ClassRedshiftRun,
    HaloClassRunConfig,
    carved_from_buffer,
    center_class_mask,
    class_output_name,
    class_shell_occupancy,
    class_shell_occupancy_for_runs,
    make_class_occupancy_figure,
    make_frame_class_contrast_figure,
    make_redshift_class_contrast_figure,
    require_classifiable_catalog,
    run_frame_comparison_for_class,
    run_redshift_comparison_for_class,
    select_class_population,
)
from dvcorr.pipeline.redshift_space_comparison import BufferedCarve, build_tracer_spaces
from dvcorr.pipeline.velocity_centered import CarvedHalos, box_number_density

_OBSERVER = np.asarray(conventions.OBSERVER_POSITION, dtype=float)

#: Small enough to run four estimator passes per class in a test, large enough
#: that every class survives the core cut with centers to spare.
_N_HALOS = 4000
_SUB_VOLUME_RADIUS = 60.0
_N_CANDIDATES = 40


def _tiny_cfg(**overrides) -> HaloClassRunConfig:
    """A `HaloClassRunConfig` sized for a test, not for a run."""
    params = dict(
        sub_volume_radius=_SUB_VOLUME_RADIUS,
        shells=ShellConfig(min_radius=2.0, max_radius=10.0, n_bins=4),
        n_candidate_centers=_N_CANDIDATES,
    )
    params.update(overrides)
    return HaloClassRunConfig(**params)


def _synthetic_halos(seed: int = 0):
    """Halos uniform in a ball around the observer, with a mixed class makeup.

    Positions are drawn in the ball rather than in a cube so every halo is
    inside `_SUB_VOLUME_RADIUS` and the carve is a no-op -- this fixture is
    about class selection, not about the carve, which `load_and_carve` already
    owns and tests elsewhere.

    Class makeup, chosen so all four cases below are populated:
    distinct-with-daughters (parents), distinct-without (neither class),
    subhalos-without-daughters, and subhalos WITH daughters (sub-subhalo hosts,
    the case that must NOT be counted as a parent).
    """
    rng = np.random.default_rng(seed)

    direction = rng.normal(size=(_N_HALOS, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    radius = _SUB_VOLUME_RADIUS * rng.random(_N_HALOS) ** (1.0 / 3.0)
    pos = _OBSERVER + direction * radius[:, None]

    vel = rng.normal(scale=300.0, size=(_N_HALOS, 3))
    mvir = conventions.PARTICLE_MASS * rng.integers(2, 100_000, size=_N_HALOS).astype(float)

    is_distinct = rng.random(_N_HALOS) > 0.30
    num_of_subhalos = np.where(rng.random(_N_HALOS) > 0.55, rng.integers(1, 6, _N_HALOS), 0)
    return pos, vel, mvir, is_distinct, num_of_subhalos.astype(np.int32)


def _carved(seed: int = 0) -> CarvedHalos:
    pos, vel, mvir, is_distinct, num_of_subhalos = _synthetic_halos(seed)
    return CarvedHalos(
        pos=pos,
        vel=vel,
        mvir=mvir,
        is_distinct=is_distinct,
        n_carved=pos.shape[0],
        n_total=pos.shape[0],
        catalog_mvir=mvir,
        num_of_subhalos=num_of_subhalos,
    )


def _buffered(cfg: HaloClassRunConfig, seed: int = 0) -> BufferedCarve:
    """A `BufferedCarve` whose buffer and core are the same synthetic ball.

    `v_margin` is set to a small fixed value rather than derived, because this
    fixture exists to exercise class selection through the redshift-space
    center path, not to re-test `v_margin_from_statistic`.
    """
    pos, vel, mvir, is_distinct, num_of_subhalos = _synthetic_halos(seed)
    v_margin_kms = 300.0
    v_margin_mpc = v_margin_kms / 100.0
    return BufferedCarve(
        pos_buffer=pos,
        vel_buffer=vel,
        pos_core=pos,
        vel_core=vel,
        n_core=pos.shape[0],
        n_buffer=pos.shape[0],
        n_total=pos.shape[0],
        mvir_core=mvir,
        is_distinct_core=is_distinct,
        catalog_mvir=mvir,
        v_margin_kms=v_margin_kms,
        v_margin_mpc=v_margin_mpc,
        buffered_radius=cfg.sub_volume_radius + v_margin_mpc,
        num_of_subhalos_core=num_of_subhalos,
    )


class TestCenterClassMask:
    def test_the_two_classes_are_disjoint(self) -> None:
        """A halo is a satellite, a host, or neither -- never two of those.

        Overlapping classes would make the plotted "difference between
        classes" partly a comparison of shared halos with themselves, with no
        error anywhere to reveal it.
        """
        _, _, _, is_distinct, num_of_subhalos = _synthetic_halos()
        parent = center_class_mask(is_distinct, num_of_subhalos, CENTER_CLASS_PARENT)
        subhalo = center_class_mask(is_distinct, num_of_subhalos, CENTER_CLASS_SUBHALO)
        assert not np.any(parent & subhalo)
        assert parent.sum() > 0 and subhalo.sum() > 0

    def test_a_subhalo_hosting_a_sub_subhalo_is_not_a_parent(self) -> None:
        """The `is_distinct` conjunct, isolated.

        Rockstar's `pid` is the IMMEDIATE parent, so a subhalo can itself be
        named as a parent by a sub-subhalo. Such a halo has
        `num_of_subhalos >= 1` and must still land in the subhalo class alone.
        """
        is_distinct = np.array([False])       # it IS a subhalo
        num_of_subhalos = np.array([2], dtype=np.int32)  # and it hosts two of its own
        assert not center_class_mask(is_distinct, num_of_subhalos, CENTER_CLASS_PARENT)[0]
        assert center_class_mask(is_distinct, num_of_subhalos, CENTER_CLASS_SUBHALO)[0]

    def test_parent_requires_at_least_one_daughter(self) -> None:
        """An isolated distinct halo is in neither class."""
        is_distinct = np.array([True])
        num_of_subhalos = np.array([0], dtype=np.int32)
        assert not center_class_mask(is_distinct, num_of_subhalos, CENTER_CLASS_PARENT)[0]
        assert not center_class_mask(is_distinct, num_of_subhalos, CENTER_CLASS_SUBHALO)[0]

    def test_min_num_of_subhalos_raises_the_richness_floor(self) -> None:
        is_distinct = np.array([True, True, True])
        num_of_subhalos = np.array([1, 3, 5], dtype=np.int32)
        np.testing.assert_array_equal(
            center_class_mask(
                is_distinct, num_of_subhalos, CENTER_CLASS_PARENT, min_num_of_subhalos=3
            ),
            [False, True, True],
        )

    def test_subhalo_class_ignores_the_daughter_count(self) -> None:
        """The subhalo class is a statement about a halo's OWN pid only."""
        is_distinct = np.array([False, False])
        np.testing.assert_array_equal(
            center_class_mask(
                is_distinct, np.array([0, 7], dtype=np.int32), CENTER_CLASS_SUBHALO
            ),
            [True, True],
        )

    def test_sentinel_raises_rather_than_classifying_everything_as_not_parent(self) -> None:
        """`NUM_OF_SUBHALOS_UNKNOWN` means "cannot say", not "hosts none"."""
        is_distinct = np.array([True, True])
        num_of_subhalos = np.full(2, NUM_OF_SUBHALOS_UNKNOWN, dtype=np.int32)
        with pytest.raises(RuntimeError, match="NUM_OF_SUBHALOS_UNKNOWN"):
            center_class_mask(is_distinct, num_of_subhalos, CENTER_CLASS_PARENT)

    def test_missing_column_raises_with_the_flag_that_fixes_it(self) -> None:
        with pytest.raises(RuntimeError, match="with_num_of_subhalos"):
            center_class_mask(np.array([True]), None, CENTER_CLASS_PARENT)

    def test_misaligned_labels_raise(self) -> None:
        with pytest.raises(RuntimeError, match="row-aligned"):
            center_class_mask(
                np.array([True, False]), np.array([1], dtype=np.int32), CENTER_CLASS_PARENT
            )

    def test_unknown_class_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown center_class"):
            center_class_mask(np.array([True]), np.array([1]), "subhalos")


class TestRunConfig:
    def test_inherits_the_knobs_of_both_parent_pipelines(self) -> None:
        """The single-config claim, asserted rather than assumed.

        If the MRO ever stops resolving to one shared `RunConfig`, one of these
        attributes disappears and both comparisons stop being driven by the
        same shells and seeds.
        """
        cfg = HaloClassRunConfig()
        assert cfg.axis_null_seed == 44                  # ComparisonRunConfig
        assert cfg.velocity_gaussian_null_seed == 47     # ComparisonRunConfig
        assert cfg.v_margin_statistic == "max"           # RedshiftSpaceRunConfig
        assert cfg.redshift_shuffle_seed == 45           # RedshiftSpaceRunConfig
        assert cfg.seed == 42                            # RunConfig, shared
        assert cfg.shells.max_radius == 64.0             # RunConfig, shared

    def test_base_validation_still_runs(self) -> None:
        """`__post_init__` must chain through to `RunConfig`'s own checks."""
        with pytest.raises(ValueError, match="sub_volume_radius"):
            HaloClassRunConfig(sub_volume_radius=-1.0)
        with pytest.raises(ValueError, match="v_margin_statistic"):
            HaloClassRunConfig(v_margin_statistic="mean")

    @pytest.mark.parametrize(
        "overrides, match",
        [
            ({"center_classes": ()}, "must not be empty"),
            ({"center_classes": ("subhalos",)}, "unknown center class"),
            ({"center_classes": ("parent", "parent")}, "repeats a class"),
            ({"min_num_of_subhalos": 0}, "must be >= 1"),
        ],
    )
    def test_rejects_bad_class_configuration(self, overrides, match) -> None:
        with pytest.raises(ValueError, match=match):
            HaloClassRunConfig(**overrides)

    def test_per_class_output_names_are_distinct(self) -> None:
        """Two classes must not overwrite each other's PNG."""
        cfg = HaloClassRunConfig()
        names = {class_output_name(c, cfg.frame_output_name) for c in cfg.center_classes}
        assert len(names) == len(cfg.center_classes)

    def test_mvir12_is_refused_by_name(self) -> None:
        cfg = HaloClassRunConfig(catalog=CatalogConfig(name=CATALOG_MVIR12))
        with pytest.raises(RuntimeError, match=CATALOG_MVIR12):
            require_classifiable_catalog(cfg)

    def test_full_catalog_is_accepted(self) -> None:
        require_classifiable_catalog(HaloClassRunConfig())


class TestClassPopulation:
    def test_every_member_is_in_the_class_and_the_counts_add_up(self) -> None:
        cfg = _tiny_cfg()
        carved = _carved()
        parents = select_class_population(
            cfg, CENTER_CLASS_PARENT, carved.pos, carved.vel, carved.mvir,
            carved.is_distinct, carved.num_of_subhalos,
        )
        subhalos = select_class_population(
            cfg, CENTER_CLASS_SUBHALO, carved.pos, carved.vel, carved.mvir,
            carved.is_distinct, carved.num_of_subhalos,
        )
        assert parents.is_distinct.all()
        assert not subhalos.is_distinct.any()
        assert parents.n_source == subhalos.n_source == carved.n_carved
        assert parents.n_class + subhalos.n_class < carved.n_carved  # isolated halos in neither

    def test_rows_stay_aligned_across_every_array(self) -> None:
        """One mask over all four arrays, so `pos[i]` still describes `mvir[i]`."""
        cfg = _tiny_cfg()
        carved = _carved()
        population = select_class_population(
            cfg, CENTER_CLASS_PARENT, carved.pos, carved.vel, carved.mvir,
            carved.is_distinct, carved.num_of_subhalos,
        )
        expected = center_class_mask(
            carved.is_distinct, carved.num_of_subhalos, CENTER_CLASS_PARENT
        )
        np.testing.assert_array_equal(population.pos, carved.pos[expected])
        np.testing.assert_array_equal(population.vel, carved.vel[expected])
        np.testing.assert_array_equal(population.mvir, carved.mvir[expected])

    def test_an_empty_class_raises_here_not_downstream(self) -> None:
        cfg = _tiny_cfg()
        n = 10
        with pytest.raises(RuntimeError, match="no carved halo is in class"):
            select_class_population(
                cfg, CENTER_CLASS_SUBHALO,
                np.zeros((n, 3)), np.zeros((n, 3)), np.ones(n),
                np.ones(n, dtype=bool), np.ones(n, dtype=np.int32),
            )


class TestCarvedFromBuffer:
    def test_repackaging_changes_nothing(self) -> None:
        """`carved_from_buffer` is a view, not a second selection."""
        cfg = _tiny_cfg()
        buffer = _buffered(cfg)
        carved = carved_from_buffer(buffer)
        np.testing.assert_array_equal(carved.pos, buffer.pos_core)
        np.testing.assert_array_equal(carved.vel, buffer.vel_core)
        np.testing.assert_array_equal(carved.mvir, buffer.mvir_core)
        np.testing.assert_array_equal(carved.is_distinct, buffer.is_distinct_core)
        np.testing.assert_array_equal(carved.num_of_subhalos, buffer.num_of_subhalos_core)
        assert carved.n_carved == buffer.n_core
        assert carved.n_total == buffer.n_total


class TestEndToEnd:
    """Both comparisons, both classes, on synthetic arrays."""

    @staticmethod
    def _frame_runs(cfg: HaloClassRunConfig) -> dict[str, ClassFrameRun]:
        carved = _carved()
        n_bar = box_number_density(carved.n_total)
        return {
            center_class: run_frame_comparison_for_class(
                cfg, center_class, carved, _OBSERVER, n_bar
            )
            for center_class in cfg.center_classes
        }

    @staticmethod
    def _redshift_runs(cfg: HaloClassRunConfig) -> dict[str, ClassRedshiftRun]:
        buffer = _buffered(cfg)
        tracers = build_tracer_spaces(cfg, buffer, _OBSERVER)
        n_bar = box_number_density(buffer.n_total)
        return {
            center_class: run_redshift_comparison_for_class(
                cfg, center_class, buffer, tracers, _OBSERVER, n_bar
            )
            for center_class in cfg.center_classes
        }

    def test_the_two_classes_measure_disjoint_center_sets(self) -> None:
        """The classes are disjoint upstream; the CENTERS must stay disjoint too.

        A shared center between the two runs would mean the class filter had
        been undone somewhere between the mask and the estimator.
        """
        cfg = _tiny_cfg()
        runs = self._frame_runs(cfg)
        as_rows = {
            name: {tuple(row) for row in run.centers.s_centers}
            for name, run in runs.items()
        }
        assert not (as_rows[CENTER_CLASS_PARENT] & as_rows[CENTER_CLASS_SUBHALO])

    def test_tracers_are_the_full_carve_not_the_class(self) -> None:
        """Only the centers differ between classes; the density field does not.

        Shell OCCUPANCY is pure geometry over the tracer field. If the tracers
        had been filtered per class, the two runs' total occupancy would scale
        with their class abundances rather than reflecting one shared field --
        so a per-center mean occupancy far apart between classes would be the
        signature. Both classes sample the same ball here, so their mean
        occupancy must agree to within sampling noise.
        """
        cfg = _tiny_cfg()
        runs = self._frame_runs(cfg)
        means = [
            run.results.obs_result.per_center_count.mean() for run in runs.values()
        ]
        assert means[0] == pytest.approx(means[1], rel=0.25)

    def test_both_frames_stay_row_aligned_within_a_class(self) -> None:
        cfg = _tiny_cfg()
        for run in self._frame_runs(cfg).values():
            assert run.results.obs_result.n_centers == run.centers.n_centers
            assert run.results.vel_result.n_centers == run.centers.n_centers

    def test_frame_contrast_figure_builds_with_cartesian_axes(self) -> None:
        """Same axis contract `test_plot_wiring.py` pins for every other figure:
        the BINNING may be logarithmic, the AXES are not."""
        cfg = _tiny_cfg()
        fig = make_frame_class_contrast_figure(cfg, self._frame_runs(cfg))
        for ax in fig.axes:
            assert ax.get_xscale() == "linear"
            assert ax.get_yscale() == "linear"
        # dipole + monopole + the monopole panel's twin
        assert len(fig.axes) == 3

    def test_redshift_contrast_figure_builds_with_cartesian_axes(self) -> None:
        cfg = _tiny_cfg()
        fig = make_redshift_class_contrast_figure(cfg, self._redshift_runs(cfg))
        for ax in fig.axes:
            assert ax.get_xscale() == "linear"
            assert ax.get_yscale() == "linear"
        assert len(fig.axes) == 2  # single shared monopole axis, no twin

    def test_contrast_figures_plot_one_curve_per_class_and_series(self) -> None:
        """Four dipole curves: {two series} x {two classes}."""
        cfg = _tiny_cfg()
        fig = make_frame_class_contrast_figure(cfg, self._frame_runs(cfg))
        labels = [
            text.get_text() for text in fig.axes[0].get_legend().get_texts()
        ]
        assert len(labels) == 2 * len(cfg.center_classes)
        for center_class in cfg.center_classes:
            assert sum(center_class in label for label in labels) == 2

    def test_empty_runs_raise_rather_than_drawing_a_blank_figure(self) -> None:
        cfg = _tiny_cfg()
        with pytest.raises(ValueError, match="runs is empty"):
            make_frame_class_contrast_figure(cfg, {})
        with pytest.raises(ValueError, match="runs is empty"):
            make_redshift_class_contrast_figure(cfg, {})

    def test_a_single_class_run_is_allowed(self) -> None:
        """`center_classes` with one entry must work -- the contrast figure
        degenerates to two curves rather than failing."""
        cfg = _tiny_cfg(center_classes=(CENTER_CLASS_PARENT,))
        fig = make_frame_class_contrast_figure(cfg, self._frame_runs(cfg))
        assert len(fig.axes[0].get_legend().get_texts()) == 2


def test_every_named_class_has_a_marker_and_a_fill() -> None:
    """A class added to `VALID_CENTER_CLASSES` without a marker would raise a
    KeyError deep inside a figure builder, several minutes into a run."""
    from dvcorr.pipeline.halo_class_comparison import (
        _CLASS_BAR_COLOR,
        _CLASS_FACECOLOR,
        _CLASS_MARKER,
    )

    assert set(_CLASS_MARKER) == VALID_CENTER_CLASSES
    assert set(_CLASS_FACECOLOR) == VALID_CENTER_CLASSES
    assert set(_CLASS_BAR_COLOR) == VALID_CENTER_CLASSES


class TestClassShellOccupancy:
    """The sub-h^-1-Mpc occupancy diagnostic: counts only, no velocities.

    The occupancy binning is overridden in every test here. `_synthetic_halos`
    puts 4000 halos in a 60 h^-1 Mpc ball, so the production default
    (`[0, 1] h^-1 Mpc`) would find an empty shell every time and the tests
    would pass on zeros -- which is precisely the failure they exist to catch.
    """

    @staticmethod
    def _cfg() -> HaloClassRunConfig:
        return _tiny_cfg(
            occupancy_shells=ShellConfig(min_radius=0.0, max_radius=12.0, radii_step=3.0)
        )

    def test_mean_and_sem_have_one_entry_per_shell(self) -> None:
        cfg = self._cfg()
        carved = _carved()
        runs = TestEndToEnd._frame_runs(cfg)
        occupancies = class_shell_occupancy_for_runs(
            cfg, runs, carved.pos, box_number_density(carved.n_total)
        )

        n_shells = cfg.occupancy_shells.shell_edges.size - 1
        assert tuple(occupancies) == cfg.center_classes
        for center_class, occupancy in occupancies.items():
            assert occupancy.center_class == center_class
            assert occupancy.mean_count.shape == (n_shells,)
            assert occupancy.sem_count.shape == (n_shells,)
            assert occupancy.expected_count.shape == (n_shells,)
            assert occupancy.n_centers == runs[center_class].centers.n_centers
            assert np.all(occupancy.mean_count > 0.0)

    def test_it_counts_the_run_s_own_centers(self) -> None:
        """`class_shell_occupancy_for_runs` must reuse the centers the frame run
        already selected. Re-drawing them would make the histogram describe a
        different population from the curves it is read beside."""
        cfg = self._cfg()
        carved = _carved()
        n_bar = box_number_density(carved.n_total)
        runs = TestEndToEnd._frame_runs(cfg)

        from_runs = class_shell_occupancy_for_runs(cfg, runs, carved.pos, n_bar)
        direct = {
            center_class: class_shell_occupancy(
                cfg, center_class, run.centers.s_centers, carved.pos, n_bar
            )
            for center_class, run in runs.items()
        }
        for center_class in cfg.center_classes:
            np.testing.assert_array_equal(
                from_runs[center_class].mean_count, direct[center_class].mean_count
            )

    def test_a_uniform_ball_sits_at_the_uniform_field_reference(self) -> None:
        """`_synthetic_halos` IS a uniform field, so `mean_count / expected_count`
        must be ~1. This is what makes a departure from the reference on real
        data readable as clustering rather than as a normalization slip.

        n_bar is the ball's own density here, not `box_number_density`: the
        synthetic population fills a ball, not the periodic box, so the box
        volume would be the wrong denominator to compare a count against.
        """
        cfg = self._cfg()
        carved = _carved()
        ball_volume = (4.0 / 3.0) * np.pi * _SUB_VOLUME_RADIUS**3  # 4pi/3: sphere
        n_bar = carved.n_carved / ball_volume

        occupancy = class_shell_occupancy(
            cfg, CENTER_CLASS_PARENT, carved.pos[:200], carved.pos, n_bar
        )
        np.testing.assert_allclose(
            occupancy.mean_count / occupancy.expected_count, 1.0, rtol=0.20
        )

    def test_figure_builds_with_cartesian_axes_by_default(self) -> None:
        """The project-wide axis contract, restated for the histogram: linear
        unless a caller explicitly asks otherwise."""
        cfg = self._cfg()
        carved = _carved()
        occupancies = class_shell_occupancy_for_runs(
            cfg, TestEndToEnd._frame_runs(cfg), carved.pos,
            box_number_density(carved.n_total),
        )
        ax = make_class_occupancy_figure(cfg, occupancies).axes[0]
        assert ax.get_xscale() == "linear"
        assert ax.get_yscale() == "linear"
        assert make_class_occupancy_figure(cfg, occupancies, log_y=True).axes[0].get_yscale() == "log"

    def test_figure_draws_one_bar_group_per_class_plus_the_reference(self) -> None:
        cfg = self._cfg()
        carved = _carved()
        occupancies = class_shell_occupancy_for_runs(
            cfg, TestEndToEnd._frame_runs(cfg), carved.pos,
            box_number_density(carved.n_total),
        )
        from matplotlib.container import BarContainer

        ax = make_class_occupancy_figure(cfg, occupancies).axes[0]
        bar_groups = [c for c in ax.containers if isinstance(c, BarContainer)]
        assert len(bar_groups) == len(cfg.center_classes)
        assert all(len(group) == cfg.occupancy_shells.shell_edges.size - 1
                   for group in bar_groups)
        # Exactly one labelled line: the n_bar * V_b reference. The other
        # entries in `ax.lines` are the errorbar caps and stems.
        reference = [line for line in ax.lines if line.get_label().startswith("uniform")]
        assert len(reference) == 1
        np.testing.assert_allclose(
            reference[0].get_ydata(), occupancies[CENTER_CLASS_PARENT].expected_count
        )
        assert len(ax.get_legend().get_texts()) == len(cfg.center_classes) + 1

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="occupancies is empty"):
            make_class_occupancy_figure(self._cfg(), {})

    def test_classes_binned_differently_raise(self) -> None:
        """Two classes on different edges would be drawn on one x-axis as though
        they shared it -- a silently wrong figure, so it is refused."""
        cfg = self._cfg()
        carved = _carved()
        n_bar = box_number_density(carved.n_total)
        parents = class_shell_occupancy(
            cfg, CENTER_CLASS_PARENT, carved.pos[:100], carved.pos, n_bar
        )
        other = _tiny_cfg(
            occupancy_shells=ShellConfig(min_radius=0.0, max_radius=6.0, radii_step=3.0)
        )
        subhalos = class_shell_occupancy(
            other, CENTER_CLASS_SUBHALO, carved.pos[:100], carved.pos, n_bar
        )
        with pytest.raises(ValueError, match="binned differently"):
            make_class_occupancy_figure(
                cfg, {CENTER_CLASS_PARENT: parents, CENTER_CLASS_SUBHALO: subhalos}
            )

    def test_occupancy_shells_are_not_the_run_s_shells(self) -> None:
        """The two binnings are independent by design: re-tuning the estimator's
        shells must not silently re-bin this histogram."""
        cfg = HaloClassRunConfig()
        assert cfg.occupancy_shells is not cfg.shells
        np.testing.assert_allclose(
            cfg.occupancy_shells.shell_edges, np.arange(0.0, 1.05, 0.1), atol=1e-12
        )
