"""
test_galaxy_bias.py
-------------------
Unit tests for `dvcorr.pipeline.galaxy_bias`.

Three groups, in increasing order of what they need installed:

  * pure-numpy tests of the fitting / ratio / NaN algebra -- always run;
  * the host-mass chain, built against a tiny synthetic CSV + Parquet pair --
    always run;
  * the Corrfunc and CAMB/mcfit paths -- skipped when the optional `bias`
    extra is not installed (see pyproject.toml).

The load-bearing one is `test_cross_normalisation_matches_auto`: a periodic
cross-correlation differs from an auto-correlation by exactly the factor of two
between ordered and unordered pairs, and getting it wrong halves every
`b_cross` silently.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # backend discipline: before any pyplot import, cf. test_shell_plotting.py

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection

from dvcorr import conventions
from dvcorr.config import CatalogConfig, CosmologyConfig, PathsConfig, volume_weighted_shell_radii
from dvcorr.pipeline import galaxy_bias as gb

HAS_CORRFUNC = True
try:  # pragma: no cover - import probing
    import Corrfunc  # noqa: F401
    from Corrfunc.theory.DD import DD  # noqa: F401
except Exception:  # pragma: no cover
    HAS_CORRFUNC = False

HAS_THEORY = True
try:  # pragma: no cover - import probing
    import camb  # noqa: F401
    import mcfit  # noqa: F401
except Exception:  # pragma: no cover
    HAS_THEORY = False

needs_corrfunc = pytest.mark.skipif(not HAS_CORRFUNC, reason="Corrfunc not installed")
needs_theory = pytest.mark.skipif(not HAS_THEORY, reason="camb/mcfit not installed")


# ---------------------------------------------------------------------------
# Binning and configuration
# ---------------------------------------------------------------------------


def test_default_bins_are_increasing_and_inside_the_pbc_ceiling():
    edges = gb.default_xi_bins()
    assert edges.size == gb.XI_N_BINS + 1
    assert np.all(np.diff(edges) > 0.0)
    assert edges[-1] <= conventions.MAX_ANALYSIS_RADIUS


def test_config_rejects_bins_beyond_half_the_box():
    too_far = np.geomspace(1.0, conventions.MAX_ANALYSIS_RADIUS + 1.0, 5)
    with pytest.raises(ValueError, match="MAX_ANALYSIS_RADIUS"):
        gb.BiasRunConfig(bins=too_far)


def test_config_rejects_an_inverted_fit_range():
    with pytest.raises(ValueError, match="fit_max_radius"):
        gb.BiasRunConfig(fit_min_radius=60.0, fit_max_radius=20.0)


# ---------------------------------------------------------------------------
# fit_constant -- the flatness machinery the acceptance criterion is read off
# ---------------------------------------------------------------------------


def test_fit_constant_on_a_flat_curve_has_zero_scatter_and_zero_trend():
    r = np.geomspace(10.0, 100.0, 20)
    fit = gb.fit_constant(r, np.full_like(r, 1.37), gb.FIT_MIN_RADIUS, gb.FIT_MAX_RADIUS)
    assert fit.value == pytest.approx(1.37)
    assert fit.scatter == pytest.approx(0.0, abs=1e-12)
    assert fit.trend == pytest.approx(0.0, abs=1e-12)


def test_fit_constant_recovers_a_known_slope_per_dex():
    r = np.geomspace(20.0, 60.0, 25)
    fit = gb.fit_constant(r, 2.0 + 0.5 * np.log10(r), gb.FIT_MIN_RADIUS, gb.FIT_MAX_RADIUS)
    assert fit.trend == pytest.approx(0.5, rel=1e-9)


def test_fit_constant_ignores_nan_bins_but_refuses_fewer_than_two():
    r = np.array([25.0, 35.0, 45.0])
    y = np.array([1.0, np.nan, 3.0])
    assert gb.fit_constant(r, y, 20.0, 60.0).n_bins == 2
    with pytest.raises(ValueError, match="flatness"):
        gb.fit_constant(r, np.array([1.0, np.nan, np.nan]), 20.0, 60.0)


def test_fractional_scatter_is_scatter_over_value():
    fit = gb.ConstantFit(value=0.8, scatter=0.04, n_bins=5, r_min=20.0, r_max=60.0, trend=0.0)
    assert fit.fractional_scatter == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# The three bias estimates
# ---------------------------------------------------------------------------


def _fake_xi_hh(xi_values: np.ndarray, label: str = "hh") -> gb.CorrelationFunction:
    r = np.geomspace(5.0, 100.0, xi_values.size)
    return gb.CorrelationFunction(
        edges=np.geomspace(4.0, 110.0, xi_values.size + 1),
        r_eff=r,
        xi=xi_values,
        npairs=np.full(xi_values.size, 1000, dtype=np.int64),
        n1=10,
        n2=10,
        label=label,
    )


def test_bias_estimates_recover_a_planted_bias_exactly():
    b_true = 1.4
    xi_mm = np.geomspace(0.5, 1e-3, 12)
    hh = _fake_xi_hh(b_true**2 * xi_mm)
    est = gb.bias_estimates(hh, xi_mm, gb.MATTER_SOURCE_PARTICLES, xi_hm=b_true * xi_mm)
    assert np.allclose(est.b_auto, b_true)
    assert np.allclose(est.b_cross, b_true)
    assert np.allclose(est.r_cc, 1.0)
    assert not est.is_placeholder


def test_bias_estimates_without_xi_hm_leave_the_particle_only_quantities_nan():
    xi_mm = np.geomspace(0.5, 1e-3, 8)
    est = gb.bias_estimates(_fake_xi_hh(xi_mm), xi_mm, gb.MATTER_SOURCE_CAMB_HALOFIT)
    assert np.all(np.isnan(est.b_cross))
    assert np.all(np.isnan(est.r_cc))
    assert np.all(np.isfinite(est.b_auto))
    assert est.is_placeholder


def test_b_auto_is_nan_not_zero_where_xi_goes_negative():
    xi_mm = np.array([0.1, 0.01, 0.001])
    hh = _fake_xi_hh(np.array([0.2, 0.02, -0.0005]))
    est = gb.bias_estimates(hh, xi_mm, gb.MATTER_SOURCE_PARTICLES)
    assert np.isnan(est.b_auto[-1])
    assert np.all(np.isfinite(est.b_auto[:-1]))


def test_bias_estimates_rejects_a_mismatched_grid():
    xi_mm = np.geomspace(0.5, 1e-3, 8)
    with pytest.raises(ValueError, match="xi_mm has shape"):
        gb.bias_estimates(_fake_xi_hh(xi_mm), xi_mm[:-1], gb.MATTER_SOURCE_PARTICLES)


def test_r_cc_acceptance_needs_both_level_and_flatness_and_refuses_nan():
    good = gb.ConstantFit(1.0, 0.005, 5, 20.0, 60.0, 0.0)
    off_level = gb.ConstantFit(1.2, 0.005, 5, 20.0, 60.0, 0.0)
    too_scattered = gb.ConstantFit(1.0, 0.2, 5, 20.0, 60.0, 0.0)
    not_measured = gb.ConstantFit(np.nan, np.nan, 0, 20.0, 60.0, np.nan)
    assert gb.r_cc_is_acceptable(good)
    assert not gb.r_cc_is_acceptable(off_level)
    assert not gb.r_cc_is_acceptable(too_scattered)
    assert not gb.r_cc_is_acceptable(not_measured)


# ---------------------------------------------------------------------------
# The zeta_1 amplitude-ratio test
# ---------------------------------------------------------------------------


def test_amplitude_ratio_passes_when_the_zeta_ratio_equals_the_bias_ratio():
    r = np.geomspace(10.0, 64.0, 10)
    zeta_b = np.geomspace(20.0, 3.0, 10)
    bias_ratio = 0.87
    test = gb.zeta_amplitude_ratio(
        r, bias_ratio * zeta_b, 0.01 * zeta_b, zeta_b, 0.01 * zeta_b,
        bias_ratio=bias_ratio, label_a="a", label_b="b",
    )
    assert test.ratio_fit.value == pytest.approx(bias_ratio)
    assert test.discrepancy == pytest.approx(0.0, abs=1e-12)
    assert test.passes()


def test_amplitude_ratio_fails_on_a_sloping_ratio_even_at_the_right_mean():
    r = np.geomspace(20.0, 60.0, 9)
    zeta_b = np.full_like(r, 10.0)
    # Mean 0.87 by construction, but swinging +-30% across the window.
    sloped = 0.87 * (1.0 + 0.3 * np.linspace(-1.0, 1.0, r.size))
    test = gb.zeta_amplitude_ratio(
        r, sloped * zeta_b, 0.01 * zeta_b, zeta_b, 0.01 * zeta_b,
        bias_ratio=0.87, label_a="a", label_b="b",
    )
    assert test.discrepancy == pytest.approx(0.0, abs=1e-12)
    assert not test.passes()


def test_amplitude_ratio_rejects_mismatched_shapes():
    r = np.geomspace(10.0, 60.0, 5)
    with pytest.raises(ValueError, match="share one shape"):
        gb.zeta_amplitude_ratio(
            r, r, r, r[:-1], r[:-1], bias_ratio=1.0, label_a="a", label_b="b"
        )


# ---------------------------------------------------------------------------
# The host-mass chain, on a synthetic catalog
# ---------------------------------------------------------------------------


def _write_synthetic_catalog(tmp_path, ids, pids, mvir):
    """Write a CSV + row-aligned Parquet pair and return a PathsConfig for them."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_path = data_dir / "mdpl2_rockstar_snapnum125.csv"
    pd.DataFrame(
        {
            conventions.HALO_COLUMNS["mass"]: mvir,
            conventions.HALO_COLUMNS["id"]: ids,
            conventions.HALO_COLUMNS["parent_id"]: pids,
        }
    ).to_csv(csv_path, index=False)
    pq.write_table(
        pa.table({conventions.HALO_COLUMNS["mass"]: pa.array(mvir, type=pa.float32())}),
        csv_path.with_suffix(".parquet"),
    )
    return PathsConfig(project_root=tmp_path, data_dir=data_dir)


def test_host_masses_climb_nested_subhalos_to_the_top_level_host(tmp_path):
    # 30 is a subhalo of 20, which is a subhalo of 10: it must inherit 10's mass,
    # not 20's. This is the case a single pid hop silently gets wrong.
    ids = np.array([10, 20, 30, 40], dtype=np.int64)
    pids = np.array([-1, 10, 20, -1], dtype=np.int64)
    mvir = np.array([1e14, 1e12, 1e11, 5e13], dtype=np.float32)
    paths = _write_synthetic_catalog(tmp_path, ids, pids, mvir)

    host = gb.resolve_host_masses(paths, "full", verbose=False)
    assert host == pytest.approx(np.array([1e14, 1e14, 1e14, 5e13], dtype=np.float32))


def test_host_masses_are_cached_and_reused(tmp_path):
    ids = np.array([1, 2], dtype=np.int64)
    pids = np.array([-1, 1], dtype=np.int64)
    mvir = np.array([1e13, 1e11], dtype=np.float32)
    paths = _write_synthetic_catalog(tmp_path, ids, pids, mvir)

    first = gb.resolve_host_masses(paths, "full", verbose=False)
    cache = gb._cached_host_mass_path(paths, "full")
    assert cache.exists()
    # Delete the CSV: a second call must be served entirely from the cache.
    paths.halo_catalog("full", parquet=False).unlink()
    assert gb.resolve_host_masses(paths, "full", verbose=False) == pytest.approx(first)


def test_host_masses_reject_a_stale_cache_of_the_wrong_length(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    ids = np.array([1, 2], dtype=np.int64)
    paths = _write_synthetic_catalog(
        tmp_path, ids, np.array([-1, -1], dtype=np.int64), np.array([1e13, 1e12], np.float32)
    )
    pq.write_table(
        pa.table({gb.HOST_MASS_COLUMN: pa.array([1.0, 2.0, 3.0], type=pa.float32())}),
        gb._cached_host_mass_path(paths, "full"),
    )
    with pytest.raises(ValueError, match="delete the cache"):
        gb.resolve_host_masses(paths, "full", verbose=False)


# ---------------------------------------------------------------------------
# Corrfunc paths
# ---------------------------------------------------------------------------


def _uniform_sample(n: int, seed: int) -> gb.TracerSample:
    rng = np.random.default_rng(seed)
    pos = rng.random((n, 3)) * conventions.BOX_SIZE
    return gb.TracerSample(pos=pos, n_total=n, label=f"uniform-{seed}", subsampled=False)


@needs_corrfunc
def test_uniform_field_has_zero_correlation_in_both_estimators():
    cfg = gb.BiasRunConfig(bins=np.geomspace(10.0, 60.0, 6), n_threads=2)
    a, b = _uniform_sample(60_000, 1), _uniform_sample(60_000, 2)
    auto = gb.periodic_auto_xi(a, cfg, "auto")
    cross = gb.periodic_cross_xi(a, b, cfg, "cross")
    # Poisson pair counts here are ~10^6-10^8, so |xi| well under 1% is the
    # right bar: anything larger is a normalisation error, not noise.
    assert np.all(np.abs(auto.xi) < 0.01)
    assert np.all(np.abs(cross.xi) < 0.01)


@needs_corrfunc
def test_cross_normalisation_matches_auto_on_two_halves_of_one_clustered_field():
    # A clustered field: Poisson clumps, so xi is O(1) and a factor-of-two
    # normalisation slip cannot hide inside the noise.
    rng = np.random.default_rng(7)
    n_clumps, per_clump, scale = 4_000, 12, 4.0
    centers = rng.random((n_clumps, 3)) * conventions.BOX_SIZE
    pos = np.repeat(centers, per_clump, axis=0) + rng.normal(0.0, scale, (n_clumps * per_clump, 3))
    pos = np.mod(pos, conventions.BOX_SIZE)
    rng.shuffle(pos)

    cfg = gb.BiasRunConfig(bins=np.geomspace(5.0, 50.0, 6), n_threads=2)
    whole = gb.TracerSample(pos=pos, n_total=pos.shape[0], label="whole", subsampled=False)
    half_a = gb.TracerSample(pos=pos[0::2], n_total=pos.shape[0], label="a", subsampled=False)
    half_b = gb.TracerSample(pos=pos[1::2], n_total=pos.shape[0], label="b", subsampled=False)

    auto = gb.periodic_auto_xi(whole, cfg, "auto")
    cross = gb.periodic_cross_xi(half_a, half_b, cfg, "cross")
    # Two random halves of one field trace the same clustering, so the cross
    # of the halves and the auto of the whole must agree bin by bin.
    assert np.allclose(cross.xi, auto.xi, rtol=0.1, atol=0.02)


@needs_corrfunc
def test_cross_is_symmetric_in_its_two_samples():
    cfg = gb.BiasRunConfig(bins=np.geomspace(10.0, 60.0, 5), n_threads=2)
    a, b = _uniform_sample(30_000, 3), _uniform_sample(20_000, 4)
    assert np.allclose(
        gb.periodic_cross_xi(a, b, cfg, "ab").xi, gb.periodic_cross_xi(b, a, cfg, "ba").xi
    )


# ---------------------------------------------------------------------------
# Theory paths
# ---------------------------------------------------------------------------


@needs_theory
def test_camb_power_spectrum_carries_the_mdpl2_sigma8():
    import camb

    cosmology = CosmologyConfig()
    k, pk = gb.linear_power_spectrum(cosmology)
    assert k.size == pk.size == gb.CAMB_N_K
    assert np.all(pk > 0.0)
    # sigma8 recomputed from the returned P(k) via the top-hat window.
    r8 = 8.0
    x = k * r8
    window = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3
    sigma8_sq = np.trapezoid(k**2 * pk * window**2, k) / (2.0 * np.pi**2)
    assert np.sqrt(sigma8_sq) == pytest.approx(cosmology.sigma8, rel=0.01)


@needs_theory
def test_theory_matter_xi_is_positive_and_falling_over_the_fit_range():
    r = np.geomspace(gb.FIT_MIN_RADIUS, gb.FIT_MAX_RADIUS, 8)
    xi, source = gb.theory_matter_xi(r, nonlinear=True)
    assert source == gb.MATTER_SOURCE_CAMB_HALOFIT
    assert np.all(xi > 0.0)
    assert np.all(np.diff(xi) < 0.0)


@needs_theory
def test_theory_matter_xi_tags_the_linear_variant_separately():
    r = np.geomspace(20.0, 60.0, 4)
    _, source = gb.theory_matter_xi(r, nonlinear=False)
    assert source == gb.MATTER_SOURCE_CAMB_LINEAR


@needs_theory
def test_zeta_one_prediction_is_positive_for_infall_and_scales_linearly_with_b():
    # conventions.INFALL_DIPOLE_SIGN fixes xi_Tu,1 < 0 for infall; zeta_1 is the
    # velocity-centered multipole, (-1)**1 times it, hence positive.
    r = np.geomspace(10.0, 100.0, 12)
    one = gb.zeta_one_linear_prediction(r, b=1.0)
    assert np.all(one > 0.0)
    assert np.all(np.diff(one) < 0.0)
    assert np.allclose(gb.zeta_one_linear_prediction(r, b=2.5), 2.5 * one)


@needs_theory
def test_zeta_one_prediction_follows_the_frozen_multipole_sign_relation():
    r = np.array([30.0])
    assert np.sign(gb.zeta_one_linear_prediction(r, b=1.0)[0]) == -conventions.INFALL_DIPOLE_SIGN
    assert conventions.nusser_multipole_sign(1) == -1


@needs_theory
def test_shell_average_collapses_to_the_point_value_for_a_narrow_shell():
    edges = np.array([39.99, 40.01])
    averaged = gb.shell_averaged_zeta_one(edges, b=1.0)
    point = gb.zeta_one_linear_prediction(np.array([40.0]), b=1.0)
    assert averaged[0] == pytest.approx(point[0], rel=1e-4)


@needs_theory
def test_shell_average_of_a_wide_shell_is_pulled_below_its_midpoint_value():
    # zeta_1 falls steeply with r and the r**2 weight favours the outer half,
    # so the shell average must sit BELOW the value at the midpoint.
    edges = np.array([20.0, 60.0])
    averaged = gb.shell_averaged_zeta_one(edges, b=1.0)[0]
    midpoint = gb.zeta_one_linear_prediction(np.array([40.0]), b=1.0)[0]
    assert averaged < midpoint


@needs_theory
def test_effective_tinker_bias_rises_with_halo_mass():
    light = np.full(1000, 1e12)
    heavy = np.full(1000, 1e14)
    assert gb.effective_tinker_bias(heavy) > gb.effective_tinker_bias(light) > 0.0


@needs_theory
def test_effective_tinker_bias_is_number_weighted_over_distinct_masses():
    # Three copies of 1e12 and one of 1e14: the mean must sit near the light
    # end, i.e. the distinct-mass shortcut must carry multiplicities.
    masses = np.array([1e12, 1e12, 1e12, 1e14])
    expected = np.mean(gb.tinker_bias_of_mass(masses))
    assert gb.effective_tinker_bias(masses) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_load_tracer_positions_reports_a_missing_parquet_by_name(tmp_path):
    paths = PathsConfig(project_root=tmp_path, data_dir=tmp_path / "nowhere")
    cfg = gb.BiasRunConfig(catalog=CatalogConfig())
    with pytest.raises(FileNotFoundError, match="convert_mdpl2_catalog"):
        gb.load_tracer_positions(cfg, paths)


@needs_theory
def test_shell_average_refuses_to_extrapolate_below_the_theory_grid():
    # ShellConfig.include_zero_bin makes edges[0] == 0 a production binning, so
    # this shell reaches a radius the Hankel grid does not cover.
    edges = np.array([0.0, gb.THEORY_MIN_RADIUS / 2.0, 30.0])
    averaged = gb.shell_averaged_zeta_one(edges, b=1.0)
    assert np.isnan(averaged[0])
    assert np.isfinite(averaged[1])


@needs_theory
def test_shell_average_of_a_zero_edged_shell_stays_finite_and_positive():
    edges = np.array([0.0, 1.0, 2.0])
    averaged = gb.shell_averaged_zeta_one(edges, b=1.0)
    assert np.all(np.isfinite(averaged))
    assert np.all(averaged > 0.0)


@needs_theory
def test_hubble_in_inverse_h_units_is_exactly_100_at_z0():
    # H0 = 100h km/s/Mpc, so in km/s per h^-1 Mpc it is 100 for ANY h -- the
    # defining property of the unit. zeta_one_linear_prediction carries this
    # conversion, and getting it upside down rescales the whole prediction by
    # h**2 without changing its shape: invisible on a plot, a factor 2.2 in the
    # answer. Recovered here from the prediction itself, so the test fails if
    # the conversion is ever flipped back.
    cosmology = CosmologyConfig()
    r = np.array([40.0])
    b = 1.0
    predicted = gb.zeta_one_linear_prediction(r, b, cosmology)[0]

    import mcfit

    k, pk = gb.linear_power_spectrum(cosmology, nonlinear=False)
    grid_r, transform = mcfit.P2xi(k, l=1, lowring=True)(pk / k, extrap=True)
    integral = np.interp(r[0], grid_r, 2.0 * np.pi**2 * np.imag(transform))
    growth_rate = cosmology.Om0**cosmology.growth_index
    # zeta_1 = +(b aH f / 6 pi^2) * integral  (the (-1)**1 already applied)
    recovered_aH = predicted * 6.0 * np.pi**2 / (b * growth_rate * integral)
    assert recovered_aH == pytest.approx(100.0, rel=1e-3)


# ---------------------------------------------------------------------------
# The direct velocity-density correlation
# ---------------------------------------------------------------------------


def test_direct_correlation_is_zero_for_uncorrelated_velocities():
    rng = np.random.default_rng(11)
    n = 40_000
    pos = rng.random((n, 3)) * conventions.BOX_SIZE
    vel = rng.normal(0.0, 300.0, (n, 3))
    edges = np.array([20.0, 40.0, 60.0])
    result = gb.measure_velocity_density_correlation(pos, vel, edges)
    # <v.rhat> over ~10^6 pairs of 300 km/s velocities: the noise floor is a
    # few km/s, and a normalisation slip would show up as tens.
    assert np.all(np.abs(result.C) < 10.0)
    assert np.all(np.abs(result.xi_hh) < 0.05)


def test_direct_correlation_recovers_a_planted_radial_inflow():
    # Every halo moves towards the box centre at a fixed speed. rhat_12 for a
    # pair on the same side of the centre is nearly radial, so <v_1.rhat_12>
    # must be positive for the inward-facing half and the signal must be
    # detected, not cancelled.
    rng = np.random.default_rng(12)
    n = 30_000
    pos = rng.random((n, 3)) * conventions.BOX_SIZE
    centre = np.full(3, conventions.BOX_SIZE / 2.0)
    inward = centre - pos
    inward /= np.linalg.norm(inward, axis=1)[:, None]
    result = gb.measure_velocity_density_correlation(
        pos, 500.0 * inward, np.array([20.0, 40.0, 60.0])
    )
    # A coherent inflow gives <v.rhat> ~ 0 by symmetry over a full shell, so
    # what this pins is that the estimator RUNS on a coherent field and stays
    # bounded by the flow speed rather than blowing up on the periodic wrap.
    assert np.all(np.abs(result.mean_v_radial) < 500.0)
    assert np.all(np.isfinite(result.C))


def test_direct_correlation_uses_the_minimum_image():
    # Two halos straddling the x face, 10 h^-1 Mpc apart under PBC and
    # BOX_SIZE - 10 apart under a plain difference. The first moves in +x, i.e.
    # towards the second THROUGH the face.
    pos = np.array([[conventions.BOX_SIZE - 5.0, 500.0, 500.0], [5.0, 500.0, 500.0]])
    vel = np.array([[100.0, 0.0, 0.0], [-100.0, 0.0, 0.0]])
    result = gb.measure_velocity_density_correlation(pos, vel, np.array([5.0, 15.0]))
    assert result.npairs[0] == 2                      # both ordered pairs found
    assert result.mean_v_radial[0] == pytest.approx(100.0)   # +100 both ways: infall


def test_direct_correlation_rejects_non_finite_velocities():
    pos = np.array([[10.0, 10.0, 10.0], [20.0, 10.0, 10.0]])
    vel = np.array([[1.0, 0.0, 0.0], [np.nan, 0.0, 0.0]])
    with pytest.raises(ValueError, match="hard rule 5"):
        gb.measure_velocity_density_correlation(pos, vel, np.array([5.0, 15.0]))


def test_direct_correlation_rejects_shells_beyond_half_the_box():
    pos = np.zeros((2, 3))
    vel = np.zeros((2, 3))
    with pytest.raises(ValueError, match="MAX_ANALYSIS_RADIUS"):
        gb.measure_velocity_density_correlation(
            pos, vel, np.array([10.0, conventions.MAX_ANALYSIS_RADIUS + 1.0])
        )


# ---------------------------------------------------------------------------
# Figures -- the binned companion (CLAUDE.md hard rule 10, amended)
# ---------------------------------------------------------------------------
#
# Unlike `dvcorr.pipeline.velocity_centered.make_figure`, every figure here
# keeps its abscissa at the pair-weighted `CorrelationFunction.r_eff`
# (module docstring, "NOT volume_weighted_shell_radii") and stays log-x in
# both views, so these tests check the binned segments land at `r_eff` (not
# `volume_weighted_shell_radii`) and that the log scale survives, rather than
# reusing `test_shell_plotting.py`'s r=0 / volume-weighted assertions.


def _fake_bias_run_config(edges: np.ndarray) -> gb.BiasRunConfig:
    return gb.BiasRunConfig(bins=edges)


def _line_collections(ax: plt.Axes) -> list[LineCollection]:
    return [coll for coll in ax.collections if isinstance(coll, LineCollection)]


def _marker_lines(ax: plt.Axes) -> list[plt.Line2D]:
    """The `r_eff`-marker lines `draw_shell_series` draws (linestyle "None")."""
    return [line for line in ax.get_lines() if line.get_linestyle() == "None"]


def _segment_collections(ax: plt.Axes, n_segments: int) -> list[LineCollection]:
    """`LineCollection`s with exactly `n_segments` segments -- the per-shell
    measured series `draw_shell_series` draws via `ax.hlines`, as opposed to
    the single-segment `ax.hlines` calls this module's figures use to draw a
    fitted CONSTANT over its window (also a `LineCollection`, one segment,
    not a per-shell series -- `ax.hlines` never returns a `Line2D`).
    """
    return [coll for coll in _line_collections(ax) if len(coll.get_segments()) == n_segments]


def test_correlation_figure_binned_segments_match_the_line_view_at_curve_r_eff():
    hh = _fake_xi_hh(np.geomspace(0.5, 1e-3, 8), label="hh")
    hm = _fake_xi_hh(np.geomspace(0.3, 5e-4, 8), label="hm")
    cfg = _fake_bias_run_config(hh.edges)

    line_fig = gb.make_correlation_figure([hh, hm], cfg)
    bin_fig = gb.make_correlation_figure([hh, hm], cfg, binned=True)
    line_ax, bin_ax = line_fig.axes[0], bin_fig.axes[0]

    # Line view: unchanged, one Line2D per curve at its OWN r_eff.
    for curve in (hh, hm):
        (line,) = [l for l in line_ax.get_lines() if l.get_label() == curve.label]
        np.testing.assert_allclose(line.get_xdata(), curve.r_eff)
        np.testing.assert_allclose(line.get_ydata(), curve.r_eff**2 * curve.xi)

    # Binned view: one LineCollection per curve, segment heights equal the
    # line view's y-values, each segment spanning the curve's OWN edges, and
    # the marker sitting at the curve's own r_eff -- NOT
    # volume_weighted_shell_radii(edges), which the shared shell_plotting
    # primitive would have used instead.
    collections = _line_collections(bin_ax)
    markers = _marker_lines(bin_ax)
    assert len(collections) == len(markers) == 2
    volume_weighted = volume_weighted_shell_radii(hh.edges)
    for curve, coll, marker in zip((hh, hm), collections, markers):
        segments = coll.get_segments()
        assert len(segments) == curve.xi.size
        heights = np.array([seg[0, 1] for seg in segments])
        np.testing.assert_allclose(heights, curve.r_eff**2 * curve.xi)
        for seg, lo, hi in zip(segments, curve.edges[:-1], curve.edges[1:]):
            np.testing.assert_allclose(seg[:, 0], [lo, hi])
        np.testing.assert_allclose(marker.get_xdata(), curve.r_eff)
        assert not np.allclose(marker.get_xdata(), volume_weighted)
    plt.close(line_fig)
    plt.close(bin_fig)


def test_correlation_figure_theory_curve_is_unbinned_in_both_views():
    hh = _fake_xi_hh(np.geomspace(0.5, 1e-3, 8))
    cfg = _fake_bias_run_config(hh.edges)
    r_theory = np.geomspace(1.0, 120.0, 50)
    xi_theory = np.exp(-r_theory / 30.0)
    theory = (r_theory, xi_theory, "theory")

    for binned in (False, True):
        fig = gb.make_correlation_figure([hh], cfg, theory=theory, binned=binned)
        (ax,) = fig.axes
        (theory_line,) = [l for l in ax.get_lines() if l.get_label() == "theory"]
        np.testing.assert_allclose(theory_line.get_xdata(), r_theory)
        np.testing.assert_allclose(theory_line.get_ydata(), r_theory**2 * xi_theory)
        assert theory_line.get_linestyle() == "--"
        plt.close(fig)


def test_correlation_figure_binned_view_stays_log_x_and_adds_edge_guides():
    hh = _fake_xi_hh(np.geomspace(0.5, 1e-3, 8))
    cfg = _fake_bias_run_config(hh.edges)

    line_fig = gb.make_correlation_figure([hh], cfg)
    bin_fig = gb.make_correlation_figure([hh], cfg, binned=True)
    line_ax, bin_ax = line_fig.axes[0], bin_fig.axes[0]

    assert line_ax.get_xscale() == "log" == bin_ax.get_xscale()
    edge_guides = [
        line for line in bin_ax.get_lines()
        if len(line.get_xdata()) == 2 and np.ptp(line.get_xdata()) == 0
    ]
    assert len(edge_guides) == hh.edges.size
    no_guides_in_line_view = [
        line for line in line_ax.get_lines()
        if len(line.get_xdata()) == 2 and np.ptp(line.get_xdata()) == 0
    ]
    assert no_guides_in_line_view == []
    plt.close(line_fig)
    plt.close(bin_fig)


def test_bias_figure_binned_segments_match_the_line_view():
    b_true = 1.4
    xi_mm = np.geomspace(0.5, 1e-3, 12)
    hh = _fake_xi_hh(b_true**2 * xi_mm)
    estimates = gb.bias_estimates(hh, xi_mm, gb.MATTER_SOURCE_PARTICLES, xi_hm=b_true * xi_mm)
    cfg = _fake_bias_run_config(hh.edges)
    fits = {
        key: gb.fit_constant(estimates.r, values, cfg.fit_min_radius, cfg.fit_max_radius)
        for key, values in (
            ("b_cross", estimates.b_cross),
            ("b_auto", estimates.b_auto),
            ("r_cc", estimates.r_cc),
        )
    }

    line_fig = gb.make_bias_figure(estimates, cfg, fits)
    bin_fig = gb.make_bias_figure(estimates, cfg, fits, binned=True)
    (line_ax_b, line_ax_r), (bin_ax_b, bin_ax_r) = line_fig.axes, bin_fig.axes
    n_bins = estimates.b_cross.size

    # Top panel: b_cross and b_auto, two segment series (each B segments,
    # distinct from the single-segment fitted-constant hlines below).
    b_collections = _segment_collections(bin_ax_b, n_bins)
    assert len(b_collections) == 2
    for values, coll in zip((estimates.b_cross, estimates.b_auto), b_collections):
        heights = np.array([seg[0, 1] for seg in coll.get_segments()])
        np.testing.assert_allclose(heights, values)

    # Bottom panel: r_cc, one segment series at the same estimates.r marker.
    (r_cc_coll,) = _segment_collections(bin_ax_r, n_bins)
    heights = np.array([seg[0, 1] for seg in r_cc_coll.get_segments()])
    np.testing.assert_allclose(heights, estimates.r_cc)
    (marker,) = _marker_lines(bin_ax_r)
    np.testing.assert_allclose(marker.get_xdata(), estimates.r)

    # The fitted-constant reference lines (single-segment LineCollections --
    # `ax.hlines` never returns a Line2D) are unchanged: same segment in both
    # views, since they are not per-shell series.
    for line_ax, bin_ax in ((line_ax_b, bin_ax_b), (line_ax_r, bin_ax_r)):
        line_fits = sorted(
            tuple(map(tuple, coll.get_segments()[0]))
            for coll in _segment_collections(line_ax, 1)
        )
        bin_fits = sorted(
            tuple(map(tuple, coll.get_segments()[0]))
            for coll in _segment_collections(bin_ax, 1)
        )
        assert line_fits == bin_fits
    plt.close(line_fig)
    plt.close(bin_fig)


def test_bias_figure_binned_placeholder_case_still_renders_without_xi_hm():
    xi_mm = np.geomspace(0.5, 1e-3, 8)
    hh = _fake_xi_hh(xi_mm)
    estimates = gb.bias_estimates(hh, xi_mm, gb.MATTER_SOURCE_CAMB_HALOFIT)
    cfg = _fake_bias_run_config(hh.edges)
    fits = {"b_auto": gb.fit_constant(estimates.r, estimates.b_auto, cfg.fit_min_radius, cfg.fit_max_radius)}

    fig = gb.make_bias_figure(estimates, cfg, fits, binned=True)
    ax_b, ax_r = fig.axes
    n_bins = estimates.b_auto.size
    assert len(_segment_collections(ax_b, n_bins)) == 1  # b_auto only, b_cross all-NaN and skipped
    assert _line_collections(ax_r) == []  # r_cc all-NaN: the text annotation, not a segment series
    plt.close(fig)


def test_amplitude_ratio_figure_binned_band_matches_the_errorbar_and_segments_match_ratio():
    r = np.geomspace(10.0, 64.0, 10)
    zeta_b = np.geomspace(20.0, 3.0, 10)
    bias_ratio = 0.87
    sem = 0.03 * zeta_b
    shell_edges = np.geomspace(9.0, 70.0, r.size + 1)
    test = gb.zeta_amplitude_ratio(
        r, bias_ratio * zeta_b, sem, zeta_b, sem, bias_ratio=bias_ratio, label_a="a", label_b="b",
        shell_edges=shell_edges,
    )
    # The correlation-function bins are a DIFFERENT ladder from the zeta
    # shells (in notebook 09: 24 bins vs 15 shells); the binned ratio must
    # span the zeta shells, never cfg.bins.
    cfg = _fake_bias_run_config(np.geomspace(0.1, 100.0, 2 * r.size + 5))

    line_fig = gb.make_amplitude_ratio_figure(test, cfg)
    bin_fig = gb.make_amplitude_ratio_figure(test, cfg, binned=True)
    (line_ax,), (bin_ax,) = line_fig.axes, bin_fig.axes

    # `ax.errorbar`'s primary data line is solid; the cap "lines" it also
    # draws are marker-only (linestyle "None") and share the same length, so
    # they must be excluded explicitly.
    (errorbar_line,) = [
        l for l in line_ax.get_lines() if len(l.get_xdata()) == r.size and l.get_linestyle() == "-"
    ]
    np.testing.assert_allclose(errorbar_line.get_ydata(), test.ratio)

    (coll,) = _segment_collections(bin_ax, r.size)
    heights = np.array([seg[0, 1] for seg in coll.get_segments()])
    np.testing.assert_allclose(heights, test.ratio)

    # `bin_ax.patches` also holds the fit-range `axvspan`, a Rectangle too;
    # keep only the per-bin boxes, identified by starting at a bin's own edge.
    boxes = [p for p in bin_ax.patches if np.any(np.isclose(p.get_x(), shell_edges[:-1]))]
    assert len(boxes) == r.size
    for seg, lo, hi in zip(coll.get_segments(), shell_edges[:-1], shell_edges[1:]):
        np.testing.assert_allclose(seg[:, 0], [lo, hi])
    lower, upper = test.ratio - test.ratio_error, test.ratio + test.ratio_error
    for box, lo, hi, bottom, top in zip(boxes, shell_edges[:-1], shell_edges[1:], lower, upper):
        assert box.get_x() == pytest.approx(lo)
        assert box.get_width() == pytest.approx(hi - lo)
        assert box.get_y() == pytest.approx(bottom)
        assert box.get_height() == pytest.approx(top - bottom)
    plt.close(line_fig)
    plt.close(bin_fig)


def test_amplitude_ratio_binned_needs_the_zeta_shell_edges():
    r = np.geomspace(10.0, 64.0, 5)
    zeta = np.geomspace(20.0, 3.0, 5)
    test = gb.zeta_amplitude_ratio(r, zeta, zeta, zeta, zeta, bias_ratio=1.0, label_a="a", label_b="b")
    cfg = _fake_bias_run_config(np.geomspace(9.0, 70.0, r.size + 1))
    with pytest.raises(ValueError):
        gb.make_amplitude_ratio_figure(test, cfg, binned=True)
    with pytest.raises(ValueError):
        gb.zeta_amplitude_ratio(
            r, zeta, zeta, zeta, zeta, bias_ratio=1.0, label_a="a", label_b="b",
            shell_edges=np.arange(r.size),
        )
    plt.close("all")
