"""
test_default_shells.py
------------------------
Pins CLAUDE.md hard rule 10's project-wide default radial binning:
`RunConfig().shells.shell_edges` is the ladder [0, 0.5, 0.5*sqrt2, 1, sqrt2,
2, ..., 32, 45.25, 64] -- the [0, 0.5) inner bin plus 14 geometric shells of
ratio exactly sqrt(2), sixteen edges in all
(`dvcorr.pipeline.velocity_centered._default_shells`).
"""

from __future__ import annotations

import numpy as np
import pytest

from dvcorr.pipeline.velocity_centered import RunConfig


def test_default_shell_edges_match_hard_rule_10():
    edges = RunConfig().shells.shell_edges

    assert edges.size == 16
    assert edges[0] == 0.0
    assert edges[1] == 0.5
    assert edges[-1] == pytest.approx(64.0)

    # Consecutive ratios above the zero bin are all exactly sqrt(2): the
    # zero bin itself (edges[0] -> edges[1], i.e. 0 -> 0.5) is excluded, since
    # a ratio against zero is undefined and it is the one deliberately
    # non-geometric shell in the array (`log_shell_edges`'s Notes).
    geometric_edges = edges[1:]
    ratios = geometric_edges[1:] / geometric_edges[:-1]
    np.testing.assert_allclose(ratios, np.sqrt(2.0))
