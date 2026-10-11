---
paths:
  - "src/dvcorr/pipeline/**"
  - "scripts/**"
  - "notebooks/**"
  - "intuition/**"
  - "tests/test_shell_plotting.py"
  - "tests/test_plot_wiring.py"
---

# Shell figures — rendering rules (hard rule 10, continued)

Binning is a *computation* choice, not a *rendering* one: the estimator accumulates per
shell, but a figure draws one point per shell at the shell's volume-weighted radius
(`volume_weighted_shell_radii`) and joins the points with a line — never a step or
`stairs` curve held flat across the bin. Every figure's x-axis starts at r = 0
(`set_xlim(left=0)`), whatever the innermost edge is.

**Every such line figure also gets a binned companion**: the same arrays, each shell drawn
as a flat segment spanning its own edges [r_lo, r_hi), with its r_eff point marked on it,
bands drawn as boxes over the same span, and the shell edges as faint vertical guides. It
is there so a point on the line figure can be looked up: which shell it belongs to and what
value was measured over it. It is a companion, never a replacement: the line figure stays
primary, and the binned one is the only place a flat-across-the-bin rendering is allowed.

Both views come from one builder called with `binned=False` / `binned=True`, drawing every
series through `dvcorr.pipeline.shell_plotting.draw_shell_series` and `finish_shell_axis`.
That way the two views cannot come from different data. The binned companion draws only the
main signal(s) with their own bands — no null curves, no monopole figure
(`DipoleFigures.monopole` is None and none is created) — since the line view already
carries both and hard rule 6's pair. A notebook shows each binned figure right after its
line figure.

An axis that begins at 1 instead of 0, or a curve drawn as steps, is the failure this rule
exists to prevent.
