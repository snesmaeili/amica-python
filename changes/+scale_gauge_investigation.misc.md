Added a characterisation test suite for the scale-fixing gauge
(`tests/test_scale_gauge_vs_palmer.py`). The solver renormalises the columns of
`A` each iteration; Palmer's AMICA technical report §II.A specifies unit-norm
rows of `W`. The two are related by an exact diagonal rescaling and the
likelihood is unchanged, so no released behaviour changes here — the tests only
pin down the current convention and measure how far it sits from Palmer's, so
that a future switch can be shown not to move any published number.
