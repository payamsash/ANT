"""Scale-free standardisation helpers.

Neurofeedback features span an enormous range of magnitudes: EEG band power is
~1e-12 V²/Hz, MEG band power ~1e-24 T²/Hz, a connectivity index is ~1.  Any
*absolute* constant used to guard a division is therefore wrong for most of
them — a floor of ``1e-6`` does not protect a band-power standard deviation of
1e-14, it **replaces** it, and the resulting "z-score" is eight orders of
magnitude too small.

The guard these constants were reaching for is not "keep the z-score bounded".
It is "do not divide by zero", and the only input that actually divides by zero
is an exactly-constant signal — which genuinely *has* no z-score.  So the rule
here is a **degeneracy predicate** rather than a floor: when the spread carries
no information, the answer is ``0.0`` (the true z of a value sitting at the
mean of a distribution with no spread), and otherwise the observed spread is
used unmodified, whatever its magnitude.

The predicate is *relative*, because ``std > 0`` alone is not enough: with a
mean of 3.2e-13 and a standard deviation of 5e-28, ``value - mean`` is pure
float64 rounding noise and the quotient is meaningless. This mirrors the
relative-tolerance pattern already used in
:func:`~mne_rt.tools.asr.ASRDenoiser` (regularisation scaled by the covariance
trace) and :mod:`~mne_rt.tools.maxwell` (a singular-value cutoff at
``sv[0] * 1e-6``).

This module deliberately imports **only the standard library** — no numpy, no
mne — so that :mod:`~mne_rt.combiners` and :mod:`~mne_rt.protocols` can use it
without pulling in the heavy stack.  Callers holding numpy scalars should pass
them through :class:`float` first, as they already do.

All arithmetic here assumes float64.  ``DEGENERATE_REL_TOL`` sits below float32
resolution, so a float32 feature would defeat the cancellation guard.

.. versionadded:: 1.2.0
"""

import math

DEGENERATE_REL_TOL: float = 1e-12
"""Relative tolerance below which a spread is treated as carrying no information.

Roughly four decades above float64 epsilon (2.2e-16), so that when the
predicate only just passes, ``value - center`` still retains about four
significant digits.
"""


def usable_std(std, mean=0.0, *, min_std=None, rel_tol=DEGENERATE_REL_TOL) -> float:
    """Return ``std`` when it carries information about the data, else ``0.0``.

    Parameters
    ----------
    std : float
        The observed standard deviation.
    mean : float, default 0.0
        The centre of the distribution, used only as a magnitude reference for
        the degeneracy test.  Pass it whenever it is known: for an offset
        feature such as band power it *is* the scale, and omitting it disables
        the cancellation guard.
    min_std : float | None, default None
        Explicit absolute floor.  When given, the result is simply
        ``max(std, min_std)`` and no other rule applies — this reproduces the
        pre-1.2.0 behaviour exactly, for callers who deliberately chose a floor
        in the native units of their own feature.  When ``None`` the
        scale-free predicate is used instead.
    rel_tol : float, default DEGENERATE_REL_TOL
        Relative tolerance for the degeneracy test.

    Returns
    -------
    std : float
        ``std`` if it is usable, otherwise ``0.0``.  A ``0.0`` return is the
        caller's signal that no standardisation is possible for this sample.

    Notes
    -----
    The scale reference is ``hypot(mean, std)``, which behaves correctly at both
    extremes: for a centred feature (``mean`` ≈ 0, e.g. ``laterality``) it
    reduces to ``std``, so nothing is ever spuriously called degenerate; for an
    offset feature (``mean`` >> ``std``, e.g. band power) it reduces to
    ``abs(mean)`` and catches catastrophic cancellation.

    Examples
    --------
    A realistic band-power spread is kept, not floored:

    >>> usable_std(7.9e-15, 3.2e-13)
    7.9e-15

    A constant signal has no usable spread:

    >>> usable_std(0.0, 3.2e-13)
    0.0
    """
    std = float(std)
    # Finiteness first, so an explicit floor cannot pass a NaN through into a
    # reward magnitude and out over OSC/LSL.
    if not math.isfinite(std):
        return 0.0
    if min_std is not None:
        return max(std, float(min_std))
    if std <= 0.0:
        return 0.0
    mean = float(mean)
    if not math.isfinite(mean):
        return 0.0
    return std if std > rel_tol * math.hypot(mean, std) else 0.0


def zscore(value, center, std, *, min_std=None, rel_tol=DEGENERATE_REL_TOL) -> float:
    """Standardise ``value`` about ``center``, or return ``0.0`` if impossible.

    Parameters
    ----------
    value : float
        The value to standardise.
    center : float
        The distribution mean — or, at the "how far past the threshold is this,
        in units of sigma" call sites, the threshold itself, which is a valid
        magnitude reference because it lives inside the data range.
    std : float
        The observed standard deviation.
    min_std : float | None, default None
        See :func:`usable_std`.
    rel_tol : float, default DEGENERATE_REL_TOL
        See :func:`usable_std`.

    Returns
    -------
    z : float
        ``(value - center) / std``, or ``0.0`` when ``std`` is degenerate.

    Notes
    -----
    ``0.0`` is the neutral answer wherever this flows: no threshold crossing, a
    reward magnitude of zero, and no contribution to a combined feature.
    """
    std = usable_std(std, center, min_std=min_std, rel_tol=rel_tol)
    if std == 0.0:
        return 0.0
    return (float(value) - float(center)) / std


def welford_std(m2, n, *, ddof=1) -> float:
    """Standard deviation from a Welford ``M2`` accumulator.

    Parameters
    ----------
    m2 : float
        The accumulated sum of squared deviations.
    n : int
        Number of samples seen.
    ddof : int, default 1
        Delta degrees of freedom.  The default of 1 (sample standard deviation)
        is the right convention throughout this library: these statistics are
        estimated from a finite baseline and used to standardise future, unseen
        values.

    Returns
    -------
    std : float
        ``sqrt(m2 / (n - ddof))``, or ``0.0`` when there are too few samples or
        the accumulator has gone negative through rounding.
    """
    if n - ddof <= 0:
        return 0.0
    variance = float(m2) / (n - ddof)
    return math.sqrt(variance) if variance > 0.0 else 0.0


def ema_variance(var, delta, alpha) -> float:
    """One exponentially-weighted variance update.

    Parameters
    ----------
    var : float
        The current variance estimate.
    delta : float
        ``value - mean`` using the mean from **before** this sample is folded
        in.  The same ``delta`` must then be used to advance the mean, so that
        mean and variance stay on the same step.
    alpha : float
        Update rate in [0, 1).

    Returns
    -------
    var : float
        The updated variance estimate.

    Notes
    -----
    An exponentially weighted mean of squared deviations,
    ``var + a * (delta**2 - var)``, whose fixed point is the true variance.
    The related form ``(1 - a) * (var + a * delta**2)`` converges to
    ``(1 - a)`` times the variance instead, which inflates z-scores by
    ``1 / sqrt(1 - a)`` — 2.6 % at ``a = 0.05`` but 41 % at ``a = 0.5``.

    It tracks a *variance*; a previous implementation exponentially averaged
    ``abs(delta)`` instead, which is the mean absolute deviation — about
    0.798 sigma for Gaussian input, so z-scores came out ~25 % too large.
    """
    alpha = float(alpha)
    delta = float(delta)
    return float(var) + alpha * (delta * delta - float(var))
