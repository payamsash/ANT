"""Instanced modality names — ``base@label``.

A neurofeedback session often needs the *same* measure computed more than once
with different parameters: imaginary coherence at theta **and** at alpha, say,
each driving its own protocol.  :meth:`~mne_rt.RTStream.record_main` addresses
those with an **instance label** appended to the modality name::

    modality=["source_connectivity@theta", "source_connectivity@alpha"]

The part before the separator is the *base* modality — it selects the entry in
``config_methods.yml`` and the ``_<base>`` / ``_<base>_prep`` methods on
:class:`~mne_rt.modalities.ModalityMixin`.  The full name is what keys every
piece of per-instance state (values, smoothing, z-score buffers, protocols) and
every output channel (plot trace, OSC address, LSL channel, saved column).

A plain name without a separator is simply an instance whose label is ``""``,
so nothing about existing sessions changes.

This module deliberately imports **only the standard library**: it is shared by
``tools.tools`` (which pulls in mne and matplotlib) and ``viz.nf_plot`` (which
pulls in Qt and must not import either), so it cannot live in either one.
"""

import re
from dataclasses import dataclass

MODALITY_SEP = "@"
"""Separator between a base modality and its instance label."""

OSC_SEP_REPLACEMENT = "_"
"""What :func:`osc_address_name` maps :data:`MODALITY_SEP` to.

``"_"`` rather than ``"/"`` on purpose: OSC wildcards do not match across a
``/``, so a subscriber listening on ``/ant/*`` would silently stop receiving
instanced modalities if the label became its own address level.
"""

# Kept deliberately narrow: the label travels into OSC addresses, LSL channel
# names and TSV headers, so anything outside this set would need a different
# escape in each of them.
_LABEL_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# Everything OSC reserves as a pattern metacharacter, plus whitespace.
_OSC_UNSAFE_RE = re.compile(r"[^A-Za-z0-9_.-]")


@dataclass(frozen=True)
class ModalitySpec:
    """One active modality instance.

    Attributes
    ----------
    name : str
        The full name as the user wrote it, e.g. ``"source_connectivity@theta"``.
        This is the key for per-instance state and for every output channel.
    base : str
        The base modality, e.g. ``"source_connectivity"``.  Selects the config
        entry and the compute/prep methods.
    label : str
        The instance label, e.g. ``"theta"``; ``""`` for a plain modality.
    """

    name: str
    base: str
    label: str

    @property
    def is_instanced(self) -> bool:
        """Whether this name carries an instance label."""
        return bool(self.label)


def split_modality(name: str) -> tuple[str, str]:
    """Split ``name`` into ``(base, label)`` without validating.

    Cheap and total — it never raises, so it is safe on the plotting path where
    a name has already been validated upstream and a failure would kill the
    display.  Use :func:`parse_modality` when the name comes straight from a
    user.

    Parameters
    ----------
    name : str
        A modality name, instanced or plain.

    Returns
    -------
    base : str
        The part before the first separator, or the whole name.
    label : str
        The part after the first separator, or ``""``.
    """
    base, sep, label = str(name).partition(MODALITY_SEP)
    return (base, label) if sep else (base, "")


def parse_modality(name: str) -> ModalitySpec:
    """Validate ``name`` and split it into a :class:`ModalitySpec`.

    Parameters
    ----------
    name : str
        A modality name, either plain (``"sensor_power"``) or instanced
        (``"sensor_power@alpha"``).

    Returns
    -------
    spec : ModalitySpec
        The parsed name.  The base is *not* checked against the config here —
        that happens in :func:`~mne_rt.tools.tools.get_params`, which owns the
        list of known modalities.

    Raises
    ------
    ValueError
        If the name is not a string, is empty, carries more than one separator,
        or has an empty or non-alphanumeric label.
    """
    if not isinstance(name, str):
        raise ValueError(f"Modality names must be strings; got {type(name).__name__}: {name!r}.")

    if name.count(MODALITY_SEP) > 1:
        raise ValueError(
            f"Modality name {name!r} contains more than one {MODALITY_SEP!r}. "
            f"The form is 'base{MODALITY_SEP}label', e.g. 'source_connectivity{MODALITY_SEP}theta'."
        )

    base, label = split_modality(name)

    if not base:
        raise ValueError(
            f"Modality name {name!r} has an empty base modality. "
            f"The form is 'base{MODALITY_SEP}label', e.g. 'source_connectivity{MODALITY_SEP}theta'."
        )
    if MODALITY_SEP in name and not label:
        raise ValueError(
            f"Modality name {name!r} has an empty instance label. Either drop the "
            f"{MODALITY_SEP!r} or name the instance, e.g. {base + MODALITY_SEP + 'theta'!r}."
        )
    if label and not _LABEL_RE.match(label):
        raise ValueError(
            f"Instance label {label!r} in {name!r} may only contain letters, digits, "
            "'_' and '-'. The label is used as an OSC address, an LSL channel name and "
            "a column header, so it has to survive all three."
        )

    return ModalitySpec(name=name, base=base, label=label)


def resolve_by_base(mapping: dict, name: str, default=None):
    """Look ``name`` up in ``mapping``, falling back to its base modality.

    Lets per-modality tables — display scales, axis labels, units — keep their
    entries keyed by base modality while still serving instanced names, so an
    instance inherits its base's presentation unless it is given its own entry.

    Parameters
    ----------
    mapping : dict
        Table keyed by modality name and/or base modality.
    name : str
        The name to resolve.
    default : object
        Returned when neither the full name nor the base is present.

    Returns
    -------
    value : object
        ``mapping[name]`` if present, else ``mapping[base]`` if present, else
        ``default``.

    Notes
    -----
    Membership is tested explicitly rather than with ``mapping.get(...) or``,
    so a legitimately falsy entry (a ``0.0`` display scale) is returned as-is
    instead of falling through to the default.
    """
    if name in mapping:
        return mapping[name]
    base, _ = split_modality(name)
    if base in mapping:
        return mapping[base]
    return default


def osc_address_name(name: str) -> str:
    """Convert a modality name into an OSC-address-safe token.

    The separator becomes :data:`OSC_SEP_REPLACEMENT`, and any character OSC
    reserves for pattern matching (``# * , ? [ ] { }``, whitespace) is replaced
    with ``"_"``.  The result is the identity for every built-in modality name,
    so existing OSC subscriptions are unaffected.

    Parameters
    ----------
    name : str
        A modality name.

    Returns
    -------
    token : str
        The sanitised single-level address component.

    Examples
    --------
    >>> osc_address_name("sensor_power")
    'sensor_power'
    >>> osc_address_name("source_connectivity@theta")
    'source_connectivity_theta'
    """
    return _OSC_UNSAFE_RE.sub("_", str(name).replace(MODALITY_SEP, OSC_SEP_REPLACEMENT))
