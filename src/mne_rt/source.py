"""Source-space modelling: ROIs, head models, and cached linear operators.

This module turns a sensor-space data window into **ROI time courses** in one
matrix multiplication.

The naive route — :func:`~mne.beamformer.apply_lcmv_raw` followed by
:func:`~mne.extract_label_time_course` — is far too slow for a closed loop: on a
5 mm whole-brain volume source space (14 629 points) the label extraction alone
costs 650–800 ms per window, against a typical 500 ms hop. But with
``pick_ori="max-power"`` the beamformer is a fixed linear map, and
``mode="mean"`` / ``"mean_flip"`` label extraction is a fixed sparse average, so
the whole chain collapses to a single ``(n_roi, n_channels)`` matrix that can be
built once and applied with a ``~1 ms`` matmul::

    R = A @ W @ whitener        # built once, in _prep
    roi_tc = R @ data           # per window

:meth:`SourceModel.roi_kernel` builds ``R``; the result is numerically identical
to the MNE path (verified to ~1e-15 relative error in the test suite).

Regions of interest
-------------------
ROIs are named groups of atlas labels.  A single volumetric atlas —
FreeSurfer's ``aparc+aseg`` — carries both cortical parcels
(``ctx-lh-parsopercularis``) and subcortical structures (``Left-Hippocampus``),
so cortical and subcortical ROIs can live in one volume source space without
needing a mixed (surface + volume) model.

:data:`ROI_ALIASES` gives friendly names for a few common language ROIs; any
atlas label name is also accepted directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union
from warnings import warn

import numpy as np
from mne import read_labels_from_annot
from mne.label import label_sign_flip

from mne_rt._logging import logger

__all__ = [
    "ROI",
    "ROI_ALIASES",
    "SourceModel",
    "list_rois",
    "resolve_rois",
]


# ---------------------------------------------------------------------------
# ROI aliases
# ---------------------------------------------------------------------------

#: Friendly ROI names → the atlas labels they are composed of.
#:
#: These are *conventions*, not settled anatomy — the boundaries of "Broca's
#: area" in particular are contested.  Pass explicit atlas label names instead
#: if your protocol defines them differently.
ROI_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "aparc+aseg": {
        # Broca ≈ BA44 + BA45 (pars opercularis + pars triangularis)
        "Broca": ("ctx-lh-parsopercularis", "ctx-lh-parstriangularis"),
        "Broca-rh": ("ctx-rh-parsopercularis", "ctx-rh-parstriangularis"),
        # Wernicke ≈ posterior superior temporal / supramarginal
        "Wernicke": ("ctx-lh-superiortemporal", "ctx-lh-supramarginal"),
        "Wernicke-rh": ("ctx-rh-superiortemporal", "ctx-rh-supramarginal"),
        "Hippocampus-lh": ("Left-Hippocampus",),
        "Hippocampus-rh": ("Right-Hippocampus",),
    },
    "aparc": {
        "Broca": ("parsopercularis-lh", "parstriangularis-lh"),
        "Broca-rh": ("parsopercularis-rh", "parstriangularis-rh"),
        "Wernicke": ("superiortemporal-lh", "supramarginal-lh"),
        "Wernicke-rh": ("superiortemporal-rh", "supramarginal-rh"),
    },
}


@dataclass(frozen=True)
class ROI:
    """A named region of interest, possibly spanning several atlas labels.

    Attributes
    ----------
    name : str
        User-facing name, e.g. ``"Broca"``.
    members : tuple of str
        Atlas label names merged into this ROI.
    kind : {"volume", "surface"}
        Which kind of source space the labels are defined on.
    """

    name: str
    members: tuple[str, ...]
    kind: str

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<ROI {self.name!r} ({self.kind}, {len(self.members)} label(s))>"


# ---------------------------------------------------------------------------
# Atlas helpers
# ---------------------------------------------------------------------------


def _is_volume_atlas(atlas: str) -> bool:
    """Volumetric atlases are ``.mgz`` volumes; surface ones are annot names."""
    return str(atlas).endswith((".mgz", ".mgh")) or "aseg" in str(atlas)


def _volume_atlas_path(atlas: str, subject: str, subjects_dir) -> Path:
    """Resolve a volumetric atlas name to a file under ``<subject>/mri/``."""
    atlas = str(atlas)
    if atlas.endswith((".mgz", ".mgh")) and Path(atlas).is_file():
        return Path(atlas)
    stem = atlas[:-4] if atlas.endswith((".mgz", ".mgh")) else atlas
    if subjects_dir is None:
        raise ValueError(
            f"`subjects_dir` is required to locate the volumetric atlas {atlas!r}. "
            "Pass subjects_fs_dir=... (or an absolute path to the .mgz)."
        )
    path = Path(subjects_dir) / subject / "mri" / f"{stem}.mgz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Volumetric atlas {stem!r} not found for subject {subject!r}: {path} "
            "does not exist. FreeSurfer ships 'aseg' and 'aparc+aseg'."
        )
    return path


def list_rois(*, atlas: str, subject: str = "fsaverage", subjects_dir=None) -> list[str]:
    """List every label name available in an atlas, plus any aliases.

    Parameters
    ----------
    atlas : str
        Volumetric atlas stem (e.g. ``"aparc+aseg"``) or surface annotation
        name (e.g. ``"aparc"``).
    subject : str
        FreeSurfer subject identifier.
    subjects_dir : path-like | None
        FreeSurfer subjects directory.

    Returns
    -------
    names : list of str
        Sorted label names.  Alias names from :data:`ROI_ALIASES` are listed
        first so they are easy to spot.
    """
    if _is_volume_atlas(atlas):
        import mne

        path = _volume_atlas_path(atlas, subject, subjects_dir)
        names = list(mne.get_volume_labels_from_aseg(str(path)))
    else:
        names = [
            label.name
            for label in read_labels_from_annot(
                subject=subject, parc=atlas, subjects_dir=subjects_dir, verbose=False
            )
        ]
    aliases = sorted(ROI_ALIASES.get(str(atlas), {}))
    return aliases + sorted(names)


def resolve_rois(
    names: Union[str, Sequence],
    *,
    atlas: str,
    subject: str = "fsaverage",
    subjects_dir=None,
) -> list[ROI]:
    """Resolve ROI names (or aliases, or explicit label groups) into :class:`ROI` objects.

    Parameters
    ----------
    names : str | sequence
        Each entry may be:

        * an alias from :data:`ROI_ALIASES` (e.g. ``"Broca"``),
        * a bare atlas label name (e.g. ``"Left-Hippocampus"``),
        * a ``{name: [label, ...]}`` mapping, or a ``(name, [label, ...])``
          pair, to define an ROI explicitly.
    atlas : str
        Volumetric atlas stem or surface annotation name.
    subject : str
        FreeSurfer subject identifier.
    subjects_dir : path-like | None
        FreeSurfer subjects directory.

    Returns
    -------
    rois : list of ROI

    Raises
    ------
    ValueError
        If a label name is not present in the atlas.  The message lists close
        matches to make typos easy to spot.
    """
    if isinstance(names, (str, dict)):
        names = [names]

    kind = "volume" if _is_volume_atlas(atlas) else "surface"
    available = set(list_rois(atlas=atlas, subject=subject, subjects_dir=subjects_dir))
    alias_table = ROI_ALIASES.get(str(atlas), {})

    rois: list[ROI] = []
    for entry in names:
        if isinstance(entry, dict):
            items = list(entry.items())
        elif isinstance(entry, (tuple, list)) and len(entry) == 2 and not isinstance(entry[1], str):
            items = [(entry[0], entry[1])]
        else:
            name = str(entry)
            items = [(name, alias_table.get(name, (name,)))]

        for roi_name, members in items:
            members = (members,) if isinstance(members, str) else tuple(members)
            missing = [m for m in members if m not in available]
            if missing:
                hints = [
                    a for a in sorted(available) if any(m.lower() in a.lower() for m in missing)
                ]
                raise ValueError(
                    f"ROI {roi_name!r}: label(s) {missing} not found in atlas {atlas!r} "
                    f"for subject {subject!r}."
                    + (f" Did you mean any of {hints[:5]}?" if hints else "")
                    + " Use mne_rt.source.list_rois() to see what is available."
                )
            rois.append(ROI(name=str(roi_name), members=members, kind=kind))

    seen = [r.name for r in rois]
    duplicates = {n for n in seen if seen.count(n) > 1}
    if duplicates:
        raise ValueError(f"Duplicate ROI name(s): {sorted(duplicates)}.")
    return rois


# ---------------------------------------------------------------------------
# SourceModel
# ---------------------------------------------------------------------------


class SourceModel:
    """Sensor data → ROI time courses, via one cached linear operator.

    Parameters
    ----------
    src : instance of SourceSpaces
        The source space the inverse/beamformer was built on.  Needed to map
        atlas labels onto source-estimate rows.
    atlas : str
        Volumetric atlas stem (``"aparc+aseg"``) or surface annotation name
        (``"aparc"``).
    subject : str
        FreeSurfer subject identifier.
    subjects_dir : path-like | None
        FreeSurfer subjects directory.
    filters : instance of Beamformer | None
        Fitted LCMV spatial filter.  Required for the fast kernel path.
    inverse : instance of InverseOperator | None
        Minimum-norm inverse operator.  Used by the slow fallback path.
    inverse_method : str
        Minimum-norm method name for the fallback path.

    Notes
    -----
    The fast path requires a beamformer with a *fixed* orientation
    (``pick_ori="max-power"``).  A free-orientation solution combines three
    orientations by norm, which is non-linear and destroys phase — meaningless
    for imaginary coherence — so it is refused rather than silently accepted.
    """

    def __init__(
        self,
        *,
        src,
        atlas: str,
        subject: str = "fsaverage",
        subjects_dir=None,
        filters=None,
        inverse=None,
        inverse_method: str = "dSPM",
    ) -> None:
        self.src = src
        self.atlas = atlas
        self.subject = subject
        self.subjects_dir = subjects_dir
        self.filters = filters
        self.inverse = inverse
        self.inverse_method = inverse_method

        if filters is None and inverse is None:
            raise ValueError("SourceModel needs either `filters` (LCMV) or `inverse` (min-norm).")

    # -- introspection --------------------------------------------------

    @property
    def kind(self) -> str:
        """``"volume"`` or ``"surface"``, taken from the source space."""
        return "volume" if self.src.kind == "volume" else "surface"

    @property
    def n_sources(self) -> int:
        return int(sum(s["nuse"] for s in self.src))

    @property
    def supports_kernel(self) -> bool:
        """Whether the fast cached-kernel path is available."""
        return self.filters is not None and not self.filters["is_free_ori"]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        how = "LCMV" if self.filters is not None else self.inverse_method
        fast = "kernel" if self.supports_kernel else "fallback"
        return f"<SourceModel | {self.kind}, {self.n_sources} sources, {how}, {fast}>"

    # -- operators ------------------------------------------------------

    def source_operator(self) -> np.ndarray:
        """Linear map from sensor data to source time courses, shape ``(n_src, n_ch)``.

        Mirrors what :func:`~mne.beamformer.apply_lcmv_raw` does internally:
        whiten (or project) the data, then apply the beamformer weights.
        """
        if not self.supports_kernel:
            raise RuntimeError(
                "A cached source operator needs an LCMV beamformer with a fixed "
                "orientation (pick_ori='max-power'). Free-orientation and "
                "minimum-norm solutions must use the fallback path."
            )
        weights = self.filters["weights"]
        whitener = self.filters.get("whitener")
        if whitener is not None:
            return weights @ whitener
        return weights @ self.filters["proj"]

    def label_operator(self, rois: Sequence[ROI], *, mri_resolution: bool = True) -> np.ndarray:
        """Sparse-average matrix mapping source points to ROIs, shape ``(n_roi, n_src)``.

        Equivalent to :func:`~mne.extract_label_time_course` with
        ``mode="mean"`` (volume) or ``"mean_flip"`` (surface), but built once
        rather than recomputed every window.
        """
        if self.kind == "volume":
            return self._volume_label_operator(rois, mri_resolution=mri_resolution)
        return self._surface_label_operator(rois)

    def roi_kernel(self, rois: Sequence[ROI], *, mri_resolution: bool = True) -> np.ndarray:
        """The full sensor → ROI operator, shape ``(n_roi, n_channels)``.

        Apply it to a data window with :meth:`apply`, or simply ``kernel @ data``.
        """
        return self.label_operator(rois, mri_resolution=mri_resolution) @ self.source_operator()

    # -- application ----------------------------------------------------

    def apply(
        self,
        data: np.ndarray,
        *,
        kernel=None,
        label_operator=None,
        info=None,
        pick_ori: Optional[str] = None,
    ) -> np.ndarray:
        """Return ROI time courses ``(n_roi, n_times)`` for one data window.

        Uses ``kernel`` when given — one matmul, and the whole point of this
        class.  Otherwise applies the source estimate the slow way and reduces
        it with ``label_operator``; correct, but ~two orders of magnitude
        slower and unsuitable for a closed loop.

        Note the label reduction is the *same* matrix either way, so the two
        paths agree exactly; only the source step differs.
        """
        if kernel is not None:
            return kernel @ data
        if label_operator is None:
            raise ValueError("Provide either `kernel` (fast path) or `label_operator`.")
        return label_operator @ self.source_estimate(data, info=info, pick_ori=pick_ori)

    def source_estimate(self, data: np.ndarray, *, info=None, pick_ori=None) -> np.ndarray:
        """Source time courses ``(n_src, n_times)`` via the full MNE route."""
        from mne.beamformer import apply_lcmv_raw
        from mne.io import RawArray
        from mne.minimum_norm import apply_inverse_raw

        if info is None:
            raise ValueError("The fallback path needs `info` to wrap the window as a Raw.")
        raw = RawArray(data, info, verbose=False)
        if self.filters is not None:
            stc = apply_lcmv_raw(raw, self.filters, verbose=False)
        else:
            stc = apply_inverse_raw(
                raw,
                self.inverse,
                lambda2=1.0 / 9,
                method=self.inverse_method,
                pick_ori=pick_ori,
                verbose=False,
            )
        return stc.data

    # -- internals ------------------------------------------------------

    def _volume_label_operator(self, rois, *, mri_resolution: bool) -> np.ndarray:
        try:
            from mne.source_estimate import _volume_labels
        except ImportError as exc:  # pragma: no cover - depends on MNE internals
            raise RuntimeError(
                "This MNE version does not expose mne.source_estimate._volume_labels, "
                "which the cached-kernel path relies on. Use the fallback path."
            ) from exc

        members = [m for roi in rois for m in roi.members]
        path = _volume_atlas_path(self.atlas, self.subject, self.subjects_dir)
        labels = _volume_labels(self.src, (str(path), members), mri_resolution=mri_resolution)

        vertno = np.concatenate([s["vertno"] for s in self.src])
        position = {int(v): i for i, v in enumerate(vertno)}
        n_src = self.n_sources

        # One row per atlas label, then merge the labels making up each ROI.
        rows: dict[str, np.ndarray] = {}
        counts: dict[str, int] = {}
        for label in labels:
            if isinstance(label, dict):
                # mri_resolution=True: csr maps source points -> the label's MRI
                # voxels, so the label mean is the column mean of that matrix.
                name = label["name"]
                rows[name] = np.asarray(label["csr"].mean(axis=0)).ravel()
                counts[name] = int(label["csr"].shape[0])
            else:
                name = label.name
                idx = [position[int(v)] for v in np.asarray(label.vertices) if int(v) in position]
                row = np.zeros(n_src)
                if idx:
                    row[idx] = 1.0 / len(idx)
                rows[name] = row
                counts[name] = len(idx)

        return self._merge_rows(rois, rows, counts, n_src)

    def _surface_label_operator(self, rois) -> np.ndarray:
        labels = read_labels_from_annot(
            subject=self.subject, parc=self.atlas, subjects_dir=self.subjects_dir, verbose=False
        )
        by_name = {label.name: label for label in labels}

        # Row offset of each hemisphere within the stacked source estimate.
        offsets, running = [], 0
        for s in self.src:
            offsets.append(running)
            running += s["nuse"]
        n_src = running

        rows: dict[str, np.ndarray] = {}
        counts: dict[str, int] = {}
        for name in {m for roi in rois for m in roi.members}:
            label = by_name[name]
            hemi = 0 if label.hemi == "lh" else 1
            vertno = self.src[hemi]["vertno"]
            _, idx, _ = np.intersect1d(vertno, label.vertices, return_indices=True)
            row = np.zeros(n_src)
            if len(idx):
                # mean_flip: sign-align vertices before averaging, so that
                # opposing dipole orientations do not cancel.
                flip = np.asarray(label_sign_flip(label, self.src)).ravel()
                flip = flip[: len(idx)] if len(flip) >= len(idx) else np.ones(len(idx))
                row[offsets[hemi] + idx] = flip / len(idx)
            rows[name] = row
            counts[name] = len(idx)

        return self._merge_rows(rois, rows, counts, n_src)

    @staticmethod
    def _merge_rows(rois, rows, counts, n_src: int) -> np.ndarray:
        """Combine per-label rows into one row per ROI, weighted by label size."""
        operator = np.zeros((len(rois), n_src))
        for i, roi in enumerate(rois):
            total = sum(counts.get(m, 0) for m in roi.members)
            if total == 0:
                warn(
                    f"ROI {roi.name!r} contains no source points; its time course "
                    "will be all zeros. The ROI may be too small for the source-space "
                    "spacing, or outside the head model.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            # Weight each member by how many source points it contributes, so a
            # multi-label ROI equals the mean over all its source points.
            for member in roi.members:
                if counts.get(member, 0):
                    operator[i] += rows[member] * (counts[member] / total)
        return operator

    # -- construction ---------------------------------------------------

    @classmethod
    def from_stream(
        cls,
        rt,
        *,
        atlas: Optional[str] = None,
        method: str = "LCMV",
        reg: float = 0.05,
        pick_ori: str = "max-power",
        weight_norm: str = "unit-noise-gain",
    ) -> "SourceModel":
        """Build a :class:`SourceModel` from a session that has recorded a baseline.

        Parameters
        ----------
        rt : instance of RTStream
            Session with ``src``/``fwd``/``data_cov`` set — i.e. one on which
            :meth:`~mne_rt.RTStream.record_baseline` has been run.
        atlas : str | None
            Overrides the session's ``source_atlas``.
        method : str
            ``"LCMV"`` for the beamformer, or a minimum-norm method name.
        reg, pick_ori, weight_norm
            Passed to :func:`~mne.beamformer.make_lcmv`.
        """
        from mne.beamformer import make_lcmv

        src = getattr(rt, "src", None)
        if src is None:
            raise RuntimeError(
                "No source space on the session. Run record_baseline() (which calls "
                "compute_inv_operator()) before building a SourceModel."
            )
        atlas = atlas if atlas is not None else getattr(rt, "source_atlas", "aparc")

        if method == "LCMV":
            data_cov = getattr(rt, "data_cov", None)
            if data_cov is None:
                raise RuntimeError(
                    "An LCMV beamformer needs a data covariance. Run record_baseline() "
                    "first, or pass data_cov=... to compute_inv_operator()."
                )
            filters = make_lcmv(
                rt.rec_info,
                rt.fwd,
                data_cov,  # data covariance, NOT the noise covariance
                reg=reg,
                noise_cov=getattr(rt, "noise_cov", None),
                pick_ori=pick_ori,
                weight_norm=weight_norm,
                rank=None,
                verbose=False,
            )
            logger.info("SourceModel: built LCMV filters (pick_ori=%s).", pick_ori)
            return cls(
                src=src,
                atlas=atlas,
                subject=rt.subject_fs_id,
                subjects_dir=rt.subjects_fs_dir,
                filters=filters,
            )

        return cls(
            src=src,
            atlas=atlas,
            subject=rt.subject_fs_id,
            subjects_dir=rt.subjects_fs_dir,
            inverse=rt.inv,
            inverse_method=method,
        )
