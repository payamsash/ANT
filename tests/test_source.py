"""Tests for mne_rt.source — ROI resolution and cached source-space operators.

Split into two tiers:

* **Tier 1** — pure linear algebra against synthetic objects. No FreeSurfer
  anatomy, no downloads, runs everywhere in milliseconds.
* **Tier 2** — marked ``slow``; needs the fsaverage anatomy that
  :func:`mne.datasets.fetch_fsaverage` provides. These build a real volumetric
  forward model once per session (~6 s) and pin the cached kernel against the
  full MNE route.
"""

import numpy as np
import pytest

from mne_rt.source import (
    ROI,
    ROI_ALIASES,
    SourceModel,
    _is_volume_atlas,
    _volume_atlas_path,
    resolve_rois,
)

# ===================================================================
# Tier 1 — no anatomy required
# ===================================================================


class _FakeSrc(list):
    """Minimal stand-in for mne.SourceSpaces."""

    def __init__(self, vertnos, kind="volume"):
        super().__init__({"vertno": np.asarray(v), "nuse": len(v)} for v in vertnos)
        self.kind = kind


def _fake_filters(n_src, n_ch, *, free_ori=False, seed=0, whitener=True):
    rng = np.random.default_rng(seed)
    filters = {
        "weights": rng.standard_normal((n_src, n_ch)),
        "is_free_ori": free_ori,
    }
    if whitener:
        filters["whitener"] = rng.standard_normal((n_ch, n_ch))
    else:
        filters["whitener"] = None
        filters["proj"] = rng.standard_normal((n_ch, n_ch))
    return filters


# ------------------------------------------------------------------
# atlas helpers
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "atlas,expected",
    [
        ("aparc+aseg", True),
        ("aseg", True),
        ("/path/to/aparc+aseg.mgz", True),
        ("something.mgh", True),
        ("aparc", False),
        ("aparc.a2009s", False),
        ("HCPMMP1", False),
    ],
)
def test_is_volume_atlas(atlas, expected):
    assert _is_volume_atlas(atlas) is expected


def test_volume_atlas_path_requires_subjects_dir():
    with pytest.raises(ValueError, match="subjects_dir"):
        _volume_atlas_path("aparc+aseg", "fsaverage", None)


def test_volume_atlas_path_missing_file(tmp_path):
    (tmp_path / "fsaverage" / "mri").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="aparc%2Baseg|aparc\\+aseg"):
        _volume_atlas_path("aparc+aseg", "fsaverage", tmp_path)


def test_volume_atlas_path_found(tmp_path):
    mri = tmp_path / "fsaverage" / "mri"
    mri.mkdir(parents=True)
    (mri / "aparc+aseg.mgz").touch()
    assert _volume_atlas_path("aparc+aseg", "fsaverage", tmp_path) == mri / "aparc+aseg.mgz"


# ------------------------------------------------------------------
# ROI resolution
# ------------------------------------------------------------------


@pytest.fixture()
def fake_atlas(monkeypatch):
    """Pretend an atlas with a handful of labels exists."""
    labels = [
        "ctx-lh-parsopercularis",
        "ctx-lh-parstriangularis",
        "ctx-lh-superiortemporal",
        "ctx-lh-supramarginal",
        "Left-Hippocampus",
        "Right-Hippocampus",
    ]
    import mne_rt.source as source

    monkeypatch.setattr(source, "list_rois", lambda **kw: labels)
    return labels


def test_resolve_rois_alias_expands_to_members(fake_atlas):
    (roi,) = resolve_rois(["Broca"], atlas="aparc+aseg", subjects_dir="/x")
    assert roi.name == "Broca"
    assert roi.members == ROI_ALIASES["aparc+aseg"]["Broca"]
    assert roi.kind == "volume"


def test_resolve_rois_bare_label(fake_atlas):
    (roi,) = resolve_rois(["Left-Hippocampus"], atlas="aparc+aseg", subjects_dir="/x")
    assert roi.members == ("Left-Hippocampus",)


def test_resolve_rois_accepts_single_string(fake_atlas):
    rois = resolve_rois("Broca", atlas="aparc+aseg", subjects_dir="/x")
    assert len(rois) == 1


def test_resolve_rois_explicit_mapping(fake_atlas):
    (roi,) = resolve_rois(
        [{"MyROI": ["Left-Hippocampus", "Right-Hippocampus"]}],
        atlas="aparc+aseg",
        subjects_dir="/x",
    )
    assert roi.name == "MyROI"
    assert roi.members == ("Left-Hippocampus", "Right-Hippocampus")


def test_resolve_rois_unknown_label_raises_with_hint(fake_atlas):
    with pytest.raises(ValueError, match="Hippocampus"):
        resolve_rois(["Hippocampus"], atlas="aparc+aseg", subjects_dir="/x")


def test_resolve_rois_duplicate_names_rejected(fake_atlas):
    with pytest.raises(ValueError, match="Duplicate"):
        resolve_rois(["Broca", "Broca"], atlas="aparc+aseg", subjects_dir="/x")


def test_resolve_rois_surface_kind(monkeypatch):
    import mne_rt.source as source

    monkeypatch.setattr(source, "list_rois", lambda **kw: ["parsopercularis-lh"])
    (roi,) = resolve_rois(["parsopercularis-lh"], atlas="aparc", subjects_dir="/x")
    assert roi.kind == "surface"


# ------------------------------------------------------------------
# SourceModel — operators
# ------------------------------------------------------------------


def test_source_model_requires_an_inverse_or_filters():
    with pytest.raises(ValueError, match="filters.*inverse|inverse.*filters"):
        SourceModel(src=_FakeSrc([[0, 1]]), atlas="aparc+aseg")


def test_source_operator_equals_weights_times_whitener():
    src = _FakeSrc([np.arange(6)])
    filters = _fake_filters(6, 4)
    model = SourceModel(src=src, atlas="aparc+aseg", filters=filters)
    np.testing.assert_allclose(model.source_operator(), filters["weights"] @ filters["whitener"])


def test_source_operator_falls_back_to_proj_without_whitener():
    src = _FakeSrc([np.arange(6)])
    filters = _fake_filters(6, 4, whitener=False)
    model = SourceModel(src=src, atlas="aparc+aseg", filters=filters)
    np.testing.assert_allclose(model.source_operator(), filters["weights"] @ filters["proj"])


def test_free_orientation_refuses_the_kernel_path():
    """A norm-combined free-ori solution is non-linear and destroys phase."""
    src = _FakeSrc([np.arange(6)])
    model = SourceModel(src=src, atlas="aparc+aseg", filters=_fake_filters(6, 4, free_ori=True))
    assert model.supports_kernel is False
    with pytest.raises(RuntimeError, match="fixed"):
        model.source_operator()


def test_supports_kernel_false_for_minimum_norm():
    model = SourceModel(src=_FakeSrc([np.arange(6)]), atlas="aparc+aseg", inverse=object())
    assert model.supports_kernel is False


def test_n_sources_sums_over_source_spaces():
    model = SourceModel(
        src=_FakeSrc([np.arange(4), np.arange(6)]),
        atlas="aparc+aseg",
        filters=_fake_filters(10, 3),
    )
    assert model.n_sources == 10


# ------------------------------------------------------------------
# Label operator merging
# ------------------------------------------------------------------


def test_merge_rows_weights_members_by_size():
    """A 2-label ROI must equal the mean over all its source points.

    Label A has 3 points, label B has 1, so a plain average of the two label
    means would over-weight B four-fold.
    """
    n_src = 4
    rows = {
        "A": np.array([1 / 3, 1 / 3, 1 / 3, 0.0]),
        "B": np.array([0.0, 0.0, 0.0, 1.0]),
    }
    counts = {"A": 3, "B": 1}
    roi = ROI(name="AB", members=("A", "B"), kind="volume")
    op = SourceModel._merge_rows([roi], rows, counts, n_src)
    np.testing.assert_allclose(op[0], np.full(n_src, 0.25))


def test_merge_rows_single_member_is_unchanged():
    rows = {"A": np.array([0.5, 0.5, 0.0])}
    counts = {"A": 2}
    roi = ROI(name="A", members=("A",), kind="volume")
    op = SourceModel._merge_rows([roi], rows, counts, 3)
    np.testing.assert_allclose(op[0], rows["A"])


def test_merge_rows_empty_roi_warns_and_is_zero():
    roi = ROI(name="Tiny", members=("A",), kind="volume")
    with pytest.warns(RuntimeWarning, match="no source points"):
        op = SourceModel._merge_rows([roi], {"A": np.zeros(3)}, {"A": 0}, 3)
    np.testing.assert_allclose(op[0], np.zeros(3))


def test_roi_kernel_is_label_operator_times_source_operator(monkeypatch):
    src = _FakeSrc([np.arange(5)])
    filters = _fake_filters(5, 3)
    model = SourceModel(src=src, atlas="aparc+aseg", filters=filters)
    label_op = np.array([[0.5, 0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.5, 0.5]])
    monkeypatch.setattr(model, "label_operator", lambda *a, **k: label_op)
    np.testing.assert_allclose(model.roi_kernel([]), label_op @ model.source_operator())


def test_apply_uses_kernel_when_given():
    model = SourceModel(
        src=_FakeSrc([np.arange(4)]), atlas="aparc+aseg", filters=_fake_filters(4, 2)
    )
    kernel = np.array([[1.0, 0.0], [0.0, 1.0]])
    data = np.arange(8, dtype=float).reshape(2, 4)
    np.testing.assert_allclose(model.apply(data, kernel=kernel), data)


def test_apply_without_kernel_or_label_operator_raises():
    model = SourceModel(
        src=_FakeSrc([np.arange(4)]), atlas="aparc+aseg", filters=_fake_filters(4, 2)
    )
    with pytest.raises(ValueError, match="label_operator"):
        model.apply(np.zeros((2, 4)))


# ===================================================================
# Tier 2 — needs fsaverage anatomy
# ===================================================================


@pytest.fixture(scope="module")
def fsaverage_volume_model():
    """A real volumetric LCMV SourceModel on fsaverage (~6 s, built once).

    Module-scoped rather than session-scoped on purpose. Every consumer lives in
    this file, so it is still constructed exactly once — but a whole-brain 5 mm
    volume source space carries an interpolator with ~17 M non-zeros over the
    256³ MRI grid, and together with the forward solution and the beamformer
    this fixture retains well over 1 GB. Session scope would hold that for the
    rest of the run, including the Qt visualisation tests, which are already
    fragile under headless Xvfb.
    """
    mne = pytest.importorskip("mne")
    try:
        fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
    except Exception as exc:  # pragma: no cover - network/OSF dependent
        pytest.skip(f"fsaverage unavailable: {exc}")

    import os.path as op

    subjects_dir = op.dirname(fs_dir)
    sfreq = 250.0
    montage = mne.channels.make_standard_montage("standard_1020")
    ch_names = [c for c in montage.ch_names if c not in ("T3", "T4", "T5", "T6")][:32]
    info = mne.create_info(ch_names, sfreq, "eeg")
    info.set_montage(montage)

    rng = np.random.default_rng(0)
    raw = mne.io.RawArray(
        rng.standard_normal((len(ch_names), int(sfreq * 10))) * 1e-6, info, verbose=False
    )
    raw.set_eeg_reference("average", projection=True, verbose=False)

    src = mne.read_source_spaces(op.join(fs_dir, "bem", "fsaverage-vol-5-src.fif"), verbose=False)
    bem = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")
    fwd = mne.make_forward_solution(
        raw.info, trans="fsaverage", src=src, bem=bem, eeg=True, meg=False, verbose=False
    )
    data_cov = mne.compute_raw_covariance(raw, method="empirical", verbose=False)
    filters = mne.beamformer.make_lcmv(
        raw.info,
        fwd,
        data_cov,
        reg=0.05,
        pick_ori="max-power",
        weight_norm="unit-noise-gain",
        rank=None,
        verbose=False,
    )
    model = SourceModel(
        src=fwd["src"],
        atlas="aparc+aseg",
        subject="fsaverage",
        subjects_dir=subjects_dir,
        filters=filters,
    )
    return model, raw, subjects_dir


@pytest.mark.slow
def test_aparc_aseg_carries_cortical_and_subcortical(fsaverage_volume_model):
    """The premise of the single-volume design: one atlas covers both."""
    _, _, subjects_dir = fsaverage_volume_model
    from mne_rt.source import list_rois

    names = set(list_rois(atlas="aparc+aseg", subjects_dir=subjects_dir))
    assert "ctx-lh-parsopercularis" in names  # cortical
    assert "Left-Hippocampus" in names  # subcortical


@pytest.mark.slow
def test_resolve_language_rois_on_fsaverage(fsaverage_volume_model):
    _, _, subjects_dir = fsaverage_volume_model
    rois = resolve_rois(
        ["Broca", "Wernicke", "Hippocampus-lh"],
        atlas="aparc+aseg",
        subjects_dir=subjects_dir,
    )
    assert [r.name for r in rois] == ["Broca", "Wernicke", "Hippocampus-lh"]
    assert all(r.kind == "volume" for r in rois)


@pytest.mark.slow
def test_roi_kernel_shape_and_finiteness(fsaverage_volume_model):
    model, raw, subjects_dir = fsaverage_volume_model
    rois = resolve_rois(["Broca", "Hippocampus-lh"], atlas="aparc+aseg", subjects_dir=subjects_dir)
    kernel = model.roi_kernel(rois)
    assert kernel.shape == (2, len(raw.ch_names))
    assert np.all(np.isfinite(kernel))
    assert np.any(kernel != 0)


@pytest.mark.slow
@pytest.mark.parametrize("mri_resolution", [True, False])
def test_label_operator_matches_mne_extract_label_time_course(
    fsaverage_volume_model, mri_resolution
):
    """The cached label operator must equal mne's own label extraction."""
    import mne

    model, _, subjects_dir = fsaverage_volume_model
    names = ["ctx-lh-parsopercularis", "Left-Hippocampus"]
    rois = resolve_rois(names, atlas="aparc+aseg", subjects_dir=subjects_dir)

    n_src = model.n_sources
    rng = np.random.default_rng(1)
    data = rng.standard_normal((n_src, 20))
    vertno = [s["vertno"] for s in model.src]
    stc = mne.VolSourceEstimate(data, vertno, 0, 1 / 250.0)

    atlas_path = _volume_atlas_path("aparc+aseg", "fsaverage", subjects_dir)
    expected = np.asarray(
        mne.extract_label_time_course(
            stc,
            (str(atlas_path), names),
            src=model.src,
            mode="mean",
            allow_empty=True,
            mri_resolution=mri_resolution,
            verbose=False,
        )
    )
    got = model.label_operator(rois, mri_resolution=mri_resolution) @ data
    np.testing.assert_allclose(got, expected, atol=1e-12)


@pytest.mark.slow
def test_kernel_matches_full_mne_route(fsaverage_volume_model):
    """The headline claim: one matmul == apply_lcmv_raw + label extraction."""
    model, raw, subjects_dir = fsaverage_volume_model
    rois = resolve_rois(
        ["Broca", "Wernicke", "Hippocampus-lh"],
        atlas="aparc+aseg",
        subjects_dir=subjects_dir,
    )
    window = raw.get_data()[:, :250]

    fast = model.apply(window, kernel=model.roi_kernel(rois))
    slow = model.apply(window, label_operator=model.label_operator(rois), info=raw.info)
    assert fast.shape == (3, 250)
    denom = max(np.max(np.abs(slow)), 1e-30)
    assert np.max(np.abs(fast - slow)) / denom < 1e-10


@pytest.mark.slow
def test_multi_label_roi_equals_mean_over_all_its_sources(fsaverage_volume_model):
    """Broca = parsopercularis ∪ parstriangularis, weighted by source count."""
    model, _, subjects_dir = fsaverage_volume_model
    broca = resolve_rois(["Broca"], atlas="aparc+aseg", subjects_dir=subjects_dir)
    parts = resolve_rois(
        ["ctx-lh-parsopercularis", "ctx-lh-parstriangularis"],
        atlas="aparc+aseg",
        subjects_dir=subjects_dir,
    )
    merged = model.label_operator(broca, mri_resolution=False)[0]
    separate = model.label_operator(parts, mri_resolution=False)

    support = np.count_nonzero(separate, axis=1)
    weights = support / support.sum()
    np.testing.assert_allclose(merged, weights @ separate, atol=1e-12)


# ------------------------------------------------------------------
# _compute_inv_operator — source-space plumbing
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def baseline_raw():
    """A short EEG recording with digitised positions, for forward modelling."""
    mne = pytest.importorskip("mne")
    sfreq = 250.0
    montage = mne.channels.make_standard_montage("standard_1020")
    ch_names = [c for c in montage.ch_names if c not in ("T3", "T4", "T5", "T6")][:32]
    info = mne.create_info(ch_names, sfreq, "eeg")
    info.set_montage(montage)
    rng = np.random.default_rng(0)
    return mne.io.RawArray(
        rng.standard_normal((len(ch_names), int(sfreq * 10))) * 1e-6, info, verbose=False
    )


@pytest.fixture(scope="session")
def fs_subjects_dir():
    mne = pytest.importorskip("mne")
    try:
        fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
    except Exception as exc:  # pragma: no cover - network/OSF dependent
        pytest.skip(f"fsaverage unavailable: {exc}")
    import os.path as op

    return op.dirname(fs_dir)


@pytest.mark.slow
def test_compute_inv_operator_surface_unchanged(baseline_raw, fs_subjects_dir):
    """The default path must stay a surface model with a minimum-norm inverse."""
    from mne.minimum_norm import InverseOperator

    from mne_rt.tools import _compute_inv_operator

    inv, fwd, noise_cov, src = _compute_inv_operator(baseline_raw, subjects_fs_dir=fs_subjects_dir)
    assert isinstance(inv, InverseOperator)
    assert src.kind == "surface"
    assert fwd["nsource"] == 20484  # fsaverage ico-5, both hemispheres
    assert noise_cov is not None


@pytest.mark.slow
def test_lcmv_round_trip_on_a_volume_source_space(baseline_raw, fs_subjects_dir):
    """Build a beamformer from the returned forward/covariances and apply it."""
    import mne
    from mne.beamformer import apply_lcmv_raw, make_lcmv

    from mne_rt.tools import _compute_inv_operator

    _, fwd, noise_cov, _ = _compute_inv_operator(
        baseline_raw,
        subjects_fs_dir=fs_subjects_dir,
        src_type="volume",
        make_inverse=False,
    )

    # exactly what RTStream._prepare_raw_array does to each window
    window = mne.io.RawArray(baseline_raw.get_data()[:, :250], baseline_raw.info, verbose=False)
    window.set_eeg_reference("average", projection=True, verbose=False)
    data_cov = mne.compute_raw_covariance(window, method="empirical", verbose=False)

    filters = make_lcmv(
        window.info,
        fwd,
        data_cov,
        reg=0.05,
        noise_cov=noise_cov,
        pick_ori="max-power",
        weight_norm="unit-noise-gain",
        rank=None,
        verbose=False,
    )
    assert filters["is_free_ori"] is False  # required by the cached-kernel path
    stc = apply_lcmv_raw(window, filters, verbose=False)
    assert stc.data.shape[1] == 250
    assert np.all(np.isfinite(stc.data))


@pytest.mark.slow
def test_compute_inv_operator_volume(baseline_raw, fs_subjects_dir):
    from mne_rt.tools import _compute_inv_operator

    inv, fwd, _, src = _compute_inv_operator(
        baseline_raw, subjects_fs_dir=fs_subjects_dir, src_type="volume", make_inverse=False
    )
    assert inv is None  # beamformer-only session
    assert src.kind == "volume"
    assert fwd["nsource"] > 10000  # prebuilt 5 mm whole-brain grid


@pytest.mark.slow
def test_volume_labels_restriction_shrinks_the_source_space(baseline_raw, fs_subjects_dir):
    """The performance lever: build the grid from the ROIs, not the whole brain."""
    from mne_rt.tools import _compute_inv_operator

    _, whole, _, _ = _compute_inv_operator(
        baseline_raw, subjects_fs_dir=fs_subjects_dir, src_type="volume", make_inverse=False
    )
    _, restricted, _, _ = _compute_inv_operator(
        baseline_raw,
        subjects_fs_dir=fs_subjects_dir,
        src_type="volume",
        volume_labels=[
            "ctx-lh-parsopercularis",
            "ctx-lh-parstriangularis",
            "ctx-lh-superiortemporal",
            "ctx-lh-supramarginal",
            "Left-Hippocampus",
        ],
        make_inverse=False,
    )
    assert restricted["nsource"] < whole["nsource"] / 10


@pytest.mark.slow
def test_volume_inverse_forces_free_orientation(baseline_raw, fs_subjects_dir):
    """loose=0.2 is illegal without a cortical surface; it must be coerced."""
    from mne_rt.tools import _compute_inv_operator

    inv, _, _, _ = _compute_inv_operator(
        baseline_raw,
        subjects_fs_dir=fs_subjects_dir,
        src_type="volume",
        loose=0.2,
        depth=0.8,
        make_inverse=True,
    )
    assert inv is not None


def test_compute_inv_operator_rejects_bad_src_type(baseline_raw):
    from mne_rt.tools import _compute_inv_operator

    with pytest.raises(ValueError, match="src_type"):
        _compute_inv_operator(baseline_raw, src_type="mixed")
