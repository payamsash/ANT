"""Tests for RTStream instantiation and validation (no LSL required)."""

import json
import pathlib

import numpy as np
import pytest


@pytest.fixture()
def subjects_dir(tmp_path):
    return str(tmp_path)


def _make_rt_stream(tmp_path, **kwargs):
    from mne_rt import RTStream

    defaults = dict(
        subject_id="sub01",
        session="01",
        subjects_dir=str(tmp_path),
        montage="easycap-M1",
    )
    defaults.update(kwargs)
    return RTStream(**defaults)


def _make_baseline_raw(n_channels=8, sfreq=256.0, duration=2.0):
    import mne

    rng = np.random.default_rng(0)
    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    data = rng.standard_normal((n_channels, int(sfreq * duration))) * 1e-6
    return mne.io.RawArray(data, info, verbose=False)


def test_valid_instantiation(subjects_dir):
    from mne_rt import RTStream as RTStream

    nf = RTStream(
        subject_id="sub01",
        session="01",
        subjects_dir=subjects_dir,
        montage="easycap-M1",
        data_type="eeg",
    )
    assert nf.subject_id == "sub01"
    assert nf.session == "01"


def test_invalid_subject_id(subjects_dir):
    from mne_rt import RTStream as RTStream

    with pytest.raises(ValueError, match="subject_id"):
        RTStream(
            subject_id="",
            session="01",
            subjects_dir=subjects_dir,
            montage="easycap-M1",
        )


def test_invalid_session(subjects_dir):
    from mne_rt import RTStream as RTStream

    with pytest.raises(ValueError, match="session"):
        RTStream(
            subject_id="sub01",
            session="",
            subjects_dir=subjects_dir,
            montage="easycap-M1",
        )


def test_invalid_data_type(subjects_dir):
    from mne_rt import RTStream as RTStream

    with pytest.raises(ValueError, match="data_type"):
        RTStream(
            subject_id="sub01",
            session="01",
            subjects_dir=subjects_dir,
            montage="easycap-M1",
            data_type="fnirs",
        )


def test_invalid_artifact_correction(subjects_dir):
    from mne_rt import RTStream as RTStream

    with pytest.raises(ValueError, match="artifact_correction"):
        RTStream(
            subject_id="sub01",
            session="01",
            subjects_dir=subjects_dir,
            montage="easycap-M1",
            artifact_correction="invalid_method",
        )


def test_subject_dir_layout(tmp_path):
    from pathlib import Path

    from mne_rt import RTStream as RTStream

    nf = RTStream(
        subject_id="newsub",
        session="pre",
        subjects_dir=str(tmp_path),
        montage="easycap-M1",
    )
    assert Path(nf.subjects_dir) == tmp_path
    assert nf.subject_dir == tmp_path / "sub-newsub" / "ses-pre"


def test_modality_params_setter(subjects_dir):
    from mne_rt import RTStream as RTStream

    nf = RTStream(
        subject_id="sub01",
        session="01",
        subjects_dir=subjects_dir,
        montage="easycap-M1",
    )
    nf.modality_params = {"sensor_power": {"frange": [8, 12]}}
    assert nf.modality_params["sensor_power"]["frange"] == [8, 12]


def test_modality_params_invalid(subjects_dir):
    from mne_rt import RTStream as RTStream

    nf = RTStream(
        subject_id="sub01",
        session="01",
        subjects_dir=subjects_dir,
        montage="easycap-M1",
    )
    with pytest.raises(ValueError):
        nf.modality_params = "not_a_dict"


# ─────────────────────────────────────────────────────────────────────────────
# NFRealtime: artifact_rate and snr_data attributes
# ─────────────────────────────────────────────────────────────────────────────


class TestNFRealtimeNewFeatures:
    def test_replay_method_exists(self):
        from mne_rt import RTStream

        assert hasattr(RTStream, "replay")

    def test_run_blocks_method_exists(self):
        from mne_rt import RTStream

        assert hasattr(RTStream, "run_blocks")

    def test_run_blocks_empty_raises(self, tmp_path):
        from mne_rt import RTStream

        nf = RTStream(
            subject_id="sub01",
            session="01",
            subjects_dir=str(tmp_path),
            montage="easycap-M1",
        )
        with pytest.raises(ValueError, match="blocks"):
            nf.run_blocks(blocks=[])


# ─────────────────────────────────────────────────────────────────────────────
# fit_asr / fit_gedai / fit_maxwell / run_orica
# ─────────────────────────────────────────────────────────────────────────────


def test_fit_asr_before_baseline_raises(tmp_path):
    nf = _make_rt_stream(tmp_path)
    with pytest.raises(RuntimeError, match="record_baseline"):
        nf.fit_asr()


def test_fit_asr_happy_path(tmp_path):
    from mne_rt.tools import ASRDenoiser

    nf = _make_rt_stream(tmp_path)
    nf.raw_baseline = _make_baseline_raw()
    nf._sfreq = 256.0
    nf.fit_asr(cutoff=3.0)
    assert isinstance(nf.asr, ASRDenoiser)
    assert nf.asr.thresholds.shape == (8,)


def test_fit_gedai_before_baseline_raises(tmp_path):
    nf = _make_rt_stream(tmp_path)
    with pytest.raises(RuntimeError, match="record_baseline"):
        nf.fit_gedai()


def test_fit_gedai_band_filter_mode(tmp_path):
    from mne_rt.tools import GEDAIDenoiser

    nf = _make_rt_stream(tmp_path)
    nf.raw_baseline = _make_baseline_raw()
    nf.rec_info = nf.raw_baseline.info
    nf._sfreq = 256.0
    nf.fit_gedai(band=(8.0, 13.0), use_leadfield=False)
    assert isinstance(nf.gedai, GEDAIDenoiser)


def test_fit_maxwell_before_connect_raises(tmp_path):
    nf = _make_rt_stream(tmp_path, data_type="meg")
    with pytest.raises(RuntimeError, match="connect_to_lsl"):
        nf.fit_maxwell()


def test_fit_maxwell_happy_path(tmp_path, meg_info):
    nf = _make_rt_stream(tmp_path, data_type="meg", montage=None)
    nf.rec_info = meg_info
    nf.fit_maxwell()
    assert nf.maxwell_filter.mode == "sss"


def test_run_orica_sets_instance(tmp_path):
    from mne_rt.tools import ORICA

    nf = _make_rt_stream(tmp_path)
    nf.run_orica(n_channels=8, block_size=64)
    assert isinstance(nf.orica, ORICA)


# ─────────────────────────────────────────────────────────────────────────────
# save / load_nf_data
# ─────────────────────────────────────────────────────────────────────────────


def test_save_with_no_data_returns_empty_dict(tmp_path):
    nf = _make_rt_stream(tmp_path)
    saved = nf.save()
    assert saved == {}
    assert (nf.subject_dir / "beh").is_dir()


def test_save_writes_nf_data_json(tmp_path):
    nf = _make_rt_stream(tmp_path)
    nf.nf_data = {"sensor_power": [0.1, 0.2, 0.3]}
    nf._sfreq = 256.0
    nf.winsize = 1.0
    nf.duration = 3.0

    saved = nf.save()

    assert "nf_data" in saved
    with open(saved["nf_data"]) as fh:
        payload = json.load(fh)
    assert payload["data"]["sensor_power"] == [0.1, 0.2, 0.3]
    assert payload["meta"]["subject_id"] == "sub01"
    assert payload["meta"]["modalities"] == ["sensor_power"]


def test_save_writes_bids_tsv_when_requested(tmp_path):
    nf = _make_rt_stream(tmp_path)
    nf.nf_data = {"sensor_power": [0.1, 0.2]}
    nf._sfreq = 256.0
    nf.winsize = 1.0
    nf.duration = 2.0

    saved = nf.save(bids_tsv=True)

    assert "nf_tsv" in saved
    lines = saved["nf_tsv"].read_text().splitlines()
    assert lines[0] == "sensor_power"
    assert len(lines) == 3  # header + 2 rows


def test_load_nf_data_round_trip(tmp_path):
    from mne_rt import RTStream

    payload = {"meta": {"subject_id": "sub01"}, "data": {"sensor_power": [1.0, 2.0]}}
    fname = tmp_path / "beh.json"
    with open(fname, "w") as fh:
        json.dump(payload, fh)

    loaded = RTStream.load_nf_data(fname)
    assert loaded == payload


# ─────────────────────────────────────────────────────────────────────────────
# create_report
# ─────────────────────────────────────────────────────────────────────────────


def test_create_report_happy_path(tmp_path):
    nf = _make_rt_stream(tmp_path)
    nf.raw_baseline = _make_baseline_raw()
    nf.modality = "sensor_power"
    nf.nf_data = {"sensor_power": [0.1, 0.2, 0.3]}

    report_path = nf.create_report(include_psd=True, include_nf_signal=True)

    assert report_path.exists()
    assert report_path.suffix == ".html"


# ─────────────────────────────────────────────────────────────────────────────
# connect_to_lsl + record_baseline (mock LSL streaming)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def short_mock_fif(tmp_path):
    import mne

    montage = mne.channels.make_standard_montage("easycap-M1")
    ch_names = montage.ch_names[:8]
    sfreq = 256.0
    duration = 8.0
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    info.set_montage(montage)

    rng = np.random.default_rng(0)
    n_samples = int(duration * sfreq)
    data = rng.standard_normal((len(ch_names), n_samples)) * 1e-6
    raw = mne.io.RawArray(data, info, verbose=False)

    fname = tmp_path / "mock-raw.fif"
    raw.save(fname, verbose=False)
    return fname


def test_connect_to_lsl_and_record_baseline(tmp_path, short_mock_fif):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    try:
        nf.connect_to_lsl(mock_lsl=True, fname=str(short_mock_fif), timeout=15.0)
        assert nf.sfreq == pytest.approx(256.0)
        assert len(nf.rec_info["ch_names"]) == 8

        nf.record_baseline(baseline_duration=1.0, winsize=0.5)
        assert nf.raw_baseline is not None
        assert nf.raw_baseline.info["sfreq"] == pytest.approx(256.0)
        assert nf.raw_baseline.n_times >= int(1.0 * 256.0)
    finally:
        nf.save()  # disconnects the stream / mock player


# ─────────────────────────────────────────────────────────────────────────────
# connect_to_array (plain numpy-array input, no LSL required)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def array_info():
    import mne

    montage = mne.channels.make_standard_montage("easycap-M1")
    ch_names = montage.ch_names[:8]
    info = mne.create_info(ch_names=ch_names, sfreq=256.0, ch_types="eeg")
    info.set_montage(montage)
    return info


def _make_array_data(info, duration=8.0):
    rng = np.random.default_rng(0)
    n_samples = int(duration * info["sfreq"])
    return rng.standard_normal((len(info["ch_names"]), n_samples)) * 1e-6


def test_connect_to_array_invalid_shape(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    with pytest.raises(ValueError, match="2D"):
        nf.connect_to_array(np.zeros((8, 10, 2)), array_info)


def test_connect_to_array_channel_mismatch(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    with pytest.raises(ValueError, match="channels"):
        nf.connect_to_array(np.zeros((4, 100)), array_info)


def test_connect_to_array_and_record_baseline(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info)
    try:
        nf.connect_to_array(data, array_info)
        assert nf.sfreq == pytest.approx(256.0)
        assert len(nf.rec_info["ch_names"]) == 8
        assert nf.stream.connected

        nf.record_baseline(baseline_duration=1.0, winsize=0.5)
        assert nf.raw_baseline is not None
        assert nf.raw_baseline.info["sfreq"] == pytest.approx(256.0)
        assert nf.raw_baseline.n_times >= int(1.0 * 256.0)
    finally:
        nf.save()  # disconnects the stream


def test_connect_to_array_record_main_end_to_end(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1", bandpass_freq=(1.0, 40.0), notch_freq=50.0)
    data = _make_array_data(array_info, duration=20.0)
    try:
        nf.connect_to_array(data, array_info, n_repeat=np.inf)
        nf.record_baseline(baseline_duration=1.0, winsize=0.5)
        nf.record_main(
            duration=2.0,
            modality="sensor_power",
            show_raw_signal=False,
            show_nf_signal=False,
        )
        assert len(nf.nf_data["sensor_power"]) > 0
    finally:
        nf.save()


def test_connect_to_array_pick_types(tmp_path):
    import mne

    ch_names = [f"EEG{i:03d}" for i in range(4)] + ["EOG001"]
    ch_types = ["eeg"] * 4 + ["eog"]
    info = mne.create_info(ch_names=ch_names, sfreq=128.0, ch_types=ch_types)
    data = np.random.default_rng(0).standard_normal((5, 256)) * 1e-6

    nf = _make_rt_stream(tmp_path, montage=None)
    try:
        nf.connect_to_array(data, info, pick_types="eeg")
        assert nf.rec_info["ch_names"] == ch_names[:4]
    finally:
        nf.save()


def test_connect_to_array_open_stream_viewer_raises(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info)
    try:
        nf.connect_to_array(data, array_info)
        with pytest.raises(RuntimeError, match="connect_to_array"):
            nf.open_stream_viewer()
    finally:
        nf.save()


# ─────────────────────────────────────────────────────────────────────────────
# set_decoder / "decode" modality (RTDecode, no LSL required)
# ─────────────────────────────────────────────────────────────────────────────


def _make_fitted_decoder(info, n_epochs=20):
    from mne_rt import RTDecode

    rng = np.random.default_rng(0)
    n_ch = len(info["ch_names"])
    X = rng.standard_normal((n_epochs, n_ch, int(info["sfreq"])))
    y = np.array([0, 1] * (n_epochs // 2))
    return RTDecode(info=info, n_components=2).fit(X, y, verbose=False)


def test_set_decoder_rejects_wrong_type(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    with pytest.raises(TypeError, match="RTDecode"):
        nf.set_decoder(object())


def test_set_decoder_rejects_unfitted_decoder(tmp_path, array_info):
    from mne_rt import RTDecode

    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    with pytest.raises(RuntimeError, match="fit"):
        nf.set_decoder(RTDecode(info=array_info))


def test_decode_modality_without_decoder_raises(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info)
    try:
        nf.connect_to_array(data, array_info, n_repeat=np.inf)
        with pytest.raises(RuntimeError, match="set_decoder"):
            nf.record_main(duration=1.0, modality="decode", show_nf_signal=False)
    finally:
        nf.save()


def test_connect_to_array_decode_modality_end_to_end(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=20.0)
    decoder = _make_fitted_decoder(array_info)
    try:
        nf.connect_to_array(data, array_info, n_repeat=np.inf)
        nf.set_decoder(decoder)
        nf.record_main(duration=2.0, modality="decode", show_nf_signal=False)
        assert len(nf.nf_data["decode"]) > 0
        assert all(0.0 <= v <= 1.0 for v in nf.nf_data["decode"])
    finally:
        nf.save()


# ─────────────────────────────────────────────────────────────────────────────
# ArrayStream (standalone duck-typed LSL-stream shim behind connect_to_array)
# ─────────────────────────────────────────────────────────────────────────────


def _make_array_stream(n_channels=4, sfreq=100.0, n_samples=500, **kwargs):
    import mne

    from mne_rt.rt_stream import ArrayStream

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    data = np.random.default_rng(0).standard_normal((n_channels, n_samples)) * 1e-6
    return ArrayStream(data, info, **kwargs)


class TestArrayStream:
    def test_not_connected_raises(self):
        stream = _make_array_stream()
        assert not stream.connected
        with pytest.raises(RuntimeError, match="not connected"):
            stream.get_data()
        with pytest.raises(RuntimeError, match="connect"):
            _ = stream.info

    def test_connect_get_data_resets_n_new_samples(self):
        import time

        stream = _make_array_stream(sfreq=200.0, bufsize=1.0, chunk_size=10, n_repeat=np.inf)
        stream.connect()
        try:
            time.sleep(0.3)
            data, ts = stream.get_data(0.2)
            assert data.shape == (4, 40)
            assert ts.shape == (40,)
            assert stream.n_new_samples == 0
            time.sleep(0.2)
            assert stream.n_new_samples > 0
        finally:
            stream.disconnect()
        assert not stream.connected

    def test_pick_reduces_channels_before_connect(self):
        stream = _make_array_stream(n_channels=4)
        stream.pick(["EEG000", "EEG002"])
        assert stream._info["ch_names"] == ["EEG000", "EEG002"]
        stream.connect()
        try:
            data, _ = stream.get_data(0.1)
            assert data.shape[0] == 2
        finally:
            stream.disconnect()

    def test_pick_preserves_requested_order(self):
        stream = _make_array_stream(n_channels=4)
        stream.pick(["EEG002", "EEG000"])
        assert stream._info["ch_names"] == ["EEG002", "EEG000"]

    def test_pick_after_connect_raises(self):
        stream = _make_array_stream()
        stream.connect()
        try:
            with pytest.raises(RuntimeError, match="before connect"):
                stream.pick("eeg")
        finally:
            stream.disconnect()

    def test_get_data_excludes_bads(self):
        stream = _make_array_stream(n_channels=4)
        stream._info["bads"] = ["EEG001"]
        stream.connect()
        try:
            data, _ = stream.get_data(0.1)
            assert data.shape[0] == 3
        finally:
            stream.disconnect()

    def test_connect_twice_does_not_leak_thread(self):
        stream = _make_array_stream(sfreq=200.0, n_repeat=np.inf)
        stream.connect()
        first_thread = stream._thread
        stream.connect()
        try:
            assert not first_thread.is_alive()
            assert stream.connected
        finally:
            stream.disconnect()

    def test_filter_and_notch_filter_return_self(self):
        stream = _make_array_stream(sfreq=256.0, n_samples=2560)
        assert stream.filter(1.0, 40.0) is stream
        assert stream.notch_filter(50.0) is stream

    def test_n_repeat_stops_advancing(self):
        import time

        stream = _make_array_stream(
            sfreq=100.0, n_samples=20, chunk_size=5, n_repeat=1, bufsize=0.5
        )
        stream.connect()
        try:
            time.sleep(0.5)  # long enough to exhaust the 20-sample array once
            assert stream.n_new_samples == 20
        finally:
            stream.disconnect()


# ------------------------------------------------------------------
# Source-space configuration
# ------------------------------------------------------------------


def test_source_space_defaults_to_surface(tmp_path):
    nf = _make_rt_stream(tmp_path)
    assert nf.source_space == "surface"
    assert nf.source_atlas == "aparc"  # a surface annotation
    assert nf.src is None
    assert nf.data_cov is None


def test_volume_source_space_defaults_to_volumetric_atlas(tmp_path):
    """'aparc' is an annot with no .mgz, so a volume session must not use it."""
    nf = _make_rt_stream(tmp_path, source_space="volume")
    assert nf.source_atlas == "aparc+aseg"


def test_source_atlas_explicit_is_respected(tmp_path):
    nf = _make_rt_stream(tmp_path, source_space="volume", source_atlas="aseg")
    assert nf.source_atlas == "aseg"


def test_invalid_source_space_raises(tmp_path):
    with pytest.raises(ValueError, match="source_space"):
        _make_rt_stream(tmp_path, source_space="mixed")


@pytest.mark.parametrize("pos", [0, -1, "5"])
def test_invalid_source_pos_raises(tmp_path, pos):
    with pytest.raises(ValueError, match="source_pos"):
        _make_rt_stream(tmp_path, source_pos=pos)


def test_source_pos_stored(tmp_path):
    assert _make_rt_stream(tmp_path, source_pos=3.0).source_pos == 3.0


# ------------------------------------------------------------------
# Feature combiners in the live loop
# ------------------------------------------------------------------


def _run_with_combiner(tmp_path, array_info, combiner, **kwargs):
    """Run a short two-modality session and return the session object."""
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=20.0)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    try:
        nf.record_main(
            duration=2.0,
            winsize=1.0,
            modality=["sensor_power", "hjorth"],
            combiner=combiner,
            zscore_normalize=True,
            zscore_warmup=2,
            show_raw_signal=False,
            show_nf_signal=False,
            **kwargs,
        )
    finally:
        nf.save()
    return nf


def test_combiner_adds_a_combined_trace(tmp_path, array_info):
    from mne_rt import WeightedSumCombiner

    combiner = WeightedSumCombiner(weights={"sensor_power": 0.6, "hjorth": 0.4})
    nf = _run_with_combiner(tmp_path, array_info, combiner)

    assert "combined" in nf.nf_data
    assert len(nf.nf_data["combined"]) > 0
    # the per-modality traces are kept, not replaced
    assert len(nf.nf_data["sensor_power"]) == len(nf.nf_data["combined"])
    assert len(nf.nf_data["hjorth"]) == len(nf.nf_data["combined"])


def test_combined_value_is_the_combiner_output(tmp_path, array_info):
    """Once z-scoring is warmed up, the combined trace is the combiner's output."""
    from mne_rt import WeightedSumCombiner

    combiner = WeightedSumCombiner(weights={"sensor_power": 0.6, "hjorth": 0.4})
    nf = _run_with_combiner(tmp_path, array_info, combiner)

    warm = 2 - 1  # zscore_warmup=2 in the helper; normalised from that index on
    power = np.asarray(nf.nf_data["sensor_power"])[warm:]
    hjorth = np.asarray(nf.nf_data["hjorth"])[warm:]
    combined = np.asarray(nf.nf_data["combined"])[warm:]
    np.testing.assert_allclose(combined, 0.6 * power + 0.4 * hjorth, rtol=1e-9)


def test_combined_held_at_zero_until_zscoring_warms_up(tmp_path, array_info):
    """Guards against a scale discontinuity at the end of warmup.

    _apply_zscore passes values through in their native units until each
    modality has `zscore_warmup` windows behind it. Combining then would mix
    V²/Hz with a dimensionless index, and the combined trace would jump by
    orders of magnitude the moment normalisation engaged — fitting any attached
    protocol's baseline on meaningless numbers.
    """
    from mne_rt import WeightedSumCombiner

    combiner = WeightedSumCombiner(weights={"sensor_power": 0.6, "hjorth": 0.4})
    nf = _run_with_combiner(tmp_path, array_info, combiner)
    assert nf.nf_data["combined"][0] == 0.0
    assert any(v != 0.0 for v in nf.nf_data["combined"][1:])


def test_geometric_mean_with_zscoring_warns(tmp_path, array_info):
    """z-scores go negative; GeometricMeanCombiner logs them and collapses."""
    from mne_rt import GeometricMeanCombiner

    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=10.0)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    try:
        with pytest.warns(RuntimeWarning, match="positive inputs"):
            nf.record_main(
                duration=1.0,
                winsize=1.0,
                modality=["sensor_power", "hjorth"],
                combiner=GeometricMeanCombiner(features=["sensor_power", "hjorth"]),
                zscore_normalize=True,
                zscore_warmup=2,
                show_raw_signal=False,
                show_nf_signal=False,
            )
    finally:
        nf.save()


def test_zscored_norm_combiner_does_not_warn(tmp_path, array_info, recwarn):
    """It normalises internally, so advising zscore_normalize would be wrong."""
    from mne_rt import ZScoredNormCombiner

    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=10.0)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    try:
        nf.record_main(
            duration=1.0,
            winsize=1.0,
            modality=["sensor_power", "hjorth"],
            combiner=ZScoredNormCombiner(features=["sensor_power", "hjorth"], warmup=2),
            zscore_normalize=False,
            show_raw_signal=False,
            show_nf_signal=False,
        )
    finally:
        nf.save()
    assert not [w for w in recwarn if "zscore_normalize" in str(w.message)]


def test_a_failing_combiner_does_not_hang_the_session(tmp_path, array_info):
    """combine() is user code; raising must not skip the loop's shutdown."""

    class _Exploding:
        features = ["sensor_power"]

        def combine(self, values):
            raise RuntimeError("boom")

    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=10.0)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    try:
        nf.record_main(
            duration=1.0,
            winsize=1.0,
            modality=["sensor_power"],
            combiner=_Exploding(),
            show_raw_signal=False,
            show_nf_signal=False,
        )
    finally:
        nf.save()
    # the session completed, and the failed windows fell back to 0.0
    assert len(nf.nf_data["combined"]) == len(nf.nf_data["sensor_power"])
    assert all(v == 0.0 for v in nf.nf_data["combined"])


@pytest.mark.parametrize("name", ["snr_db", "entropy", "reward_x"])
def test_reserved_combined_names_rejected(tmp_path, array_info, name):
    """The combined trace shares namespaces with scales, labels and saved columns."""
    from mne_rt import WeightedSumCombiner

    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=5.0)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    with pytest.raises(ValueError, match="already in use"):
        nf.record_main(
            duration=1.0,
            modality=["sensor_power"],
            combiner=WeightedSumCombiner(weights={"sensor_power": 1.0}),
            combined_name=name,
            show_raw_signal=False,
            show_nf_signal=False,
        )


def test_combined_trace_can_drive_a_protocol(tmp_path, array_info):
    from mne_rt import WeightedSumCombiner, ZScoreProtocol

    combiner = WeightedSumCombiner(weights={"sensor_power": 1.0, "hjorth": 1.0})
    nf = _run_with_combiner(
        tmp_path,
        array_info,
        combiner,
        protocol={"combined": ZScoreProtocol(warmup_windows=1)},
    )
    assert "combined" in nf.reward_data
    assert len(nf.reward_data["combined"]) == len(nf.nf_data["combined"])


def test_combined_trace_is_saved(tmp_path, array_info):
    import json

    from mne_rt import WeightedSumCombiner

    combiner = WeightedSumCombiner(weights={"sensor_power": 1.0, "hjorth": 1.0})
    nf = _run_with_combiner(tmp_path, array_info, combiner)
    saved = nf.save()
    payload = json.loads(saved["nf_data"].read_text())
    assert "combined" in payload["data"]


def test_combiner_must_expose_combine(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=5.0)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    with pytest.raises(TypeError, match="combine"):
        nf.record_main(
            duration=1.0,
            modality=["sensor_power"],
            combiner=object(),
            show_raw_signal=False,
            show_nf_signal=False,
        )


def test_combined_name_collision_rejected(tmp_path, array_info):
    from mne_rt import WeightedSumCombiner

    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=5.0)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    with pytest.raises(ValueError, match="collides"):
        nf.record_main(
            duration=1.0,
            modality=["sensor_power"],
            combiner=WeightedSumCombiner(weights={"sensor_power": 1.0}),
            combined_name="sensor_power",
            show_raw_signal=False,
            show_nf_signal=False,
        )


def test_combiner_requiring_an_inactive_modality_is_rejected(tmp_path, array_info):
    from mne_rt import WeightedSumCombiner

    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=5.0)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    with pytest.raises(ValueError, match="laterality"):
        nf.record_main(
            duration=1.0,
            modality=["sensor_power"],
            combiner=WeightedSumCombiner(weights={"sensor_power": 1.0, "laterality": 1.0}),
            show_raw_signal=False,
            show_nf_signal=False,
        )


def test_combiner_without_zscore_warns_about_scale(tmp_path, array_info):
    """Mixing raw features of different magnitudes needs normalisation."""
    from mne_rt import WeightedSumCombiner

    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=10.0)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    try:
        with pytest.warns(RuntimeWarning, match="zscore_normalize"):
            nf.record_main(
                duration=1.0,
                winsize=1.0,
                modality=["sensor_power", "hjorth"],
                combiner=WeightedSumCombiner(weights={"sensor_power": 1.0, "hjorth": 1.0}),
                zscore_normalize=False,
                show_raw_signal=False,
                show_nf_signal=False,
            )
    finally:
        nf.save()


# ------------------------------------------------------------------
# Bad channels
# ------------------------------------------------------------------


def _bads_info(n_channels=16):
    """Info with digitised positions, no bads yet."""
    import mne

    montage = mne.channels.make_standard_montage("standard_1020")
    names = [c for c in montage.ch_names if c not in ("T3", "T4", "T5", "T6")][:n_channels]
    info = mne.create_info(names, 250.0, "eeg")
    info.set_montage(montage)
    return info, names


def _connected(tmp_path, info, names, duration=24.0):
    nf = _make_rt_stream(tmp_path, montage=None)
    data = np.random.default_rng(0).standard_normal((len(names), int(250.0 * duration))) * 1e-6
    nf.connect_to_array(data, info, n_repeat=np.inf)
    return nf


def test_acquired_ch_names_excludes_bads(tmp_path):
    """get_data() drops bads; rec_info keeps them. The two must not be confused."""
    info, names = _bads_info()
    nf = _connected(tmp_path, info, names)
    try:
        nf.rec_info["bads"] = [names[2], names[7]]
        acquired = nf._acquired_ch_names()
        assert names[2] not in acquired and names[7] not in acquired
        assert len(acquired) == len(names) - 2
        # rec_info still lists them, which the forward model and Maxwell need
        assert names[2] in nf.rec_info["ch_names"]
        window = nf.stream.get_data(1.0)[0]
        assert window.shape[0] == len(acquired)
    finally:
        nf.save()


def test_record_baseline_with_a_bad_channel_marked_after_connect(tmp_path):
    """Regression: this raised before a session could start.

    Bads are normally identified *after* connecting — during the baseline, or
    by BadChannelDetector — so excluding them at connect time would not help.
    """
    info, names = _bads_info()
    nf = _connected(tmp_path, info, names)
    try:
        nf.rec_info["bads"] = [names[2]]
        nf.record_baseline(baseline_duration=2.0, winsize=1.0)
        assert len(nf.raw_baseline.ch_names) == len(names) - 1
    finally:
        nf.save()


def test_record_main_with_a_bad_channel(tmp_path):
    info, names = _bads_info()
    nf = _connected(tmp_path, info, names)
    try:
        nf.rec_info["bads"] = [names[2]]
        nf.record_main(
            duration=2.0,
            winsize=1.0,
            modality=["sensor_power"],
            show_raw_signal=False,
            show_nf_signal=False,
        )
        assert len(nf.nf_data["sensor_power"]) > 0
    finally:
        nf.save()


def test_connectivity_indices_follow_the_acquired_channels(tmp_path):
    """A bad channel shifts every position in the acquired array.

    Indices built from rec_info would silently select the wrong pair.
    """
    info, names = _bads_info()
    nf = _connected(tmp_path, info, names)
    try:
        nf.rec_info["bads"] = [names[0]]  # shifts everything after it
        nf.picks = None
        nf.winsize = 1.0
        nf._sfreq = 250.0
        nf.params = {
            "frange": [8, 13],
            "channels": [[names[3], names[5]]],
            "method": "coh",
            "mode": "cwt_morlet",
            "n_cycles": 5,
        }
        prep = nf._sensor_connectivity_prep()
        acquired = nf._acquired_ch_names()
        assert prep["indices"][0][0] == acquired.index(names[3])
        assert prep["indices"][1][0] == acquired.index(names[5])
    finally:
        nf.save()


def test_no_bad_channels_leaves_indexing_unchanged(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=5.0)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    try:
        assert nf._acquired_ch_names() == list(nf.rec_info["ch_names"])
        assert nf._acquired_info() is nf.rec_info
    finally:
        nf.save()


# ------------------------------------------------------------------
# Instanced modalities — the same measure, several times over
# ------------------------------------------------------------------


def _run_instanced(tmp_path, array_info, modality, **kwargs):
    """Run a short session and return the session object."""
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=20.0)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    try:
        nf.record_main(
            duration=2.0,
            winsize=1.0,
            modality=modality,
            show_raw_signal=False,
            show_nf_signal=False,
            **kwargs,
        )
    finally:
        nf.save()
    return nf


def test_two_instances_of_one_modality_are_independent(tmp_path, array_info):
    """The load-bearing test: two bands of one measure, two distinct traces.

    If ``precomps`` or ``mod_params_dict`` collapsed on the shared base name,
    both instances would compute the same band and the traces would be equal.
    """
    nf = _run_instanced(
        tmp_path,
        array_info,
        ["sensor_power@alpha", "sensor_power@theta"],
        modality_params={
            "sensor_power@alpha": {"frange": [8, 13]},
            "sensor_power@theta": {"frange": [4, 8]},
        },
    )

    assert set(nf.nf_data) == {"sensor_power@alpha", "sensor_power@theta"}
    alpha = nf.nf_data["sensor_power@alpha"]
    theta = nf.nf_data["sensor_power@theta"]
    assert len(alpha) == len(theta) > 0
    assert alpha != theta

    assert nf.mod_params_dict["sensor_power@alpha"]["frange"] == [8, 13]
    assert nf.mod_params_dict["sensor_power@theta"]["frange"] == [4, 8]


def test_base_params_are_shared_across_instances(tmp_path, array_info):
    nf = _run_instanced(
        tmp_path,
        array_info,
        ["sensor_power@alpha", "sensor_power@theta"],
        modality_params={
            "sensor_power": {"method": "multitaper"},
            "sensor_power@alpha": {"frange": [8, 13]},
            "sensor_power@theta": {"frange": [4, 8]},
        },
    )
    for name in ("sensor_power@alpha", "sensor_power@theta"):
        assert nf.mod_params_dict[name]["method"] == "multitaper"


def test_plain_and_instanced_names_coexist(tmp_path, array_info):
    nf = _run_instanced(tmp_path, array_info, ["sensor_power", "sensor_power@theta"])
    assert set(nf.nf_data) == {"sensor_power", "sensor_power@theta"}


def test_duplicate_instance_names_rejected(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    with pytest.raises(ValueError, match="listed more than once"):
        nf.record_main(
            duration=1.0,
            modality=["sensor_power@a", "sensor_power@a"],
            show_raw_signal=False,
            show_nf_signal=False,
        )


@pytest.mark.parametrize(
    "name, match",
    [
        ("sensor_power@a@b", "more than one"),
        ("@alpha", "empty base"),
        ("sensor_power@", "empty instance label"),
        ("sensor_power@bad label", "may only contain"),
    ],
)
def test_malformed_instance_names_rejected(tmp_path, array_info, name, match):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    with pytest.raises(ValueError, match=match):
        nf.record_main(duration=1.0, modality=[name], show_raw_signal=False, show_nf_signal=False)


def test_picks_rejected_for_instanced_source_modality(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    with pytest.raises(ValueError, match="'picks' must be None"):
        nf.record_main(
            duration=1.0,
            modality=["source_connectivity@theta"],
            picks=array_info["ch_names"][:4],
            show_raw_signal=False,
            show_nf_signal=False,
        )


def test_picks_allowed_when_only_the_label_says_source(tmp_path, array_info):
    """The sensor/source test must read the base, not the whole name."""
    nf = _run_instanced(
        tmp_path,
        array_info,
        ["sensor_power@source"],
        picks=array_info["ch_names"][:4],
    )
    assert "sensor_power@source" in nf.nf_data


# ---- protocols ---------------------------------------------------


class _CountingProtocol:
    """Records how many times it was evaluated, and with what."""

    def __init__(self, threshold=0.0):
        self.current_threshold = threshold
        self.values = []

    def evaluate(self, value):
        self.values.append(float(value))
        return (float(value) > self.current_threshold, abs(float(value)))


def test_each_instance_drives_its_own_protocol(tmp_path, array_info):
    """One protocol object per instance, each evaluated exactly once a window.

    A shared object would be fired twice per window with two different values,
    silently corrupting any protocol that accumulates state.
    """
    p_alpha, p_theta = _CountingProtocol(), _CountingProtocol()
    nf = _run_instanced(
        tmp_path,
        array_info,
        ["sensor_power@alpha", "sensor_power@theta"],
        modality_params={
            "sensor_power@alpha": {"frange": [8, 13]},
            "sensor_power@theta": {"frange": [4, 8]},
        },
        protocol={"sensor_power@alpha": p_alpha, "sensor_power@theta": p_theta},
    )

    n_windows = len(nf.nf_data["sensor_power@alpha"])
    assert n_windows > 0
    assert len(p_alpha.values) == len(p_theta.values) == n_windows
    assert p_alpha.values != p_theta.values
    assert len(nf.reward_data["sensor_power@alpha"]) == n_windows


def test_protocol_keyed_by_instanced_base_rejected(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    with pytest.raises(ValueError, match="active instance"):
        nf.record_main(
            duration=1.0,
            modality=["sensor_power@alpha", "sensor_power@theta"],
            protocol={"sensor_power": _CountingProtocol()},
            show_raw_signal=False,
            show_nf_signal=False,
        )


def test_protocol_validation_happens_before_resources_are_created(tmp_path, array_info):
    """record_main has no teardown path, so it must reject before it allocates.

    Raising after the thread pool and the Qt windows exist would leave both
    orphaned.
    """
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    with pytest.raises(ValueError, match="active instance"):
        nf.record_main(
            duration=1.0,
            modality=["sensor_power@alpha", "sensor_power@theta"],
            protocol={"sensor_power": _CountingProtocol()},
            show_raw_signal=False,
            show_nf_signal=False,
        )
    assert getattr(nf, "executor", None) is None


def test_protocol_keyed_by_unknown_modality_rejected(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    with pytest.raises(ValueError, match="does not name an active modality"):
        nf.record_main(
            duration=1.0,
            modality=["sensor_power@alpha"],
            protocol={"hjorth": _CountingProtocol()},
            show_raw_signal=False,
            show_nf_signal=False,
        )


# ---- combiner ----------------------------------------------------


def test_combiner_over_two_instances(tmp_path, array_info):
    from mne_rt import WeightedSumCombiner

    combiner = WeightedSumCombiner(weights={"sensor_power@alpha": 0.5, "sensor_power@theta": 0.5})
    nf = _run_instanced(
        tmp_path,
        array_info,
        ["sensor_power@alpha", "sensor_power@theta"],
        modality_params={
            "sensor_power@alpha": {"frange": [8, 13]},
            "sensor_power@theta": {"frange": [4, 8]},
        },
        combiner=combiner,
        zscore_normalize=True,
        zscore_warmup=2,
    )
    assert "combined" in nf.nf_data
    assert len(nf.nf_data["combined"]) == len(nf.nf_data["sensor_power@alpha"])


def test_combiner_naming_an_instanced_base_suggests_the_instances(tmp_path, array_info):
    from mne_rt import WeightedSumCombiner

    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    with pytest.raises(ValueError, match="Did you mean"):
        nf.record_main(
            duration=1.0,
            modality=["sensor_power@alpha", "sensor_power@theta"],
            combiner=WeightedSumCombiner(weights={"sensor_power": 1.0}),
            show_raw_signal=False,
            show_nf_signal=False,
        )


# ---- outputs -----------------------------------------------------


def test_instanced_names_round_trip_through_save(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=20.0)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    try:
        nf.record_main(
            duration=2.0,
            winsize=1.0,
            modality=["sensor_power@alpha", "sensor_power@theta"],
            protocol={"sensor_power@alpha": _CountingProtocol(threshold=-1e30)},
            show_raw_signal=False,
            show_nf_signal=False,
        )
    finally:
        nf.save(bids_tsv=True)

    beh = next(pathlib.Path(nf.subject_dir).rglob("*_beh.json"))
    payload = json.loads(beh.read_text())
    assert set(payload["meta"]["modalities"]) == {"sensor_power@alpha", "sensor_power@theta"}
    assert payload["meta"]["modality_bases"] == {
        "sensor_power@alpha": "sensor_power",
        "sensor_power@theta": "sensor_power",
    }
    assert "sensor_power@alpha" in payload["data"]
    assert "reward_sensor_power@alpha" in payload["data"]

    tsv = next(pathlib.Path(nf.subject_dir).rglob("*_beh.tsv"))
    header = tsv.read_text().splitlines()[0].split("\t")
    assert "sensor_power@alpha" in header and "sensor_power@theta" in header


def test_osc_addresses_are_sanitised_and_distinct(tmp_path, array_info):
    """'@' must not reach an OSC address, and the two must stay distinct."""

    class _FakeOSC:
        def __init__(self):
            self.sent = []

        def send_all(self, modalities, values):
            self.sent.append(list(modalities))

    sender = _FakeOSC()
    _run_instanced(
        tmp_path,
        array_info,
        ["sensor_power@alpha", "sensor_power@theta"],
        osc_sender=sender,
    )
    assert sender.sent
    for names in sender.sent:
        assert names == ["sensor_power_alpha", "sensor_power_theta"]
        assert not any("@" in n for n in names)


def test_per_instance_delays_are_recorded(tmp_path, array_info):
    nf = _run_instanced(
        tmp_path,
        array_info,
        ["sensor_power@alpha", "sensor_power@theta"],
        estimate_delays=True,
    )
    assert set(nf.method_delays) == {"sensor_power@alpha", "sensor_power@theta"}
    assert all(v for v in nf.method_delays.values())


# ------------------------------------------------------------------
# Z-scoring is scale-free
# ------------------------------------------------------------------


def _zscored_session(tmp_path, array_info, **kwargs):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=40.0)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    try:
        nf.record_main(
            duration=12.0,
            winsize=1.0,
            modality="sensor_power",
            zscore_normalize=True,
            show_raw_signal=False,
            show_nf_signal=False,
            **kwargs,
        )
    finally:
        nf.save()
    return nf


def test_zscored_trace_is_on_the_order_of_one(tmp_path, array_info):
    """Band power is ~1e-13, so an absolute std floor swamped it entirely.

    Before this was fixed the "z-scores" came out ~1e-8. The loose bound is
    deliberate — the point is the order of magnitude, not the exact spread.
    """
    nf = _zscored_session(tmp_path, array_info, zscore_warmup=4)
    z = np.asarray(nf.nf_data["sensor_power"][4:])
    assert len(z) > 2
    assert np.max(np.abs(z)) > 0.1  # would be ~1e-8 under the old floor
    assert np.max(np.abs(z)) < 100.0


def test_zscore_state_is_kept_after_the_warmup_buffer_is_dropped(tmp_path, array_info):
    """The combiner warm-gate reads a counter, not the (now freed) buffer."""
    from mne_rt import WeightedSumCombiner

    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=40.0)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    try:
        nf.record_main(
            duration=12.0,
            winsize=1.0,
            modality=["sensor_power", "hjorth"],
            combiner=WeightedSumCombiner(weights={"sensor_power": 0.5, "hjorth": 0.5}),
            zscore_normalize=True,
            zscore_warmup=3,
            show_raw_signal=False,
            show_nf_signal=False,
        )
    finally:
        nf.save()
    combined = nf.nf_data["combined"]
    # Held at 0.0 until every feature is normalised, then live.
    assert combined[0] == 0.0
    assert any(v != 0.0 for v in combined[3:])


def test_adaptive_zscore_tracks_variance_not_absolute_deviation(tmp_path, array_info):
    """With zscore_alpha > 0 the running spread must stay a variance.

    Exponentially averaging ``abs(delta)`` instead gives the mean absolute
    deviation (~0.798 sigma for Gaussian input), inflating z by ~25%.
    """
    nf = _zscored_session(tmp_path, array_info, zscore_warmup=4, zscore_alpha=0.05)
    z = np.asarray(nf.nf_data["sensor_power"][4:])
    assert np.all(np.isfinite(z))
    assert np.max(np.abs(z)) > 0.1


def test_display_scales_follow_the_zscored_units(tmp_path, array_info):
    """A z-score is O(1); dividing it by a 1e-12 band-power scale hides it."""
    import mne_rt.rt_stream as rs

    captured = {}
    real_init = rs.NFPlot.__init__

    def _spy(self, modalities, scales_dict, *a, **kw):
        captured["scales"] = dict(scales_dict)
        return real_init(self, modalities, scales_dict, *a, **kw)

    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=10.0)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(rs.NFPlot, "__init__", _spy)
    try:
        nf.record_main(
            duration=2.0,
            winsize=1.0,
            modality="sensor_power",
            zscore_normalize=True,
            zscore_warmup=2,
            show_raw_signal=False,
            show_nf_signal=True,
        )
    finally:
        monkey.undo()
        nf.save()
    assert captured["scales"]["sensor_power"] == 1.0
