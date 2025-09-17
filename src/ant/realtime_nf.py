## don't enter here without a good guide!
from __future__ import annotations

import datetime
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Any, Union
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from PyQt6.QtWidgets import QPushButton
from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph as pg
from scipy.optimize import curve_fit
from scipy.signal import sosfiltfilt
from pactools import Comodulogram

import mne
from mne import (
        Report,
        read_labels_from_annot,
        set_log_level,
        write_forward_solution,
        write_cov
)
from mne.channels import get_builtin_montages, read_dig_captrak
from mne.io import RawArray
from mne.utils import logger
from mne.minimum_norm import (
        apply_inverse_raw,
        read_inverse_operator,
        write_inverse_operator,
)
from mne.beamformer import apply_lcmv_raw, make_lcmv
from mne_connectivity import spectral_connectivity_time
from mne_features.univariate import (
        compute_app_entropy,
        compute_samp_entropy,
        compute_spect_entropy,
        compute_svd_entropy,
)
from mne_lsl.lsl import local_clock
from mne_lsl.player import PlayerLSL as Player
from mne_lsl.stream import StreamLSL as Stream
from mne_lsl.stream_viewer import StreamViewer as Viewer

from ant.tools import *
from ant.tools import _compute_inv_operator

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class NFRealtime:
        """
        EEG/Neurofeedback real-time session.

        This class sets up a subject session for real-time EEG/neurofeedback
        processing. It validates inputs, configures session parameters and
        prepares logging. Optional preprocessing settings such as filtering
        and artifact correction can be specified.

        Parameters
        ----------
        subject_id : str
                Unique identifier for the subject. Must be a non-empty string.
        visit : int
                Visit number (>= 1).
        session : {"baseline", "main"}
                Session type.
        subjects_dir : str
                Path to the directory containing subject data.
        montage : str
                EEG montage: either a built-in montage or a path to a ``.bvct`` file.
        mri : bool
                Whether MRI data are available.
        subject_fs_id : str, default "fsaverage"
                FreeSurfer subject ID.
        subjects_fs_dir : str | None, default None
                Path to FreeSurfer subjects directory.
        filtering : bool, default False
                Whether to apply band-pass filtering.
        l_freq : float, default 1.0
                Low-frequency cutoff (Hz).
        h_freq : float, default 40.0
                High-frequency cutoff (Hz).
        artifact_correction : {False, "orica", "lms"}, default False
                Artifact-correction method.
        ref_channel : str, default "Fp1"
                Reference EEG channel.
        save_raw : bool, default True
                Whether to save raw EEG data.
        save_nf_signal : bool, default True
                Whether to save the neurofeedback signal.
        config_file : str | None, default None
                Path to a YAML configuration file. If None, uses the project default.
        verbose : int | None, default None
                Logging verbosity level.

        Raises
        ------
        ValueError
                If any parameter fails validation.
        """

        VALID_SESSIONS = {"baseline", "main"}
        VALID_ARTIFACT_METHODS = {False, "orica", "lms"}

        def __init__(
                self,
                subject_id: str,
                visit: int,
                session: str,
                subjects_dir: str,
                montage: str,
                mri: bool,
                subject_fs_id: str = "fsaverage",
                subjects_fs_dir: Optional[str] = None,
                filtering: bool = False,
                l_freq: float = 1.0,
                h_freq: float = 40.0,
                artifact_correction: bool | str = False,
                ref_channel: str = "Fp1",
                save_raw: bool = True,
                save_nf_signal: bool = True,
                config_file: Optional[str] = None,
                verbose: Optional[int] = None,
        ) -> None:
                # --- Validation ---
                if not subject_id or not isinstance(subject_id, str):
                        raise ValueError("`subject_id` must be a non-empty string.")

                if not isinstance(visit, int) or visit < 1:
                        raise ValueError("`visit` must be a positive integer (>= 1).")

                if session not in self.VALID_SESSIONS:
                        raise ValueError(
                                f"`session` must be one of {self.VALID_SESSIONS}, got {session!r}."
                        )

                if not (
                        montage in get_builtin_montages()
                        or (montage.endswith(".bvct") and Path(montage).is_file())
                ):
                        raise ValueError(
                                "`montage` must be a built-in montage name or a valid '.bvct' file path."
                        )

                if not isinstance(mri, bool):
                        raise ValueError("`mri` must be a boolean.")

                if not subject_fs_id or not isinstance(subject_fs_id, str):
                        raise ValueError("`subject_fs_id` must be a non-empty string.")

                if subjects_fs_dir is not None:
                        if not Path(subjects_fs_dir).is_dir():
                                raise ValueError("`subjects_fs_dir` must be None or an existing directory.")

                if not isinstance(filtering, bool):
                        raise ValueError("`filtering` must be a boolean.")

                if artifact_correction not in self.VALID_ARTIFACT_METHODS:
                        raise ValueError(
                                f"`artifact_correction` must be one of "
                                f"{self.VALID_ARTIFACT_METHODS}, got {artifact_correction!r}."
                        )

                if not isinstance(save_raw, bool):
                        raise ValueError("`save_raw` must be a boolean.")

                if not isinstance(save_nf_signal, bool):
                        raise ValueError("`save_nf_signal` must be a boolean.")

                if config_file is None:
                        config = PROJECT_ROOT.parent / "config_methods.yml"
                elif config_file.endswith(".yml") and Path(config_file).is_file():
                        config = config_file
                else:
                        raise ValueError("`config_file` must be None or a valid YAML file path.")

                # --- Assignments ---
                self.subject_id = subject_id
                self.visit = visit
                self.session = session
                self.subjects_dir = subjects_dir
                self.montage = montage
                self.mri = mri
                self.subject_fs_id = subject_fs_id
                self.subjects_fs_dir = subjects_fs_dir
                self.filtering = filtering
                self.l_freq = l_freq
                self.h_freq = h_freq
                self.artifact_correction = artifact_correction
                self.ref_channel = ref_channel
                self.save_raw = save_raw
                self.save_nf_signal = save_nf_signal
                self.config_file = config
                self.verbose = verbose

                set_log_level(verbose)

        def connect_to_lsl(
                self,
                chunk_size: int = 10,
                mock_lsl: bool = False,
                fname: Optional[str] = None,
                n_repeat: Union[int, float] = np.inf,
                bufsize_baseline: int = 4,
                bufsize_main: int = 3,
                acquisition_delay: float = 0.001,
                timeout: float = 2.0,
        ) -> None:
                """
                Connect to a Lab Streaming Layer (LSL) EEG stream.

                This method connects to a live or mock LSL stream, sets the montage and
                metadata, and exposes the stream’s public methods as attributes.

                Parameters
                ----------
                chunk_size : int, default 10
                        Number of samples per chunk for streaming.
                mock_lsl : bool, default False
                        If True, stream a pre-recorded EEG file instead of live LSL data.
                fname : str | Path | None, default None
                        Path to the EEG file for mock streaming. Uses sample data if None.
                n_repeat : int | float, default np.inf
                        Number of times to repeat the mock recording.
                bufsize_baseline : int, default 4
                        Buffer size for 'baseline' sessions.
                bufsize_main : int, default 3
                        Buffer size for 'main' sessions.
                acquisition_delay : float, default 0.001
                        Delay (s) between consecutive acquisition attempts.
                timeout : float, default 2.0
                        Max time (s) to wait for an LSL connection.

                Raises
                ------
                FileNotFoundError
                        If `mock_lsl` is True and the specified `fname` does not exist.
                ConnectionError
                        If the LSL stream cannot be connected within `timeout`.
                """
                # --- Prepare subject directory ---
                self.subject_dir = Path(self.subjects_dir) / self.subject_id
                self.subject_dir.mkdir(parents=True, exist_ok=True)

                # --- Disconnect existing stream if present ---
                if hasattr(self, "stream") and getattr(self.stream, "connected", False):
                        self.stream.disconnect()

                # --- Determine file for mock streaming ---
                if mock_lsl and fname is None:
                        fname = PROJECT_ROOT / "data" / "sample" / "sample_data.vhdr"

                # --- Buffer size based on session ---
                self.bufsize = bufsize_baseline if self.session == "baseline" else bufsize_main

                # --- Load montage if file path provided ---
                if Path(self.montage).is_file():
                        self.montage = read_dig_captrak(self.montage)

                # --- Create and connect stream ---
                self.source_id = uuid.uuid4().hex
                if mock_lsl:
                        Player(
                        fname,
                        chunk_size=chunk_size,
                        n_repeat=n_repeat,
                        source_id=self.source_id,
                        ).start()

                stream = Stream(bufsize=self.bufsize, source_id=self.source_id)
                stream.connect(acquisition_delay=acquisition_delay, timeout=timeout)
                stream.set_montage(self.montage, on_missing="warn")
                stream.pick("eeg")
                stream.set_meas_date(
                        datetime.datetime.now().replace(tzinfo=datetime.timezone.utc)
                )
                self.stream = stream

                # --- Metadata ---
                self.sfreq = stream.info["sfreq"]
                self.rec_info = stream.info
                self.rec_info["subject_info"] = {"his_id": self.subject_id}

                # --- Expose stream methods as attributes ---
                for name in dir(self.stream):
                        if not name.startswith("__"):
                                attr = getattr(self.stream, name)
                                if callable(attr):
                                        setattr(self, name, attr)
        
        def record_baseline(
                self,
                baseline_duration: float,
                winsize: float = 3.0,
        ) -> None:
                """
                Record a baseline EEG segment for neurofeedback feature extraction.

                Continuously streams data from the LSL connection for the given duration,
                fetching samples in chunks of length ``winsize`` seconds. The baseline is
                saved to disk as a Raw FIF file and an inverse operator is computed.

                Parameters
                ----------
                baseline_duration : float
                        Desired duration of the baseline recording in seconds.
                winsize : float, default 3.0
                        Size of each data window (seconds) fetched from the buffer.

                Returns
                -------
                None
                        The method saves the baseline data and updates:
                        * ``self.raw_baseline`` : mne.io.Raw
                        The recorded baseline as a Raw object.
                        * Internal inverse operator via ``self.compute_inv_operator()``.

                Notes
                -----
                - Output file: ``<subject_dir>/baseline/visit_<visit>-raw.fif``.
                - The LSL stream remains connected after this method completes.
                """
                
                self.baseline_duration = baseline_duration
                print("Recording initiated ...")

                # --- Collect data in windows ---
                t_start = local_clock()
                chunks: list[np.ndarray] = []
                while local_clock() < t_start + baseline_duration:
                        chunks.append(self.stream.get_data(winsize)[0])
                        time.sleep(winsize)

                data = np.concatenate(chunks, axis=1)

                # --- Create MNE Raw object and save it ---
                raw_baseline = RawArray(data, self.rec_info)
                baseline_dir = Path(self.subject_dir) / "baseline"
                baseline_dir.mkdir(parents=True, exist_ok=True)
                fname_save = baseline_dir / f"visit_{self.visit}-raw.fif"
                raw_baseline.save(fname_save, overwrite=True)

                self.raw_baseline = raw_baseline

                # --- Compute inverse operator ---
                inv_dir = Path(self.subject_dir) / "inv"
                inv_dir.mkdir(parents=True, exist_ok=True)
                self.compute_inv_operator()

        def record_main(
                self,
                duration: float,
                modality: str | list[str] = "sensor_power",
                picks: str | list[str] | None = None,
                winsize: float = 1.0,
                estimate_delays: bool = False,
                modality_params: dict[str, Any] | None = None,
                show_raw_signal: bool = True,
                show_nf_signal: bool = True,
                time_window: float = 10.0,
                show_brain_activation: bool = False,
                show_design_viz: bool = True,
                design_viz: str = "VisualRorschach",
                use_ring_buffer: bool = False,
        ) -> None:
                """
                Record EEG data and extract neural features for neurofeedback.

                Streams EEG data from the LSL connection for ``duration`` seconds,
                applies the chosen feature-extraction modality (or list of modalities),
                and optionally visualizes the neurofeedback signal in real time.

                Parameters
                ----------
                duration : float
                        Desired recording length in seconds.
                modality : str | list of str, default 'sensor_power'
                        Neurofeedback feature(s) to extract.
                picks : str | list of str | None, default None
                        Channel names to include. If None, all channels are used.
                winsize : float, default 1.0
                        Window size in seconds for fetching data from the buffer.
                estimate_delays : bool, default False
                        If True, acquisition, artifact, method, and plot delays are measured.
                modality_params : dict | None, default None
                        Optional parameter overrides for each modality.
                show_raw_signal : bool, default True
                        Show real-time raw EEG signal plot.
                show_nf_signal : bool, default True
                        Show real-time neurofeedback feature plot.
                time_window : float, default 10.0
                        Length of time window (seconds) displayed in the NF plot.
                show_design_viz : bool, default True
                        Show the neurofeedback “design” visualization (e.g., py5 sketch).
                design_viz : str, default 'VisualRorschach'
                        Name of the visualization preset.
                use_ring_buffer : bool, default False
                        If True, use ring-buffer processing instead of fixed winsize pulls.

                Raises
                ------
                NotImplementedError
                        If a requested modality is not implemented.
                AssertionError
                        If a source-based modality is used with non-None ``picks``.

                Notes
                -----
                - Baseline artifact correction supports 'lms' or 'orica'.
                - Real-time visualization uses PyQt6 + pyqtgraph.

                Examples
                --------
                >>> session.record_main(duration=120,
                ...                     modality="sensor_power",
                ...                     picks=["C3", "C4"],
                ...                     winsize=2)
                """
                # ---------------------- Configuration ----------------------
                self.duration = duration
                self.modality = modality
                self.picks = picks
                self.modality_params = modality_params
                self.winsize = winsize
                self.window_size_s = int(self.winsize * self.rec_info["sfreq"])
                self.estimate_delays = estimate_delays
                self._sfreq = self.rec_info["sfreq"]
                self.show_raw_signal = show_raw_signal
                self.show_nf_signal = show_nf_signal
                self.time_window = time_window
                self.show_design_viz = show_design_viz
                self.design_viz = design_viz
                self.use_ring_buffer = use_ring_buffer


                # ---------------------- Artifact correction setup ----------------------
                if self.artifact_correction == "lms":
                        ref_ch_idx = self.rec_info["ch_names"].index(self.ref_channel)
                if self.artifact_correction == "orica":
                        self.run_orica(n_channels=len(self.rec_info["ch_names"]), forgetfac=0.99)

                # ---------------------- Modality preparation ----------------------
                mods = [modality] if isinstance(modality, str) else modality
                self._mods = mods
                self.executor = ThreadPoolExecutor(max_workers=len(mods))
                self.mod_params_dict = {
                        mod: get_params(self.config_file, mod, self.modality_params)
                        for mod in mods
                }
                precomps, nf_mods = [], []
                for mod in mods:
                        self.params = self.mod_params_dict[mod]
                        prep = getattr(self, f"_{mod}_prep", None)
                        fn = getattr(self, f"_{mod}", None)
                        if not callable(fn):
                                raise NotImplementedError(f"{mod} modality not implemented.")
                        if "source" in mod:
                                assert self.picks is None, "picks must be None for source methods."
                        precomps.append(prep())
                        nf_mods.append(fn)

                # ---------------------- Delay bookkeeping ----------------------
                if estimate_delays:
                        acq_delays, artifact_delays = [], []
                        method_delays = {m: [] for m in mods}
                        plot_delays = []

                # ---------------------- Neurofeedback plot setup ----------------------
                if show_nf_signal:
                        self.app = QtWidgets.QApplication([])
                        self.plot_widget = pg.PlotWidget(title="Neurofeedback")
                        self.plot_widget.setLabel("bottom", "Time", units="s")
                        self.plot_widget.setLabel("left", "NF Signals")
                        self.plot_widget.resize(1500, 900)
                        self.plot_widget.showGrid(x=True, y=True)
                        self.plot_widget.show()

                        self.colors_list = [
                        "#5DA5A4", "#9A7DFF", "#FFB085",
                        "#8FBF87", "#D98BA3", "#E0C368",
                        ]
                        self.scales_dict = {
                        "sensor_power": 1e-12,
                        "band_ratio": 4,
                        "source_power": 3e-2,
                        "sensor_connectivity": 1,
                        "source_connectivity": 1,
                        "sensor_graph": 0.05,
                        "source_graph": 2e-17,
                        "entropy": 3,
                        "argmax_freq": 8,
                        "individual_peak_power": 3e-12,
                        }
                        self.channel_scales = [1.0] * len(mods)
                        self.shifts = np.arange(0, len(mods) * 2, 2)
                        self.plot_widget.setYRange(-1 + self.shifts[0], 1 + self.shifts[-1])
                        self.curve = self.plot_widget.plot(pen="y")
                        self.time_axis = np.linspace(0, time_window, int(self._sfreq))
                        self.text_items = None

                        # Scale buttons
                        for i, shift in enumerate(self.shifts):
                                vb = self.plot_widget.getViewBox()
                                pos = vb.mapViewToScene(QtCore.QPointF(0, shift))
                                px, py = int(pos.x()), int(pos.y())

                                for sign, dx in [("+", -20), ("-", 10)]:
                                        btn = QPushButton(sign, self.plot_widget)
                                        btn.setStyleSheet(
                                        "color: grey; background: transparent; border: none; font-weight: bold;"
                                        )
                                        btn.resize(25, 25)
                                        btn.move(px + dx, py + 10)
                                        btn.show()
                                        handler = self.scale_up if sign == "+" else self.scale_down
                                        btn.clicked.connect(lambda checked, idx=i, h=handler: h(idx))

                if show_brain_activation:
                        self.initiate_brain_plot()
                if show_raw_signal:
                        self.plot_rt()
                if self.show_design_viz:
                        pass

                if self.filtering:
                        self.stream.filter(l_freq=self.l_freq, h_freq=self.h_freq)

                # ---------------------- Main loop: direct or ring buffer ----------------------
                nf_data = {m: [] for m in mods}
                t_start = local_clock()

                if not use_ring_buffer:
                        # --- Direct winsize fetching ---
                        while local_clock() < t_start + duration:
                                tic = time.time()
                                data = self.stream.get_data(winsize, picks=picks)[0]
                                if estimate_delays:
                                        acq_delays.append(time.time() - tic)
                                if data.shape[1] != self.window_size_s:
                                        continue

                                # Artifact correction
                                if self.artifact_correction == "lms":
                                        art_tic = time.time()
                                        data = remove_blinks_lms(data, ref_ch_idx=ref_ch_idx, n_taps=5, mu=0.01)
                                        if estimate_delays:
                                                artifact_delays.append(time.time() - art_tic)
                                elif self.artifact_correction == "orica":
                                        art_tic = time.time()
                                        sources = self.orica.transform(data)
                                        blink_idx, _ = self.orica.find_blink_ic(self.blink_template, threshold=0.4)
                                        sources_clean = sources.copy()
                                        for idx in blink_idx:
                                                sources_clean[idx, :] = 0.0
                                        data = self.orica.inverse_transform(sources_clean)
                                        if estimate_delays:
                                                artifact_delays.append(time.time() - art_tic)

                                # Feature extraction (parallel)
                                futures = [self.executor.submit(nf_mods[i], data, **precomps[i])
                                                for i in range(len(mods))]
                                for m, fut in zip(mods, futures):
                                        nf_val, m_delay = fut.result()
                                        nf_data[m].append(nf_val)
                                        if estimate_delays:
                                                method_delays[m].append(m_delay)

                                # Plot update
                                if show_nf_signal:
                                        plot_tic = time.time()
                                        last_vals = [nf_data[m][-1] for m in mods]
                                        self.update_nf_plot(last_vals, labels=mods)
                                        self.app.processEvents()
                                        if estimate_delays:
                                                plot_delays.append(time.time() - plot_tic)
                                        time.sleep(0.001)

                                if show_brain_activation:
                                        plot_tic = time.time()
                                        self.plot_brain_activation(data)
                                        if estimate_delays:
                                                plot_delays.append(time.time() - plot_tic)
                                        time.sleep(0.05)
                                if show_design_viz:
                                        plot_design(nf_val)

                else:
                        self.nf_data = self._record_with_ring_buffer(
                                duration=self.duration,
                                mods=mods,
                                nf_mods=nf_mods,
                                precomps=precomps,
                                estimate_delays=self.estimate_delays,
                                ref_ch_idx=ref_ch_idx if self.artifact_correction == "lms" else None,
                                fetch_secs=max(self.winsize * 4, 2.0),
                                hop_samples=max(self.window_size_s // 2, 1),
                        )

                # ---------------------- Save results ----------------------
                self.nf_data = nf_data
                if estimate_delays:
                        self.acq_delays = acq_delays
                        self.artifact_delays = artifact_delays
                        self.method_delays = method_delays
                        self.plot_delays = plot_delays
                        self.save(
                        nf_data=True,
                        acq_delay=True,
                        artifact_delay=True,
                        method_delay=True,
                        format="json",
                        )
                else:
                        self.save(
                        nf_data=True,
                        acq_delay=False,
                        artifact_delay=False,
                        method_delay=False,
                        format="json",
                        )
                if show_nf_signal:
                        # self.app.exec()
                        self.plot_widget.close()
                        self.app.quit()
        
        def _record_with_ring_buffer(
                self,
                duration: float,
                mods: list[str],
                nf_mods: list[callable],
                precomps: list[dict],
                estimate_delays: bool = False,
                ref_ch_idx: int | None = None,
                fetch_secs: float = 2.0,
                hop_samples: int = 1,
                plot_interval: float = 0.05,
        ):
                """
                Internal helper: acquire neurofeedback data using a ring buffer.

                Parameters
                ----------
                duration : float
                        Total acquisition time in seconds.
                mods : list of str
                        Names of the NF modalities.
                nf_mods : list of callables
                        Functions implementing each modality.
                precomps : list of dict
                        Pre-computed parameters for each modality.
                estimate_delays : bool, optional
                        Whether to collect acquisition / artifact / method / plot delays.
                ref_ch_idx : int | None, optional
                        Index of the reference channel (needed if LMS artifact correction is used).
                fetch_secs : float, optional
                        Amount of data (in seconds) to pull from the stream at each fetch.
                hop_samples : int, optional
                        Number of samples to advance the ring buffer each iteration.
                plot_interval : float, optional
                        Minimum time (s) between successive UI updates.

                Returns
                -------
                nf_data : dict
                        Dictionary mapping modality name → list of NF values over time.
                """
                # prepare containers
                n_ch = len(self.rec_info["ch_names"])
                ring_buffer = np.zeros((n_ch, 0), dtype=np.float32)
                nf_data = {mod: [] for mod in mods}

                if estimate_delays:
                        acq_delays, artifact_delays = [], []
                        method_delays = {mod: [] for mod in mods}
                        plot_delays = []

                max_buf_len = int(fetch_secs * self._sfreq) + self.window_size_s
                last_plot_time = 0.0
                t_start = local_clock()

                while local_clock() < t_start + duration:
                        fetch_tic = time.time()
                        fetched = self.stream.get_data(fetch_secs, picks=self.picks)[0]
                        acq_delay = time.time() - fetch_tic
                        if estimate_delays:
                                acq_delays.append(acq_delay)

                        if fetched is None or fetched.size == 0:
                                time.sleep(0.001)
                                continue

                        # extend ring buffer and keep only max_buf_len samples
                        ring_buffer = np.concatenate((ring_buffer, fetched), axis=1)
                        if ring_buffer.shape[1] > max_buf_len:
                                ring_buffer = ring_buffer[:, -max_buf_len:]

                        # process windows
                        while ring_buffer.shape[1] >= self.window_size_s:
                                window = ring_buffer[:, :self.window_size_s].copy()

                                # optional artifact correction
                                if self.artifact_correction:
                                        art_tic = time.time()
                                        if self.artifact_correction == "lms":
                                                window = remove_blinks_lms(
                                                        window, ref_ch_idx=ref_ch_idx, n_taps=5, mu=0.01
                                        )
                                        elif self.artifact_correction == "orica":
                                                sources = self.orica.transform(window)
                                                blink_score = np.abs(
                                                        np.dot(
                                                        sources.T, self.blink_template
                                                        )
                                                        / (
                                                        np.linalg.norm(sources, axis=1)
                                                        * np.linalg.norm(self.blink_template)
                                                        + 1e-12
                                                        )
                                                )
                                                blink_idx = np.argmax(blink_score)
                                                if blink_score[blink_idx] > 0.7:
                                                        sources[blink_idx, :] = 0
                                                window = self.orica.inverse_transform(sources)
                                if estimate_delays:
                                        artifact_delays.append(time.time() - art_tic)

                                # compute neurofeedback features
                                for idx, mod in enumerate(mods):
                                        meth_tic = time.time()
                                        nf_data_, method_delay = nf_mods[idx](window, **precomps[idx])
                                        nf_data[mod].append(nf_data_)
                                        if estimate_delays:
                                                method_delays[mod].append(
                                                        {
                                                        "measured": time.time() - meth_tic,
                                                        "reported": method_delay,
                                                        }
                                                )

                                # update real-time plot
                                if self.show_nf_signal:
                                        now = time.time()
                                        if (now - last_plot_time) >= plot_interval:
                                                plot_tic = time.time()
                                                last_vals = [nf_data[key][-1] for key in mods]
                                                self.update_nf_plot(last_vals, labels=mods)
                                                self.app.processEvents()
                                                if estimate_delays:
                                                        plot_delays.append(time.time() - plot_tic)
                                                last_plot_time = now

                                if self.show_design_viz:
                                        plot_design(nf_data_)

                                ring_buffer = ring_buffer[:, hop_samples:]

                # store delays if requested
                if estimate_delays:
                        self.acq_delays = acq_delays
                        self.artifact_delays = artifact_delays
                        self.method_delays = method_delays
                        self.plot_delays = plot_delays

                return nf_data
        
        @property
        def modality_params(self) -> dict:
                """
                Current modality parameters.

                Returns
                -------
                dict
                        Dictionary with the current neural-feature modality parameters.
                """
                return self._modality_params

        @modality_params.setter
        def modality_params(self, params: dict | None) -> None:
                """
                Set the modality parameters for neural-feature extraction.

                Parameters
                ----------
                params : dict | None
                        Modality-specific parameters. If None, an empty dict is used.

                Raises
                ------
                ValueError
                        If ``params`` is neither a dict nor None.
                """
                if params is not None and not isinstance(params, dict):
                        raise ValueError("modality_params must be a dict or None.")
                self._modality_params = params or {}

        def get_default_params(self) -> dict:
                """
                Default parameters defined in the configuration YAML.

                Returns
                -------
                dict
                        Default parameters for all neural-feature modalities.
                """
                return self._default_params
        
        ## --------------------------- General Methods --------------------------- ##

        def update_nf_plot(self, new_vals, labels=None):
                """Update the neurofeedback plot in real time.

                Parameters
                ----------
                new_vals : list of float | ndarray
                        Latest neural-feature values for each modality,
                        e.g. ``[sensor_power_last, band_ratio_last]``.
                labels : list of str | None
                        Labels for each modality to display on the y-axis.
                        Example: ``["Sensor Power", "Band Ratio"]``.

                Notes
                -----
                - On the first call the plot data structures and y-axis ticks
                are created; subsequent calls simply roll and update the data.
                - Values are normalised by the modality-specific scaling factors
                in ``self.scales_dict`` and vertically offset by ``self.shifts``
                so that multiple modalities can be visualised together.

                Examples
                --------
                >>> session.update_nf_plot([0.3, 0.7], labels=["Sensor Power", "Band Ratio"])
                """
                new_vals = np.asarray(new_vals, dtype=float)
                n_labels = len(new_vals)

                # Normalise and vertically offset each value
                scales = [self.scales_dict[k] for k in self._mods]
                norm_vals = [
                        shift + (val / scale) * ch_scale
                        for val, scale, shift, ch_scale in zip(
                        new_vals, scales, self.shifts, self.channel_scales
                        )
                ]
                norm_vals = np.asarray(norm_vals)

                # Initialise buffers and curves on first call
                if not hasattr(self, "plot_data"):
                        self.plot_data = np.zeros((n_labels, len(self.time_axis)))
                        self.curves = []
                        pens = [pg.mkPen(color=color, width=4) for color in self.colors_list]
                        for i in range(n_labels):
                                pen = pens[i % len(pens)]
                                curve = self.plot_widget.plot(
                                        self.time_axis, self.plot_data[i, :], pen=pen, name=labels[i]
                                )
                                self.curves.append(curve)

                        if self.text_items is None:
                                self.text_items = []
                                pretty_labels = [lbl.replace("_", " ").title() for lbl in labels]
                                yticks = list(zip(self.shifts, pretty_labels))
                                self.plot_widget.getAxis("left").setTicks([yticks])

                # Roll buffer and append new data
                self.plot_data = np.roll(self.plot_data, -1, axis=1)
                self.plot_data[:, -1] = norm_vals

                # Update curves
                for i, curve in enumerate(self.curves):
                        curve.setData(self.time_axis, self.plot_data[i, :])


        def scale_up(self, ch_idx):
                """Double the vertical scale for the given channel index."""
                self.channel_scales[ch_idx] *= 2


        def scale_down(self, ch_idx):
                """Halve the vertical scale for the given channel index."""
                self.channel_scales[ch_idx] /= 2


        def plot_rt(self, bufsize=0.2):
                """Visualise raw EEG signals from the LSL stream in real time.

                Parameters
                ----------
                bufsize : float
                        Buffer/window size in seconds for the StreamReceiver display.
                        Default is 0.2 s.

                Notes
                -----
                Uses :class:`mne_lsl.stream_viewer.StreamViewer` to create
                a real-time display of the EEG stream.

                Examples
                --------
                >>> session.plot_rt(bufsize=0.2)
                """
                Viewer(stream_name=self.stream.name).start(bufsize)
        
        ## --------------------------- Preprocessing & Source Estimation --------------------------- ##

        def get_blink_template(self, max_iter=800, method="infomax"):
                """
                Identify the eye-blink component from the baseline EEG and store its template.

                Parameters
                ----------
                max_iter : int, optional
                        Maximum iterations for ICA fitting. Default is 800.
                method : str, optional
                        ICA method to use (e.g., 'infomax'). Default is 'infomax'.

                Sets
                ----
                self.blink_template : np.ndarray
                        Spatial topography of the ICA component corresponding to eye blinks.

                Examples
                --------
                >>> session.get_blink_template(max_iter=1000, method="fastica")
                """
                self.blink_template = create_blink_template(
                        self.raw_baseline,
                        max_iter=max_iter,
                        method=method,
                )


        def compute_inv_operator(self):
                """
                Compute and save the inverse operator for source localization.

                Notes
                -----
                - Stores the operator in ``self.inv``.
                - Saves it under ``subject_dir/inv/visit_{visit}-inv.fif``.

                Examples
                --------
                >>> session.compute_inv_operator()
                """
                self.inv, self.fwd, self.noise_cov = \
                        _compute_inv_operator(
                                self.raw_baseline,
                                subject_fs_id=self.subject_fs_id,
                                subjects_fs_dir=self.subjects_fs_dir,
                        )
                write_inverse_operator(
                        fname=self.subject_dir / "inv" / f"visit_{self.visit}-inv.fif",
                        inv=self.inv,
                        overwrite=True,
                )
                write_forward_solution(
                        fname=self.subject_dir / "inv" / f"visit_{self.visit}-fwd.fif",
                        fwd=self.fwd,
                        overwrite=True,
                )
                write_cov(
                        fname=self.subject_dir / "inv" / f"visit_{self.visit}-cov.fif",
                        cov=self.noise_cov,
                        overwrite=True,
                )




        def run_orica(
                self,
                n_channels,
                learning_rate=0.1,
                block_size=256,
                online_whitening=True,
                calibrate_pca=False,
                forgetfac=1.0,
                nonlinearity="tanh",
                random_state=None,
                ):
                """
                Initialize an online ICA (ORICA) instance for real-time artifact removal.

                Parameters
                ----------
                n_channels : int
                        Number of EEG channels to include.
                learning_rate : float
                        Learning rate for online ICA updates.
                block_size : int
                        Number of samples processed per update.
                online_whitening : bool
                        Whether to apply online whitening to the data.
                calibrate_pca : bool
                        If True, calibrates PCA before running ICA.
                forgetfac : float
                        Forgetting factor for ORICA update (1.0 means no forgetting).
                nonlinearity : str
                        Nonlinearity function used in ICA (e.g., 'tanh', 'cube').
                random_state : int | None
                        Random seed for reproducibility.

                Notes
                -----
                After initialization, use ``self.orica.transform(data_chunk)`` to process data
                and optionally remove identified artifact components.

                Examples
                --------
                >>> session.run_orica(n_channels=64, learning_rate=0.05)
                """
                self.orica = ORICA(
                        n_channels=n_channels,
                        learning_rate=learning_rate,
                        block_size=block_size,
                        online_whitening=online_whitening,
                        calibrate_pca=calibrate_pca,
                        forgetfac=forgetfac,
                        nonlinearity=nonlinearity,
                        random_state=random_state,
                )
        
        ## --------------------------- 3D brain activation --------------------------- ##

        def initiate_brain_plot(self):
                """
                Initialize the 3D brain plot for real-time visualization.

                This method sets up the cortical surface mesh, scalar arrays, 
                and vertex indices for both hemispheres. It then initializes 
                the PyVista plotter to visualize brain activation.

                Notes
                -----
                - The `subjects_dir` currently points to a fixed FreeSurfer directory.
                Update this path as needed.
                - The method calls `setup_surface` to generate surface mesh data 
                and `setup_plotter` to create the PyVista plotter.

                Attributes Set
                ----------------
                self.hemi_offsets : dict
                        Hemisphere offsets for proper vertex indexing.
                self.scalars_full : ndarray
                        Array to store activation scalars for the full mesh.
                self.mesh : pyvista.PolyData
                        The cortical surface mesh.
                self.verts_stc : dict
                        Vertex indices corresponding to source estimates for each hemisphere.
                self.plotter : pyvista.Plotter
                        The PyVista plotter object for real-time updates.
                """
                subjects_dir = "/Applications/freesurfer/dev/subjects" # fix this later
                self.hemi_offsets, self.scalars_full, self.mesh, self.verts_stc = \
                                                setup_surface(subjects_dir, hemi_distance=100.0)
                self.plotter = setup_plotter(self.mesh)

        def plot_brain_activation(self, data, interval=0.05): 
                """
                Update the brain plot with real-time source activation.

                This method projects M/EEG data onto the cortical surface using an
                inverse operator, computes mean activation in time blocks for each 
                hemisphere, and updates the 3D brain visualization.

                Parameters
                ----------
                data : ndarray, shape (n_channels, n_times)
                        The raw EEG/MEG data to project and visualize.
                interval : float, default=0.05
                        Pause in seconds between rendering each block, controlling
                        the update speed of the visualization.

                Notes
                -----
                - Uses `RawArray` from MNE to wrap the data and set an average EEG
                reference if applicable.
                - Applies the inverse operator via `apply_inverse_raw` with
                `pick_ori='normal'`.
                - Data is divided into blocks for smoother visualization; each
                block is averaged and assigned to the mesh scalars.
                - PyVista mesh is updated in place and rendered in real-time.

                Raises
                ------
                RuntimeError
                        If `initiate_brain_plot` has not been called before this method.

                """
                raw_data = RawArray(data, self.rec_info)
                raw_data.set_eeg_reference("average", projection=True)
                stc = apply_inverse_raw(raw_data, self.inv, lambda2=1 / 9, pick_ori="normal")

                n_times = data.shape[1]
                block = int(data.shape[1] / 2)
                n_blocks = n_times // block

                for b in range(n_blocks):
                        for hemi in ["lh", "rh"]:
                                verts = self.verts_stc[hemi] + self.hemi_offsets[hemi]
                                if hemi == "rh": block_mean = stc.rh_data[:, b*block:(b+1)*block].mean(axis=1)
                                if hemi == "lh": block_mean = stc.lh_data[:, b*block:(b+1)*block].mean(axis=1)
                                self.scalars_full[verts] = block_mean

                        self.mesh["activity"] = self.scalars_full
                        self.mesh.Modified()
                        self.plotter.update_scalars(self.mesh["activity"], render=True)
                        time.sleep(interval)

                #self.plotter.show(auto_close=False)

        ## --------------------------- Data I/O & Session Management --------------------------- ##

        def save(
                self,
                nf_data=True,
                acq_delay=True,
                artifact_delay=True,
                method_delay=True,
                raw_data=False,
                format="json",
                ):
                """
                Save neurofeedback session data to disk.

                Disconnects the LSL stream and writes neurofeedback results, delay metrics,
                and (optionally) raw EEG data to the subject directory.

                Parameters
                ----------
                nf_data : bool, optional
                        If True, save the neurofeedback data. Default is True.
                acq_delay : bool, optional
                        If True, save acquisition delay data. Default is True.
                artifact_delay : bool, optional
                        If True, save artifact-related delay data. Default is True.
                method_delay : bool, optional
                        If True, save method delay data. Default is True.
                raw_data : bool, optional
                        If True, save raw EEG data. (Not implemented.) Default is False.
                format : str, optional
                        File format to save the data. Currently only ``'json'`` is supported. Default is ``'json'``.

                Raises
                ------
                NotImplementedError
                        If ``raw_data=True``, since saving raw EEG data is not yet implemented.

                Notes
                -----
                - Creates the folders ``neurofeedback``, ``delays``, ``main``, and ``reports``
                inside ``self.subject_dir`` if they do not exist.
                - JSON files are named using the subject ID and visit number.

                Examples
                --------
                >>> session.save(nf_data=True, acq_delay=False, raw_data=False)
                """
                self.stream.disconnect()
                for folder in ["neurofeedback", "delays", "main", "reports"]:
                        (self.subject_dir / folder).mkdir(parents=True, exist_ok=True)

                if format == "json":
                        if nf_data:
                                fname = self.subject_dir / "neurofeedback" / f"nf_data_visit_{self.visit}.json"
                                with open(fname, "w") as file:
                                        json.dump(self.nf_data, file)
                        if acq_delay:
                                fname = self.subject_dir / "delays" / f"acq_delay_visit_{self.visit}.json"
                                with open(fname, "w") as file:
                                        json.dump(self.acq_delays, file)
                        if artifact_delay:
                                fname = self.subject_dir / "delays" / f"artifact_delay_visit_{self.visit}.json"
                                with open(fname, "w") as file:
                                        json.dump(self.artifact_delays, file)
                        if method_delay:
                                fname = self.subject_dir / "delays" / f"method_delay_visit_{self.visit}.json"
                                with open(fname, "w") as file:
                                        json.dump(self.method_delays, file)

                if raw_data:
                        raise NotImplementedError("Saving raw_data is not implemented yet.")
        
        ## --------------------------- Reporting --------------------------- ##

        def create_report(self, overwrite: bool = True) -> None:
                """
                Create an HTML report summarizing the neurofeedback session.

                The report includes baseline recordings, EEG sensor visualizations,
                and—if applicable—brain-label figures for source-based modalities.

                Parameters
                ----------
                overwrite : bool, optional
                        If True (default), overwrite an existing report file with the same name.

                Notes
                -----
                - Baseline raw data is added without PSD or butterfly plots.
                - For sensor-space modalities, both 2-D topomap and 3-D sensor layouts are shown.
                - For source-space modalities, brain-label figures are added via
                :func:`plot_glass_brain`.
                - The HTML file is saved to ``<subject_dir>/reports`` with a name containing
                the subject ID, visit number, and modality.

                Examples
                --------
                >>> session.create_report(overwrite=True)
                """
                report = Report(title=f"Neurofeedback Session with {self.modality} modality")
                report.add_raw(self.raw_baseline, title="Baseline recording", psd=False, butterfly=False)

                # Handle modality as a list without mutating the original attribute
                modalities = [self.modality] if isinstance(self.modality, str) else list(self.modality)

                for mod in modalities:
                        if "source" not in mod:
                                # Mark bads for plotting
                                bads = self.picks if self.picks is not None else self.rec_info["ch_names"]
                                self.rec_info["bads"].extend(bads)

                                fig_sensors = plt.figure(figsize=(10, 5))
                                ax1 = fig_sensors.add_subplot(121)
                                ax2 = fig_sensors.add_subplot(122, projection="3d")
                                mne.viz.plot_sensors(info=self.rec_info, kind="topomap", axes=ax1, show=False)
                                mne.viz.plot_sensors(info=self.rec_info, kind="3d", axes=ax2, show=False)
                                ax2.axis("off")

                                # Reset bad channels
                                self.rec_info["bads"] = []
                                report.add_figure(fig=fig_sensors, title="Sensors")

                        else:
                                if mod in {"source_connectvity", "source_graph"}:
                                        fig_brain = plot_glass_brain(bl1=self.params["brain_label"], bl2=None)
                                else:
                                        fig_brain = plot_glass_brain(
                                        bl1=self.params["brain_label_1"],
                                        bl2=self.params["brain_label_2"],
                                        )
                                report.add_figure(fig=fig_brain, title="Selected brain labels")

                report_fname = (
                        f"subject_{self.subject_id}_visit_{self.visit}_modality_{'_'.join(modalities)}.html"
                )
                report.save(self.subject_dir / "reports" / report_fname, overwrite=overwrite)

        ## --------------------------- Neural Feature Extraction Methods (preparation) --------------------------- ##

        def _sensor_power_prep(self) -> dict:
                """
                Prepare parameters for the ``sensor_power`` neural feature modality.

                Returns
                -------
                dict
                        Precomputed parameters:
                        - ``sfreq`` : float
                        Sampling frequency (Hz).
                        - ``frange`` : tuple of float
                        Frequency range for power computation.
                        - ``method`` : str
                        Method used to compute power.
                        - ``relative`` : bool
                        Whether to compute relative power.
                """
                return {
                        "sfreq": self.rec_info["sfreq"],
                        "frange": self.params["frange"],
                        "method": self.params["method"],
                        "relative": self.params["relative"],
                }

        def _argmax_freq_prep(self) -> dict:
                """
                Prepare parameters for the ``argmax_freq`` modality by extracting aperiodic components.

                Returns
                -------
                dict
                        Precomputed parameters:
                        - ``fft_window`` : ndarray
                        Hann window for FFT computation.
                        - ``freq_band_mask`` : ndarray of bool
                        Mask selecting frequencies in the requested band.
                        - ``freqs_band`` : ndarray
                        Frequencies within the selected band.
                        - ``ap_model`` : ndarray
                        Estimated aperiodic component.
                        - ``gaussian`` : callable
                        Gaussian function ``f(x, a, mu, sigma)``.

                Raises
                ------
                AssertionError
                        If no baseline recording is available (``self.raw_baseline``).
                """
                assert hasattr(self, "raw_baseline"), (
                        "Baseline recording must be completed before calling _argmax_freq_prep()."
                )

                # Estimate the aperiodic component from the baseline recording
                ap_params, _ = estimate_aperiodic_component(
                        raw_baseline=self.raw_baseline,
                        picks=self.picks,
                        method=self.params["method"],
                )

                # FFT preparation
                n_samples = int(self.winsize * self._sfreq)
                fft_window = np.hanning(n_samples)
                freqs = np.fft.rfftfreq(n_samples, d=1 / self.rec_info["sfreq"])
                freq_band_mask = (freqs >= self.params["frange"][0]) & (freqs <= self.params["frange"][1])
                freqs_band = freqs[freq_band_mask]

                # Aperiodic model and Gaussian peak function
                ap_model = (10 ** ap_params[0]) / (freqs_band ** ap_params[1])

                def gaussian(x: np.ndarray, a: float, mu: float, sigma: float) -> np.ndarray:
                        return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

                return {
                        "fft_window": fft_window,
                        "freq_band_mask": freq_band_mask,
                        "freqs_band": freqs_band,
                        "ap_model": ap_model,
                        "gaussian": gaussian,
                }

        def _band_ratio_prep(self) -> dict:
                """
                Prepare parameters for the ``band_ratio`` neural feature modality.

                Returns
                -------
                dict
                        Precomputed parameters:
                        - ``sfreq`` : float
                        Sampling frequency (Hz).
                        - ``frange_1`` : tuple of float
                        Frequency range for numerator band.
                        - ``frange_2`` : tuple of float
                        Frequency range for denominator band.
                        - ``method`` : str
                        Method for computing band power.
                """
                return {
                        "sfreq": self.rec_info["sfreq"],
                        "frange_1": self.params["frange_1"],
                        "frange_2": self.params["frange_2"],
                        "method": self.params["method"],
                }


        def _individual_peak_power_prep(self) -> dict:
                """
                Prepare parameters for the 'individual_peak_power' modality.

                Computes the individual peak frequency from baseline data and
                returns the center frequency together with auxiliary parameters.

                Returns
                -------
                dict
                        Dictionary containing:
                        - sfreq : float
                        Sampling frequency.
                        - freq_var : float
                        Half-width (Hz) around the center frequency to use when
                        selecting the frequency band.
                        - cf : float
                        Center frequency (Hz) of the individual peak.

                Notes
                -----
                If multiple peaks are found in the selected frequency range,
                the center frequency is set to the midpoint of the range and a
                warning is issued.

                Requires that a baseline recording has been performed.
                """
                _, peak_params_ = estimate_aperiodic_component(
                        raw_baseline=self.raw_baseline,
                        picks=self.picks,
                        method=self.params["method"],
                )

                peak_params = [
                        p[0] for p in peak_params_
                        if self.params["frange"][0] < p[0] < self.params["frange"][1]
                ]
                if len(peak_params) == 1:
                        cf = peak_params[0]
                else:
                        cf = (self.params["frange"][0] + self.params["frange"][1]) / 2
                        warn(
                        "Center frequency set to the midpoint of the selected "
                        "frequency range because multiple peaks were found."
                        )

                return dict(sfreq=self._sfreq, freq_var=2, cf=cf)

        def _entropy_prep(self) -> dict:
                """
                Prepare parameters for the 'entropy' neural feature modality.

                Returns
                -------
                dict
                        Dictionary containing:
                        - sos : ndarray
                        Second-order sections of the Butterworth band-pass filter.
                        - method : str
                        Entropy computation method.
                        - psd_method : str
                        PSD estimation method.
                """
                sos = butter_bandpass(
                        self.params["frange"][0],
                        self.params["frange"][1],
                        self._sfreq,
                        order=5,
                )
                return dict(
                        sos=sos,
                        method=self.params["method"],
                        psd_method=self.params["psd_method"],
                )


        def _source_power_prep(self) -> dict:
                """
                Prepare parameters for the 'source_power' modality.

                Returns
                -------
                dict
                        Dictionary containing:
                        - fft_window : ndarray
                        FFT window to apply.
                        - freq_band_idxs : ndarray
                        Indices of the selected frequency band.
                        - brain_label : mne.Label
                        Target brain label.
                        - inverse_operator : mne.minimum_norm.InverseOperator
                        Inverse operator for source localization.

                Notes
                -----
                Reads the inverse operator from the subject's visit folder.
                Requires that the subject FreeSurfer ID and directory are set.
                """
                fft_window, _, freq_band_idxs, _ = compute_fft(
                        sfreq=self._sfreq,
                        winsize=self.winsize,
                        freq_range=self.params["frange"],
                )

                bls = read_labels_from_annot(
                        subject=self.subject_fs_id,
                        parc=self.params["atlas"],
                        subjects_dir=self.subjects_fs_dir,
                )
                bl_names = [bl.name for bl in bls]
                bl_idx = bl_names.index(self.params["brain_label"])
                brain_label = bls[bl_idx]

                if self.params["method"] in ["MNE", "dSPM", "sLORETA", "eLORETA"]:
                        inv_fname = self.subject_dir / "inv" / f"visit_{self.visit}-inv.fif"
                        inverse_operator = read_inverse_operator(fname=inv_fname)


                if self.params["method"] == "LCMV":
                        inverse_operator = make_lcmv(
                                                self.rec_info,
                                                self.fwd,
                                                self.noise_cov,
                                                reg=0.05,
                                                pick_ori="max-power",
                                                weight_norm="unit-noise-gain",
                                                rank=None,
                                        )

                return dict(
                        fft_window=fft_window,
                        freq_band_idxs=freq_band_idxs,
                        brain_label=brain_label,
                        inverse_operator=inverse_operator,
                        method=self.params["method"],
                )

        def _sensor_connectivity_prep(self) -> dict:
                """
                Prepare parameters for the 'sensor_connectivity' modality.

                Returns
                -------
                dict
                        Dictionary containing:
                        - indices : tuple of ndarray
                        Channel index pairs for connectivity computation.
                        - freqs : ndarray
                        Frequencies (Hz) at which to compute connectivity.
                        - fmin : float
                        Lower bound of the frequency range (Hz).
                        - fmax : float
                        Upper bound of the frequency range (Hz).
                        - mode : str
                        Connectivity mode (e.g., 'multitaper', 'fourier').
                        - method : str
                        Connectivity metric (e.g., 'coh', 'pli').
                """
                ch_names = self.rec_info["ch_names"]
                chs = self.params["channels"]
                indices = tuple(
                        np.array([ch_names.index(ch1), ch_names.index(ch2)])
                        for ch1, ch2 in zip(chs[0], chs[1])
                )

                freq_res = 6
                freqs = np.linspace(self.params["frange"][0],
                                        self.params["frange"][1],
                                        freq_res)

                return dict(
                        indices=indices,
                        freqs=freqs,
                        fmin=self.params["frange"][0],
                        fmax=self.params["frange"][1],
                        mode=self.params["mode"],
                        method=self.params["method"],
                )

        def _source_connectivity_prep(self) -> dict:
                """
                Prepare parameters for the 'source_connectivity' modality.

                Returns
                -------
                dict
                        Dictionary containing:
                        - merged_label : mne.Label
                        Combined label of left- and right-hemisphere regions.
                        - inverse_operator : mne.minimum_norm.InverseOperator
                        Inverse operator for source localization.
                        - freqs : ndarray
                        Frequencies (Hz) at which to compute connectivity.

                Raises
                ------
                AssertionError
                        If the provided brain labels are not from the correct
                        hemispheres ('lh' for left, 'rh' for right).
                """
                assert self.params["brain_label_1"].endswith("lh"), \
                        "First brain label must be from the left hemisphere."
                assert self.params["brain_label_2"].endswith("rh"), \
                        "Second brain label must be from the right hemisphere."

                bls = read_labels_from_annot(
                        subject=self.subject_fs_id,
                        parc=self.params["atlas"],
                        subjects_dir=self.subjects_fs_dir,
                )
                bl_names = [bl.name for bl in bls]
                merged_label = (
                        bls[bl_names.index(self.params["brain_label_1"])] +
                        bls[bl_names.index(self.params["brain_label_2"])]
                )

                inv_fname = self.subject_dir / "inv" / f"visit_{self.visit}-inv.fif"
                inverse_operator = read_inverse_operator(fname=inv_fname)

                freq_res = 6
                freqs = np.linspace(self.params["frange"][0],
                                        self.params["frange"][1],
                                        freq_res)

                return dict(
                        merged_label=merged_label,
                        inverse_operator=inverse_operator,
                        freqs=freqs,
                )

        def _sensor_graph_prep(self) -> dict:
                """
                Prepare parameters for the 'sensor_graph' modality.

                Returns
                -------
                dict
                        Dictionary containing:
                        - indices : tuple of ndarray
                        Channel index pairs for graph computation.
                        - sos : ndarray
                        Second-order sections of the Butterworth band-pass filter.
                        - dist_type : str
                        Distance metric for graph computation.
                        - alpha : float
                        Alpha parameter for graph weighting.
                        - beta : float
                        Beta parameter for graph weighting.
                """
                ch_names = self.rec_info["ch_names"]
                chs = self.params["channels"]
                indices = tuple(
                        np.array([ch_names.index(ch1), ch_names.index(ch2)])
                        for ch1, ch2 in zip(chs[0], chs[1])
                )

                sos = butter_bandpass(
                        self.params["frange"][0],
                        self.params["frange"][1],
                        self._sfreq,
                        order=5,
                )

                return dict(
                        indices=indices,
                        sos=sos,
                        dist_type=self.params["dist_type"],
                        alpha=self.params["alpha"],
                        beta=self.params["beta"],
                )

        def _source_graph_prep(self) -> dict:
                """
                Prepare parameters for the 'source_graph' modality.

                Returns
                -------
                dict
                        Dictionary containing:
                        - bls : list of mne.Label
                        All labels from the specified atlas.
                        - bl_idxs : tuple of int
                        Indices of the two selected brain labels for graph computation.
                        - inverse_operator : mne.minimum_norm.InverseOperator
                        Inverse operator for source localization.
                        - sos : ndarray
                        Second-order sections of the Butterworth band-pass filter.
                """
                bls = read_labels_from_annot(
                        subject=self.subject_fs_id,
                        parc=self.params["atlas"],
                        subjects_dir=self.subjects_fs_dir,
                )
                bl_names = [bl.name for bl in bls]
                bl_idxs = (
                        bl_names.index(self.params["brain_label_1"]),
                        bl_names.index(self.params["brain_label_2"]),
                )

                inv_fname = self.subject_dir / "inv" / f"visit_{self.visit}-inv.fif"
                inverse_operator = read_inverse_operator(fname=inv_fname)

                sos = butter_bandpass(
                        self.params["frange"][0],
                        self.params["frange"][1],
                        self._sfreq,
                        order=5,
                )

                return dict(
                        bls=bls,
                        bl_idxs=bl_idxs,
                        inverse_operator=inverse_operator,
                        sos=sos,
                )

        def _sensor_cfc_prep(self) -> dict:

                comod = Comodulogram(
                                        fs=self._sfreq,
                                        low_fq_range=np.linspace(self.params["frange_1"][0], self.params["frange_1"][1], 5),
                                        high_fq_range=np.linspace(self.params["frange_2"][0], self.params["frange_2"][1], 5),
                                        method=self.params["method"],
                                        n_surrogates=0
                                )
                return comod


        ## --------------------------- Neural Feature Extraction Methods (main) --------------------------- ##

        @timed
        def _sensor_power(
                self,
                data: np.ndarray,
                sfreq: float,
                frange: tuple[float, float],
                method: str = "welch",
                relative: bool = False,
                ) -> float:
                """Compute mean sensor-level power in a given frequency band.

                Parameters
                ----------
                data : ndarray, shape (n_channels, n_samples)
                        EEG data.
                sfreq : float
                        Sampling frequency in Hz.
                frange : tuple of float
                        Frequency range of interest (fmin, fmax).
                method : {'fft', 'periodogram', 'welch', 'multitaper'}
                        PSD estimation method.
                relative : bool
                        If True, return power relative to total broadband power.

                Returns
                -------
                float
                        Mean power across channels in the selected band.
                """
                if method == "fft":
                        n_channels, n_samples = data.shape
                        n_fft = int(2 ** np.ceil(np.log2(n_samples)))
                        win = get_window("hann", n_samples, fftbins=True)
                        data_win = data * win
                        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sfreq)
                        psd = (np.abs(np.fft.rfft(data_win, n=n_fft, axis=1)) ** 2) / (
                        sfreq * np.sum(win ** 2)
                        )
                elif method == "periodogram":
                        freqs, psd = periodogram(data, sfreq, axis=1)
                elif method == "welch":
                        logger.info("Estimating PSD using Welch method.")
                        freqs, psd = welch(data, sfreq, axis=1)
                elif method == "multitaper":
                        psd, freqs = psd_array_multitaper(data, sfreq, axis=1)
                else:
                        raise ValueError(f"Unknown method: {method}")

                mask = (freqs >= frange[0]) & (freqs <= frange[1])
                freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
                band_power = simpson(psd[:, mask], dx=freq_res, axis=1)

                if relative:
                        total_power = simpson(psd, dx=freq_res, axis=1)
                        band_power = band_power / total_power

                return float(band_power.mean())


        @timed
        def _argmax_freq(
                self,
                data: np.ndarray,
                fft_window: np.ndarray,
                ap_model: np.ndarray,
                gaussian: callable,
                ) -> float:
                """Estimate the individual peak frequency using FFT and a Gaussian fit.

                Parameters
                ----------
                data : ndarray, shape (n_channels, n_samples)
                        EEG data.
                fft_window : ndarray
                        Precomputed FFT window (1-D Hann or similar).
                ap_model : ndarray, shape (n_freqs,)
                        Estimated aperiodic (1/f) component to subtract.
                gaussian : callable
                        Gaussian function: f(x, amp, mean, std) -> y.

                Returns
                -------
                float
                        Estimated peak frequency (Hz). Returns 0 if Gaussian fit fails.
                """
                data_win = data * fft_window
                fftval = np.abs(np.fft.rfft(data_win, axis=1) / data.shape[-1])
                freqs = np.fft.rfftfreq(data.shape[-1], d=1.0 / self._sfreq)

                mask = (freqs >= self.params["frange"][0]) & (freqs <= self.params["frange"][1])
                freqs_band = freqs[mask]

                total_power_band = np.mean(np.square(fftval[:, mask]), axis=0)
                periodic_power = total_power_band - ap_model

                # Initial guess: amplitude, frequency of max power, std=1.0
                p0 = [periodic_power.max(),
                        freqs_band[np.argmax(periodic_power)],
                        1.0]

                try:
                        popt, _ = curve_fit(gaussian, freqs_band, periodic_power, p0=p0)
                        individual_peak = float(popt[1])
                        logger.info(f"Individual peak frequency: {individual_peak:.2f} Hz")
                except RuntimeError:
                        individual_peak = 0.0
                        warn("Gaussian fit failed; individual peak set to 0 Hz.", RuntimeWarning)

                return individual_peak


        @timed
        def _band_ratio(
                self,
                data: np.ndarray,
                sfreq: float,
                frange_1: tuple[float, float],
                frange_2: tuple[float, float],
                method: str = "welch",
                ) -> float:
                """Compute the ratio of band power between two frequency bands.

                Parameters
                ----------
                data : ndarray, shape (n_channels, n_samples)
                        EEG data.
                sfreq : float
                        Sampling frequency in Hz.
                frange_1 : tuple of float
                        Frequency range for numerator band (fmin, fmax).
                frange_2 : tuple of float
                        Frequency range for denominator band (fmin, fmax).
                method : {'fft', 'periodogram', 'welch', 'multitaper'}
                        PSD estimation method.

                Returns
                -------
                float
                        Ratio of mean power in `frange_1` to mean power in `frange_2`.
                """
                if method == "fft":
                        n_channels, n_samples = data.shape
                        n_fft = int(2 ** np.ceil(np.log2(n_samples)))
                        win = get_window("hann", n_samples, fftbins=True)
                        data_win = data * win
                        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sfreq)
                        psd = (np.abs(np.fft.rfft(data_win, n=n_fft, axis=1)) ** 2) / (
                        sfreq * np.sum(win ** 2)
                        )
                elif method == "periodogram":
                        freqs, psd = periodogram(data, sfreq, axis=1)
                elif method == "welch":
                        freqs, psd = welch(data, sfreq, axis=1)
                elif method == "multitaper":
                        psd, freqs = psd_array_multitaper(data, sfreq, axis=1)
                else:
                        raise ValueError(f"Unknown method: {method}")

                mask1 = (freqs >= frange_1[0]) & (freqs <= frange_1[1])
                mask2 = (freqs >= frange_2[0]) & (freqs <= frange_2[1])
                freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

                bp1 = simpson(psd[:, mask1], dx=freq_res, axis=1)
                bp2 = simpson(psd[:, mask2], dx=freq_res, axis=1)

                return float(bp1.mean() / bp2.mean())

        @timed
        def _individual_peak_power(
                self,
                data: np.ndarray,
                sfreq: float,
                freq_var: float,
                cf: float,
                ) -> float:
                """Compute power around the individual peak frequency.

                Parameters
                ----------
                data : ndarray, shape (n_channels, n_samples)
                        EEG data.
                sfreq : float
                        Sampling frequency in Hz.
                freq_var : float
                        Half-width of frequency window around the peak frequency.
                cf : float
                        Center (peak) frequency.

                Returns
                -------
                float
                        Mean power across channels in the selected individual frequency band.
                """
                freqs, psd = welch(data, sfreq, axis=1)
                mask = (freqs >= cf - freq_var) & (freqs <= cf + freq_var)
                freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
                band_power = simpson(psd[:, mask], dx=freq_res, axis=1)

                logger.info(f"Individual peak band power: {band_power.mean():.4f}")
                return float(band_power.mean())


        @timed
        def _entropy(
                self,
                data: np.ndarray,
                sos: np.ndarray,
                method: str,
                psd_method: str | None = None,
                ) -> float:
                """Compute entropy of EEG signals.

                Parameters
                ----------
                data : ndarray, shape (n_channels, n_samples)
                        EEG data.
                sos : ndarray
                        Second-order sections of the bandpass filter.
                method : {'AppEn', 'SampEn', 'Spectral', 'SVD'}
                        Entropy method to use.
                psd_method : str | None
                        Method for computing PSD (used for spectral entropy).

                Returns
                -------
                float
                        Mean entropy value across channels.
                """
                data_filt = sosfiltfilt(sos, data)

                if method == "AppEn":
                        ents = compute_app_entropy(data_filt)
                elif method == "SampEn":
                        ents = compute_samp_entropy(data_filt)
                elif method == "Spectral":
                        ents = compute_spect_entropy(sfreq=self._sfreq, data=data_filt, psd_method=psd_method)
                elif method == "SVD":
                        ents = compute_svd_entropy(data_filt)
                else:
                        raise ValueError(f"Unknown entropy method: {method}")

                return float(ents.mean() - 2)


        @timed
        def _source_power(
                self,
                data: np.ndarray,
                fft_window: np.ndarray,
                freq_band_idxs: np.ndarray,
                brain_label,
                inverse_operator,
                method,
                ) -> float:
                """Compute source-level power for a specific brain label.

                Parameters
                ----------
                data : ndarray, shape (n_channels, n_samples)
                        EEG data.
                fft_window : ndarray
                        Precomputed FFT window.
                freq_band_idxs : ndarray
                        Indices of the frequency band of interest.
                brain_label : mne.Label
                        Label of the brain region to extract source activity from.
                inverse_operator : mne.minimum_norm.InverseOperator
                        Inverse operator for source reconstruction.

                Returns
                -------
                float
                        Mean power in the selected source region.
                """
                raw_data = RawArray(data, self.rec_info)
                raw_data.set_eeg_reference("average", projection=True)


                if method in ["MNE", "dSPM", "sLORETA", "eLORETA"]:
                        stc_data = apply_inverse_raw(
                                raw_data,
                                inverse_operator,
                                lambda2=1.0 / 9,
                                method=method,
                                pick_ori="normal",
                                label=brain_label,
                                ).data
                        
                if method == "LCMV":
                        stc_data = apply_lcmv_raw(raw_data, inverse_operator).data

                stc_data *= fft_window
                fft_val = np.abs(np.fft.rfft(stc_data, axis=1) / stc_data.shape[-1])
                power = np.mean(np.square(fft_val[:, freq_band_idxs]))
                print(power)

                return float(power)


        @timed
        def _sensor_connectivity(
                self,
                data: np.ndarray,
                indices: tuple[np.ndarray, ...],
                freqs: np.ndarray,
                fmin: float,
                fmax: float,
                mode: str,
                method: str,
                ) -> float:
                """Compute sensor-level connectivity between channel pairs.

                Parameters
                ----------
                data : ndarray, shape (n_channels, n_samples)
                        EEG data.
                indices : tuple of arrays
                        Channel index pairs for connectivity computation.
                freqs : ndarray
                        Frequencies at which to compute connectivity.
                fmin : float
                        Minimum frequency of interest.
                fmax : float
                        Maximum frequency of interest.
                mode : str
                        Connectivity mode (e.g., 'coh', 'pli').
                method : str
                        Method to compute connectivity (e.g., 'multitaper').

                Returns
                -------
                float
                        Mean connectivity across selected channel pairs.
                """
                con = spectral_connectivity_time(
                        data=data[np.newaxis, :],
                        freqs=freqs,
                        indices=indices,
                        average=False,
                        sfreq=self._sfreq,
                        fmin=fmin,
                        fmax=fmax,
                        faverage=True,
                        mode=mode,
                        method=method,
                        n_cycles=5,
                )

                con_data = np.squeeze(con.get_data(output="dense"))[indices].mean()
                return float(con_data)

        @timed
        def _source_connectivity(
                self,
                data: np.ndarray,
                merged_label,
                inverse_operator,
                freqs: np.ndarray,
                ) -> float:
                """Compute source-level connectivity between two brain regions.

                Parameters
                ----------
                data : ndarray, shape (n_channels, n_samples)
                        EEG data.
                merged_label : mne.Label
                        Merged label covering the two brain regions of interest.
                inverse_operator : mne.minimum_norm.InverseOperator
                        Inverse operator for source reconstruction.
                freqs : ndarray
                        Frequencies at which to compute connectivity.

                Returns
                -------
                float
                        Connectivity value between the two source regions.
                """
                raw_data = RawArray(data, self.rec_info)
                raw_data.set_eeg_reference("average", projection=True)

                stcs = apply_inverse_raw(
                        raw_data,
                        inverse_operator,
                        lambda2=1.0 / 9,
                        pick_ori="normal",
                        label=merged_label,
                )

                stc_lh_data = stcs.lh_data.mean(axis=0)
                stc_rh_data = stcs.rh_data.mean(axis=0)

                con = spectral_connectivity_time(
                        data=np.array([[stc_lh_data, stc_rh_data]]),
                        freqs=freqs,
                        indices=None,
                        average=False,
                        sfreq=self._sfreq,
                        fmin=self.params["frange"][0],
                        fmax=self.params["frange"][1],
                        faverage=True,
                        mode=self.params["mode"],
                        method=self.params["method"],
                        n_cycles=5,
                )

                con_data = float(np.squeeze(con.get_data(output="dense"))[1][0])
                return con_data


        @timed
        def _sensor_graph(
                self,
                data: np.ndarray,
                indices: tuple[np.ndarray, ...],
                sos: np.ndarray,
                dist_type: str,
                alpha: float,
                beta: float,
                ) -> float:
                """Compute graph-theoretical metrics from sensor-level EEG data.

                Parameters
                ----------
                data : ndarray, shape (n_channels, n_samples)
                        EEG data.
                indices : tuple of arrays
                        Pairs of channel indices to compute graph edges.
                sos : ndarray
                        Second-order sections of Butterworth bandpass filter.
                dist_type : str
                        Distance metric for graph computation.
                alpha : float
                        Alpha parameter for graph computation.
                beta : float
                        Beta parameter for graph computation.

                Returns
                -------
                float
                        Average edge value of the computed graph across selected channel pairs.
                """
                data_filt = sosfiltfilt(sos, data)
                graph_matrix = log_degree_barrier(data_filt, dist_type=dist_type, alpha=alpha, beta=beta)
                avg_edge = np.mean([graph_matrix[idxs] for idxs in indices]) - 0.025
                return float(avg_edge)


        @timed
        def _source_graph(
                self,
                data: np.ndarray,
                bls: list,
                bl_idxs: tuple[int, int],
                inverse_operator,
                sos: np.ndarray,
                ) -> float:
                """Compute graph-theoretical metrics from source-level EEG data.

                Parameters
                ----------
                data : ndarray, shape (n_channels, n_samples)
                        EEG data.
                bls : list of mne.Label
                        Labels of brain regions used to extract time courses.
                bl_idxs : tuple of int
                        Indices of the brain labels to compute the graph edge between.
                inverse_operator : mne.minimum_norm.InverseOperator
                        Inverse operator for source reconstruction.
                sos : ndarray
                        Second-order sections of Butterworth bandpass filter.

                Returns
                -------
                float
                        Average edge value of the computed graph between the two selected brain labels.
                """
                raw_data = RawArray(data, self.rec_info)
                raw_data.set_eeg_reference("average", projection=True)

                stcs = apply_inverse_raw(
                        raw_data,
                        inverse_operator,
                        lambda2=1 / 9.0,
                        pick_ori="normal",
                )

                tcs = stcs.extract_label_time_course(
                        bls,
                        src=inverse_operator["src"],
                        mode="mean_flip",
                        allow_empty=True,
                )

                tcs_filt = sosfiltfilt(sos, tcs)
                graph_matrix = log_degree_barrier(
                        tcs_filt,
                        dist_type=self.params["dist_type"],
                        alpha=self.params["alpha"],
                        beta=self.params["beta"],
                )

                avg_edge = float(graph_matrix[bl_idxs[0], bl_idxs[1]])
                return avg_edge

        @timed
        def _sensor_cfc(self,
                        data: np.ndarray,
                        comod
                        ) -> float:
                comod.fit(data)

                return comod.comod_.mean()