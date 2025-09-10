## don't enter here without a good guide!

import datetime
import json
import time
import uuid
from pathlib import Path
from warnings import warn
import threading

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import sosfiltfilt
import matplotlib.pyplot as plt

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from PyQt6.QtGui import QFont

import mne
from mne import set_log_level, read_labels_from_annot, Report
from mne.io import RawArray
from mne.channels import get_builtin_montages, read_dig_captrak
from mne.minimum_norm import apply_inverse_raw, read_inverse_operator, write_inverse_operator
from mne_connectivity import spectral_connectivity_time
from mne_features.univariate import (
                                        compute_app_entropy,
                                        compute_samp_entropy,
                                        compute_spect_entropy,
                                        compute_svd_entropy
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
        Initialize a subject session for EEG/Neurofeedback processing.

        This constructor validates inputs, sets up session configuration, and prepares
        logging. Optional preprocessing settings such as filtering and artifact correction
        can be specified.

        Parameters
        ----------
        subject_id : str
                Unique identifier for the subject. Must be a non-empty string.
        visit : int
                Visit number. Must be a positive integer (>=1).
        session : str
                Session type. Must be one of the valid sessions defined in `VALID_SESSIONS`.
        subjects_dir : str
                Path to the directory containing subject data.
        montage : str
                EEG montage. Can be a built-in montage or a path to a `.bvct` file.
        mri : bool
                Whether MRI data is available for the subject.
        subject_fs_id : str, optional
                FreeSurfer subject ID. Default is 'fsaverage'.
        subjects_fs_dir : str | None, optional
                Path to FreeSurfer subjects directory. Default is None.
        filtering : bool, optional
                Whether to apply band-pass filtering. Default is False.
        l_freq : float, optional
                Low frequency cutoff for filtering. Default is 1 Hz.
        h_freq : float, optional
                High frequency cutoff for filtering. Default is 40 Hz.
        artifact_correction : str | bool, optional
                Method for artifact correction. Must be one of `VALID_ARTIFACT_METHODS`. Default is False.
        ref_channel : str, optional
                Reference channel for EEG. Default is 'Fp1'.
        save_raw : bool, optional
                Whether to save raw EEG data. Default is True.
        save_nf_signal : bool, optional
                Whether to save the neurofeedback signal. Default is True.
        config_file : str | None, optional
                Path to a YAML config file. Default is project default config.
        verbose : int | None, optional
                Logging verbosity level. Default is None.

        Raises
        ------
        ValueError
                If any of the input parameters fail validation checks.

        Attributes
        ----------
        subject_id : str
                Stored subject ID.
        visit : int
                Stored visit number.
        session : str
                Stored session type.
        subjects_dir : str
                Stored path to subjects directory.
        montage : str
                Stored EEG montage.
        mri : bool
                MRI availability flag.
        subject_fs_id : str
                FreeSurfer subject ID.
        subjects_fs_dir : str | None
                FreeSurfer subjects directory path.
        filtering : bool
                Band-pass filtering flag.
        l_freq : float
                Low frequency cutoff.
        h_freq : float
                High frequency cutoff.
        artifact_correction : str | bool
                Artifact correction method ('orica' or 'lms').
        ref_channel : str
                Reference channel.
        save_raw : bool
                Save raw EEG flag.
        save_nf_signal : bool
                Save neurofeedback signal flag.
        config_file : str
                Path to the config file used.
        verbose : int | None
                Logging verbosity level.

        Examples
        --------
        >>> session = NFRealtime(
        ...     subject_id="sub01",
        ...     visit=1,
        ...     session="baseline",
        ...     subjects_dir="/data/subjects",
        ...     montage="standard_1020",
        ...     mri=True
        ... )
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
                subjects_fs_dir: str = None,
                filtering=False,
                l_freq=1,
                h_freq=40,
                artifact_correction=False,
                ref_channel="Fp1",
                save_raw: bool = True,
                save_nf_signal: bool = True,
                config_file=None,
                verbose=None,
                ):
                
                # --- Validation ---
                if not isinstance(subject_id, str) or not subject_id.strip():
                        raise ValueError("`subject_id` must be a non-empty string.")

                if not isinstance(visit, int) or visit < 1:
                        raise ValueError("`visit` must be a positive integer (>= 1).")

                if session not in self.VALID_SESSIONS:
                        raise ValueError(f"`session` must be one of {self.VALID_SESSIONS}, got {session!r}.")

                if not (
                        montage in get_builtin_montages()
                        or (montage.endswith(".bvct") and Path(montage).is_file())
                        ):
                        raise ValueError(
                                f"`montage` must be one of the built-in montages "
                                f"or a valid '.bvct' file path. Got {montage!r}."
                        )

                if not isinstance(subject_id, str) or not subject_id.strip():
                        raise ValueError("`subject_id` must be a non-empty string.")
                
                if not isinstance(mri, bool):
                        raise ValueError("`mri` must be a boolean (True/False).")

                if not isinstance(subject_fs_id, str) or not subject_fs_id.strip():
                        raise ValueError("`subject_fs_id` must be a non-empty string.")

                if not (subjects_fs_dir is None or isinstance(subject_fs_id, str) or Path(subjects_fs_dir).isdir()):
                        raise ValueError("`subject_fs_dir` must be None or path to a directory.")

                if not isinstance(filtering, bool):
                        raise ValueError("`filtering` must be bool.")
                
                if artifact_correction not in self.VALID_ARTIFACT_METHODS:
                        raise ValueError(
                                f"`artifact_correction` must be one of {self.VALID_ARTIFACT_METHODS}, "
                                f"got {artifact_correction!r}."
                        )

                if not isinstance(save_raw, bool):
                        raise ValueError("`save_raw` must be True or False.")

                if not isinstance(save_nf_signal, bool):
                        raise ValueError("`save_nf_signal` must be True or False.")

                if config_file is None:
                        config = PROJECT_ROOT.parent / "config_methods.yml"
                elif config_file.endswith(".yml") and Path(config_file).is_file():
                        config = config_file
                
                else:
                        raise ValueError("`config_file` must be None or Path to a config file.")
                        
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
                                chunk_size=10,
                                mock_lsl=False,
                                fname=None,
                                n_repeat=np.inf,
                                bufsize_baseline=4,
                                bufsize_main=3,
                                acquisition_delay=0.001,
                                timeout=2
                                ):
                """
                Connect to a Lab Streaming Layer (LSL) stream for EEG data acquisition.

                This method sets up a live or mock LSL stream depending on the parameters.
                It handles stream creation, montage assignment, and metadata initialization.
                Any existing stream is disconnected before a new connection. For mock streams,
                a pre-recorded EEG file can be streamed as if it were live data.

                Parameters
                ----------
                chunk_size : int, optional
                        Number of samples per chunk for streaming. Default is 10.
                mock_lsl : bool, optional
                        If True, a pre-recorded EEG file will be streamed instead of connecting to a live LSL stream.
                        Default is False.
                fname : str | Path | None, optional
                        Path to the EEG file used for mock streaming. If None, a sample dataset will be used.
                        Default is None.
                n_repeat : int | float, optional
                        Number of times to repeat the mock recording. Default is np.inf.
                bufsize_baseline : int, optional
                        Buffer size to use for 'baseline' sessions. Default is 4.
                bufsize_main : int, optional
                        Buffer size to use for 'main' sessions. Default is 3.
                acquisition_delay : float, optional
                        Delay in seconds between consecutive LSL acquisition attempts. Default is 0.001.
                timeout : float, optional
                        Maximum time in seconds to wait for LSL stream connection. Default is 2.

                Raises
                ------
                FileNotFoundError
                        If `mock_lsl` is True and the specified `fname` does not exist.
                ConnectionError
                        If the LSL stream cannot be connected within the timeout period.

                Attributes
                ----------
                stream : Stream
                        The connected LSL stream object.
                sfreq : float
                        Sampling frequency of the connected stream.
                rec_info : dict
                        Metadata/info dictionary from the connected stream.
                bufsize : int
                        Buffer size used for the stream, based on session type.
                source_id : str
                        Unique identifier for the LSL stream.
                subject_dir : Path
                        Directory created for the current subject.

                Notes
                -----
                - Previous streams are disconnected automatically before a new connection.
                - For mock streams, a `Player` object is automatically started.
                - Callable attributes of the stream that are not private are copied to `self`.

                Examples
                --------
                >>> session.connect_to_lsl(chunk_size=20, mock_lsl=True)
                """

                self.subject_dir = Path(self.subjects_dir) / self.subject_id
                self.subject_dir.mkdir(parents=True, exist_ok=True)
                
                ## disconnect the previous streaming if any
                if hasattr(self, "stream"): 
                        if self.stream.connected:
                                self.stream.disconnect()
                if mock_lsl and fname is None:
                        fname = PROJECT_ROOT / "data" / "sample" / "sample_data.vhdr"

                ## get the recording info and create the stream
                if self.session == "baseline":
                        self.bufsize = bufsize_baseline
                if self.session == "main":
                        self.bufsize = bufsize_main
                if Path(self.montage).is_file():
                        self.montage = read_dig_captrak(self.montage)

                self.source_id = uuid.uuid4().hex
                if mock_lsl:
                        player = Player(
                                        fname,
                                        chunk_size=chunk_size,
                                        n_repeat=n_repeat,
                                        source_id=self.source_id
                                        ).start()
                        stream = Stream(bufsize=self.bufsize, source_id=self.source_id)
                        stream.connect(acquisition_delay=acquisition_delay, timeout=timeout)
                        stream.set_montage(self.montage, on_missing="warn")
                        stream.pick("eeg")
                        stream.set_meas_date(datetime.datetime.now().replace(tzinfo=datetime.timezone.utc))
                        self.stream = stream
                else:
                        stream = Stream(
                                        bufsize=self.bufsize,
                                        name=None,
                                        source_id=self.source_id
                                        )
                        stream.connect(acquisition_delay=acquisition_delay, timeout=timeout)
                        stream.set_montage(self.montage, on_missing="warn")
                        stream.pick("eeg")
                        stream.set_meas_date(datetime.datetime.now().replace(tzinfo=datetime.timezone.utc))
                        self.stream = stream
                        
                self.sfreq = stream.info["sfreq"]  
                self.rec_info = stream.info     
                self.rec_info["subject_info"] = {"his_id": self.subject_id}

                ## copying attributes
                for attr_name in dir(self.stream):
                        attr_value = getattr(self.stream, attr_name)
                        if callable(attr_value) and not attr_name.startswith("__"):
                                setattr(self, attr_name, attr_value)
        

        def record_baseline(self, baseline_duration, winsize=3):
                """
                Record a baseline EEG segment to extract features for neurofeedback.

                The method continuously streams data from the LSL connection for the 
                specified baseline duration. Data is collected in windows of size `winsize`
                and stored as a Raw object. The raw baseline is saved to disk, and the 
                inverse operator is computed for the session.

                Parameters
                ----------
                baseline_duration : float
                        Desired duration of the baseline recording in seconds.
                winsize : float, optional
                        Size of the data window to fetch from the buffer in seconds. The method
                        will fetch the last `winsize * sfreq` samples from the buffer. If None,
                        the entire buffer is returned. Default is 3.

                Returns
                -------
                None

                Notes
                -----
                - The baseline data is saved under `subject_dir/baseline/visit_{visit}-raw.fif`.
                - After recording, `self.raw_baseline` holds the MNE Raw object for the baseline.
                - The inverse operator is computed after baseline recording and stored internally.
                - The LSL stream remains connected after this method completes.

                Examples
                --------
                >>> session.record_baseline(baseline_duration=60, winsize=5)
                """
                
                self.baseline_duration = baseline_duration
                #time.sleep(self.bufsize)
                print("Recording Initiated ...!")
                data = [] 
                t_start = local_clock()
                while local_clock() < t_start + self.baseline_duration:
                        data.append(self.stream.get_data(winsize)[0])
                        time.sleep(winsize)

                data = np.concatenate(np.array(data), axis=1)
                raw_baseline = RawArray(data, self.rec_info)
                (self.subject_dir / "baseline").mkdir(parents=True, exist_ok=True)
                fname_save = self.subject_dir / "baseline" / f"visit_{self.visit}-raw.fif"
                raw_baseline.save(fname_save, overwrite=True)
                
                self.raw_baseline = raw_baseline
                (self.subject_dir / "inv").mkdir(parents=True, exist_ok=True)
                self.compute_inv_operator()

        def get_blink_template(self, max_iter=800, method="infomax"):
                """
                Identify the eye blink component from the baseline EEG and store its template.

                Parameters
                ----------
                max_iter : int, optional
                        Maximum iterations for ICA fitting. Default is 800.
                method : str, optional
                        ICA method to use (e.g., 'infomax'). Default is 'infomax'.

                Sets
                ----
                self.blink_template : np.ndarray
                        The spatial topography of the ICA component corresponding to eye blinks.
                """
                self.blink_template = create_blink_template(
                                                                self.raw_baseline,
                                                                max_iter=max_iter,
                                                                method=method
                                                        )

        def record_main(
                        self,
                        duration,
                        modality="sensor_power",
                        picks=None,
                        winsize=1,
                        estimate_delays=False,
                        modality_params=None,
                        show_raw_signal=True,
                        show_nf_signal=True,
                        show_design_viz=True,
                        design_viz="VisualRorschach"
                        ):
                """
                Record EEG data and extract neural features for neurofeedback.

                This method streams data from the LSL connection for the specified duration, 
                applies the selected neural feature extraction modality, and optionally visualizes 
                the neurofeedback signal in real-time. Supports multiple modalities and 
                optional artifact correction.

                Parameters
                ----------
                duration : float
                        Desired duration of the main recording in seconds.
                modality : str, optional
                        Method used to extract the neural feature for neurofeedback. Can also be a list 
                        of modalities. Default is 'sensor_power'.
                picks : str | list | None, optional
                        Channel names to include in the analysis. If None, all channels are selected.
                        Default is None.
                winsize : float, optional
                        Window size in seconds for fetching data from the buffer. The method will
                        fetch the last `winsize * sfreq` samples. If None, the entire buffer is returned.
                        Default is 1.
                estimate_delays : bool, optional
                        If True, acquisition and method delays will be estimated and stored. Default is False.
                modality_params : dict | None, optional
                        Dictionary of parameters to override default modality settings. If None, parameters
                        from the config file are used. Default is None.
                show_nf_signal : bool, optional
                        If True, displays real-time neurofeedback signal using pyqtgraph. Default is True.
                show_design_viz : bool, optional
                        If True, displays real-time neurofeedback design visualization using py5. Default is True.
                design_viz: str, optional
                        Preset name to be visualized.

                Raises
                ------
                NotImplementedError
                        If a requested modality is not implemented.
                AssertionError
                        If a source-based modality is used while `picks` is not None.

                Attributes
                ----------
                duration : float
                        Duration of the main recording.
                modality : str | list
                        Selected modality or list of modalities.
                picks : list | None
                        Channels selected for analysis.
                winsize : float
                        Window size in seconds for fetching data from the buffer.
                window_size_s : float
                        Window size in samples (winsize * sfreq).
                estimate_delays : bool
                        Flag indicating whether delays are estimated.
                _sfreq : float
                        Sampling frequency of the stream.
                show_nf_signal : bool
                        Flag for real-time neurofeedback visualization.
                _mods : list
                        List of modalities prepared for the session.
                app : QtWidgets.QApplication
                        PyQt application instance for visualization (if show_nf_signal is True).
                plot_widget : pg.PlotWidget
                        Plot widget for real-time visualization.
                colors_list : list
                        List of colors used for plotting multiple modalities.
                scales_dict : dict
                        Scaling factors for different modalities.

                Notes
                -----
                - Prepares modality-specific preprocessing methods (`_modality_prep`) before recording.
                - Callable neural feature extraction methods (`_modality`) are applied in real-time.
                - Real-time visualization uses a 10-second time window.
                - For artifact correction using "LMS", the reference channel index is computed internally.

                Examples
                --------
                >>> session.record_main(duration=120, modality="sensor_power", picks=["C3", "C4"], winsize=2)
                """     
                
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
                self.show_design_viz = show_design_viz
                self.design_viz = design_viz

                ######## If you don't want to use ring buffer ...
                # # fetch_secs can be tuned; larger => fewer LSL calls, but larger latency for new info
                # fetch_secs = max(self.winsize * 4, 2.0)      # fetch at least 2s or 4x winsize
                # fetch_size_s = int(fetch_secs * self._sfreq)  # samples fetched per LSL call
                # # hop between consecutive processed windows (50% overlap by default)
                # hop_samples = max(self.window_size_s // 2, 1)
                # n_ch = len(self.rec_info["ch_names"])
                # ring_buffer = np.zeros((n_ch, 0), dtype=np.float32)
                # last_plot_time = 0.0
                # plot_interval = 0.05  # seconds between UI updates (tweak to taste)

                if self.artifact_correction == "lms":
                        ref_ch_idx = self.rec_info["ch_names"].index(self.ref_channel)
                if self.artifact_correction == "orica":
                        self.run_orica(n_channels=len(self.rec_info["ch_names"]), forgetfac=0.99)

                ## preparing the methods
                if isinstance(modality, str): 
                        mods = [modality]
                else:
                        mods = modality
                self._mods = mods

                precomps, nf_mods = [], []
                for modality in mods:
                        self.params = get_params(self.config_file, modality, self.modality_params)
                        nf_mod_prep = getattr(self, f"_{modality}_prep", None)
                        nf_mod = getattr(self, f"_{modality}", None)
                        if not callable(nf_mod):
                                raise NotImplementedError(f"{modality} modality not implemented yet.")
                        if "source" in self.modality:
                                assert self.picks is None, "picks should be None for source methods." 
                        precomp = nf_mod_prep()

                        precomps.append(precomp)
                        nf_mods.append(nf_mod)
                
                if estimate_delays:
                        acq_delays = []
                        artifact_delays = []
                        method_delays = {mod: [] for mod in mods}
                        plot_delays = []
                        
                ## add vizualisation
                if self.show_nf_signal:
                        self.app = QtWidgets.QApplication([])
                        self.plot_widget = pg.PlotWidget(title="Neurofeedback")
                        self.plot_widget.showGrid(x=True, y=True)
                        self.plot_widget.addLegend()
                        self.plot_widget.setLabel('bottom', 'Time', units='s')
                        self.plot_widget.setLabel('left', 'Signal')
                        self.plot_widget.setYRange(-1, len(mods) * 3 + 2)
                        self.plot_widget.resize(1000, 500)
                        self.plot_widget.show()
                        self.colors_list = ["#5DA5A4", "#9A7DFF", "#FFB085", "#8FBF87", "#D98BA3", "#E0C368"]
                        self.scales_dict = {
                                                "sensor_power": 7e-13,
                                                "band_ratio": 1.5,
                                                "source_power": 3e-2,
                                                "sensor_connectivity": 1,
                                                "source_connectivity": 1,
                                                "sensor_graph": 0.05, 
                                                "source_graph": 2e-17, 
                                                "entropy": 0.35,
                                                "argmax_freq": 8,
                                                "individual_peak_power": 1
                                                }

                        self.legend = None
                        self.curve = self.plot_widget.plot(pen='y')
                        self.time_axis = np.linspace(0, 10, int(self._sfreq)) # show for 10 seconds
                        

                if self.show_raw_signal:
                        self.plot_rt()
                if self.show_design_viz:
                        pass

                if self.filtering:
                        self.stream.filter(l_freq=l_freq, h_freq=h_freq)

                nf_data = {mod: [] for mod in mods}
                t_start = local_clock()

                ######## If you don't want to use ring buffer ...
                while local_clock() < t_start + self.duration:
                        tic = time.time()

                        ## get data
                        data = self.stream.get_data(self.winsize, picks=self.picks)[0] # n_chs * n_times
                        if estimate_delays:
                                acq_delays.append(time.time() - tic)
                        if data.shape[1] != self.window_size_s: continue

                        ## add artifact correction
                        if self.artifact_correction == "lms":
                                art_tic = time.time()
                                data = remove_blinks_lms(data, ref_ch_idx=ref_ch_idx, n_taps=5, mu=0.01)
                                artifact_delays.append(time.time() - art_tic)
                        
                        if self.artifact_correction == "orica":
                                art_tic = time.time()
                                sources = self.orica.transform(data)
                                blink_idx, corrs = self.orica.find_blink_ic(self.blink_template, threshold=0.4)
                                # print(f"blink_idx and corr: {blink_idx}, {corrs}")
                                sources_clean = sources.copy()
                                if blink_idx:
                                        print(blink_idx)
                                        for idx in blink_idx:
                                                sources_clean[idx, :] = 0.0
                                        data = self.orica.inverse_transform(sources_clean)
                                artifact_delays.append(time.time() - art_tic)

                        ## compute nf
                        for idx, mod in enumerate(mods):
                                nf_data_, method_delay = nf_mods[idx](data, **precomps[idx])
                                nf_data[mod].append(nf_data_)
                                if estimate_delays:
                                        method_delays[mod].append(method_delay)

                        ## QT signal vizualisation
                        if self.show_nf_signal:
                                plot_tic = time.time()
                                last_vals = [nf_data[key][-1] for key in mods]
                                self.update_nf_plot(last_vals, labels=mods)
                                self.app.processEvents()
                                if estimate_delays:
                                        plot_delays.append(time.time() - plot_tic)
                                time.sleep(0.005)

                        ## Py5 visualization
                        if self.show_design_viz:
                                plot_design(nf_data_)

                ######## If you want to use ring buffer ...
                # while local_clock() < t_start + self.duration:
                #         fetch_tic = time.time()
                #         fetched = self.stream.get_data(fetch_secs, picks=self.picks)[0]  # shape: n_ch x n_samples
                #         acq_delay = time.time() - fetch_tic
                #         if estimate_delays:
                #                 acq_delays.append(acq_delay)
                        
                #         if fetched is None or fetched.size == 0:
                #                 time.sleep(0.001)
                #                 continue

                #         ring_buffer = np.concatenate((ring_buffer, fetched), axis=1)
                #         max_buf_len = fetch_size_s + self.window_size_s
                #         if ring_buffer.shape[1] > max_buf_len:
                #                 ring_buffer = ring_buffer[:, -max_buf_len:]

                #         while ring_buffer.shape[1] >= self.window_size_s:
                #                 window = ring_buffer[:, :self.window_size_s].copy()  # shape n_ch x window_size_s

                #                 ## add artifact correction
                #                 if self.artifact_correction is not None:
                #                         art_tic = time.time()
                #                         if self.artifact_correction == "lms":
                #                                 window = remove_blinks_lms(window, ref_ch_idx=ref_ch_idx, n_taps=5, mu=0.01)
                                        
                #                         elif self.artifact_correction == "orica":
                #                                 sources = self.orica.transform(window)
                #                                 blink_score = np.abs(np.dot(
                #                                                         sources.T, self.blink_template) /
                #                                                         (np.linalg.norm(sources, axis=1) * \
                #                                                         np.linalg.norm(self.blink_template) + 1e-12))
                #                                 blink_idx = np.argmax(blink_score)
                #                                 if blink_score[blink_idx] > 0.7:  # threshold of blink correlation
                #                                         sources[blink_idx, :] = 0
                #                                 #data = self.orica.inverse_transform(sources)
                #                                 window = self.orica.inverse_transform(sources)
                #                         if estimate_delays:
                #                                 artifact_delays.append(time.time() - art_tic)

                #                 ## compute nf
                #                 for idx, mod in enumerate(mods):
                #                         meth_tic = time.time()
                #                         nf_data_, method_delay = nf_mods[idx](window, **precomps[idx])
                #                         nf_data[mod].append(nf_data_)
                #                         if estimate_delays:
                #                                 method_delays[mod].append({
                #                                                         "measured": time.time() - meth_tic,
                #                                                         "reported": method_delay
                #                                                         })

                #                 ## QT signal vizualisation
                #                 if self.show_nf_signal:
                #                         now = time.time()
                #                         if (now - last_plot_time) >= plot_interval:
                #                                 plot_tic = time.time()
                #                                 last_vals = [nf_data[key][-1] for key in mods]
                #                                 self.update_nf_plot(last_vals, labels=mods)
                #                                 self.app.processEvents()
                #                                 if estimate_delays:
                #                                         plot_delays.append(time.time() - plot_tic)
                #                                 last_plot_time = now

                #                 ## Py5 visualization
                #                 if self.show_design_viz:
                #                         plot_design(nf_data_)

                #                 ring_buffer = ring_buffer[:, hop_samples:]


                self.nf_data = nf_data
                if estimate_delays:
                        self.acq_delays = acq_delays
                        self.artifact_delays = artifact_delays
                        self.method_delays = method_delays
                        self.plot_delays = plot_delays
                        self.save(nf_data=True,
                                acq_delay=True,
                                artifact_delay=True,
                                method_delay=True,
                                format="json")
                else:
                        self.save(nf_data=True,
                                acq_delay=False,
                                artifact_delay=False,
                                method_delay=False,
                                format="json")

                self.app.exec()
        
        @property
        def modality_params(self):
                """
                Get the currently set modality parameters.

                Returns
                -------
                dict
                        Dictionary containing the current modality parameters.
                """
                return self._modality_params

        @modality_params.setter
        def modality_params(self, params):
                """
                Set the modality parameters for neural feature extraction.

                Parameters
                ----------
                params : dict | None
                        Dictionary of modality-specific parameters. If None, an empty dictionary is used.

                Raises
                ------
                ValueError
                        If `params` is not a dictionary or None.
                """
                if params is not None and not isinstance(params, dict):
                        raise ValueError("Can only be a dictionary.")
                else:
                        self._modality_params = params or {}

        def get_default_params(self):
                """
                Return the default parameters from the configuration YAML file.

                Returns
                -------
                dict
                        Default parameters for all neural feature modalities.
                """

                return self._default_params
        
        ## --------------------------- General Methods --------------------------- ##

        def update_nf_plot(self, new_vals, labels=None):
                """
                Update the neurofeedback plot in real-time.

                This method updates the visual representation of neural feature values for 
                each modality or channel. Normalization and vertical offsets are applied 
                to allow multiple modalities to be plotted simultaneously.

                Parameters
                ----------
                new_vals : list | np.ndarray
                        List or array of the latest neural feature values for each modality.
                        Example: [sensor_power_last, band_ratio_last].
                labels : list of str, optional
                        Labels for each modality to display in the plot legend.
                        Example: ["Sensor Power", "Band Ratio"].

                Notes
                -----
                - The first call initializes the plot and legend.
                - Subsequent calls update the plot data in a rolling window fashion.
                - The y-axis values are normalized based on modality-specific scaling factors
                stored in `self.scales_dict`.
                - The time axis is defined by `self.time_axis` and the number of curves by
                the number of modalities in `new_vals`.

                Examples
                --------
                >>> session.update_nf_plot([0.3, 0.7], labels=["Sensor Power", "Band Ratio"])
                """
                
                n_labels = len(new_vals)
                new_vals = np.array(new_vals, dtype=float)

                ## normalize
                shifts = np.arange(0, n_labels * 2, 2)
                scales = [self.scales_dict[k] for k in self._mods]

                norm_vals = []
                for val, scale, shift in zip(new_vals, scales, shifts):
                        norm = (val / scale) + shift
                        norm_vals.append(norm)
                norm_vals = np.array(norm_vals)
                print(norm_vals)
                
                # Initialize plot_data and curves on first call
                if not hasattr(self, "plot_data"):
                        self.plot_data = np.zeros(shape=(n_labels, len(self.time_axis)))
                        self.curves = []
                        colors = [pg.mkPen(color=color, width=4) for color in self.colors_list]
                        for lb in range(n_labels):
                                pen = colors[lb % len(colors)]
                                curve = self.plot_widget.plot(self.time_axis, self.plot_data[lb, :], pen=pen, name=labels[lb])
                                self.curves.append(curve)
                        
                        if self.legend is None:
                                self.legend = self.plot_widget.addLegend()
                                self.legend.setLabelTextSize('14pt')
                                self.legend.anchor(itemPos=(0,0), parentPos=(0,0))

                self.plot_data = np.roll(self.plot_data, -1, axis=1)
                self.plot_data[:, -1] = norm_vals

                # Update curves
                for lb, curve in enumerate(self.curves):
                        curve.setData(self.time_axis, self.plot_data[lb, :])
        
        def plot_rt(self, bufsize=0.2):
                """
                Visualize EEG signals in real-time from the LSL stream.

                Parameters
                ----------
                bufsize_view : float, optional
                        Buffer/window size in seconds for the StreamReceiver display.
                        Default is 0.2 s.

                Notes
                -----
                - Uses `Viewer` to create a real-time visualization of the EEG stream.
                - The buffer size controls how much of the recent signal is visible.

                Examples
                --------
                >>> session.plot_rt(bufsize=0.2)
                """  
                Viewer(stream_name=self.stream.name).start(bufsize)
        
        def compute_inv_operator(self):
                """
                Compute and save the inverse operator for source localization.

                The method computes the MNE inverse operator from the baseline recording
                and FreeSurfer subject data, then saves it to the subject's directory.

                Notes
                -----
                - The inverse operator is stored in `self.inv`.
                - The operator is saved under `subject_dir/inv/visit_{visit}-inv.fif`.

                Examples
                --------
                >>> session.compute_inv_operator()
                """
                inv = _compute_inv_operator(self.raw_baseline,
                                                subject_fs_id=self.subject_fs_id,
                                                subjects_fs_dir=self.subjects_fs_dir)
                self.inv = inv
                write_inverse_operator(
                                        fname=self.subject_dir / "inv" / f"visit_{self.visit}-inv.fif",
                                        inv=inv,
                                        overwrite=True
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
                        random_state=None
                        ):
                """
                Initialize an online ICA (ORICA) instance for real-time artifact removal.

                Parameters
                ----------
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
                After initialization, use `self.orica.transform(data_chunk)` to process data,
                and optionally remove identified artifact components.
                """
                self.orica = ORICA(
                        n_channels=n_channels,
                        learning_rate=learning_rate,
                        block_size=block_size,
                        online_whitening=online_whitening,
                        calibrate_pca=calibrate_pca,
                        forgetfac=forgetfac,
                        nonlinearity=nonlinearity,
                        random_state=random_state
                )

        def save(self, nf_data=True, acq_delay=True, artifact_delay=True, method_delay=True, raw_data=False, format="json"):
                """
                Save neurofeedback session data to disk.

                This method disconnects the LSL stream and saves neurofeedback data,
                acquisition delays, method delays, and optionally raw data. Currently,
                saving raw data is not implemented.

                Parameters
                ----------
                nf_data : bool, optional
                        If True, save the neurofeedback data. Default is True.
                acq_delay : bool, optional
                        If True, save acquisition delay data. Default is True.
                method_delay : bool, optional
                        If True, save method delay data. Default is True.
                raw_data : bool, optional
                        If True, save raw EEG data. Not implemented yet. Default is False.
                format : str, optional
                        File format to save the data. Currently only 'json' is supported. Default is 'json'.

                Raises
                ------
                NotImplementedError
                        If `raw_data=True`, as saving raw EEG data is not implemented.

                Notes
                -----
                - Creates directories `neurofeedback`, `delays`, `main`, and `reports` under the subject folder.
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
                        raise NotImplementedError("saving raw_data isn't implemented yet")
        
        def create_report(self, overwrite=True):
                """
                Create an HTML report summarizing the neurofeedback session.

                The report includes baseline recordings, EEG sensor visualizations,
                and brain label plots depending on the modality used.

                Parameters
                ----------
                overwrite : bool, optional
                        If True, overwrite an existing report file with the same name. Default is True.

                Notes
                -----
                - Baseline raw data is included in the report without PSD or butterfly plots.
                - Sensor plots include both topographic and 3D views.
                - Brain label figures are added for source-based modalities using `plot_glass_brain`.
                - The report is saved under `subject_dir/reports` with a filename including
                subject ID, visit, and modality.

                Examples
                --------
                >>> session.create_report(overwrite=True)
                """
                report = Report(title=f"Neurofeedback Session with {self.modality} modality")
                report.add_raw(self.raw_baseline, title="Baseline recording", psd=False, butterfly=False)
                if isinstance(self.modality, str): self.modality = [self.modality]
                for mod in self.modality:
                        if not "source" in mod:
                                if self.picks is not None:
                                        self.rec_info["bads"].extend(self.picks)
                                else:
                                        self.rec_info["bads"].extend(self.rec_info["ch_names"])

                                fig_sensors = plt.figure(figsize=(10, 5))
                                ax1 = fig_sensors.add_subplot(121)
                                ax2 = fig_sensors.add_subplot(122, projection='3d')
                                mne.viz.plot_sensors(info=self.rec_info, kind="topomap", axes=ax1, show=False)
                                mne.viz.plot_sensors(info=self.rec_info, kind="3d", axes=ax2, show=False)
                                ax2.axis("off")
                                self.rec_info["bads"] = []
                                report.add_figure(fig=fig_sensors, title="Sensors")
                        else :
                                if mod in ["source_connectvity", "source_graph"]:
                                        figure_brain = plot_glass_brain(bl1=self.params["brain_label"], bl2=None)
                                else:
                                        figure_brain = plot_glass_brain(bl1=self.params["brain_label_1"],
                                                                        bl2=self.params["brain_label_2"])
                                        
                                report.add_figure(fig=figure_brain, title=f"selected brain labels")
                

                report_fname = f"subject_{self.subject_id}_visit_{self.visit}_modality_{self.modality}.html"
                report.save(self.subject_dir / "reports" / report_fname, overwrite=overwrite)

        ## --------------------------- Neural Feature Extraction Methods (preparation) --------------------------- ##

        def _sensor_power_prep(self):
                """
                Prepare parameters for the 'sensor_power' neural feature modality.

                Returns
                -------
                dict
                        Dictionary containing precomputed parameters for the modality:
                        - sfreq : float, sampling frequency
                        - frange : list or tuple, frequency range to compute power
                        - method : str, method for power computation
                        - relative : bool, whether to compute relative power
                """
                precomp = dict(
                                sfreq=self.rec_info["sfreq"],
                                frange=self.params["frange"],
                                method=self.params["method"],
                                relative=self.params["relative"]
                                )
                return precomp

        def _argmax_freq_prep(self):
                """
                Prepare parameters for the 'argmax_freq' modality by extracting aperiodic components.

                Returns
                -------
                dict
                        Precomputed parameters including:
                        - fft_window : array, FFT windows
                        - freq_band : array, frequency band
                        - freq_band_idxs : array, indices of frequencies
                        - ap_model : array, estimated aperiodic component
                        - gaussian : callable, function to model Gaussian peaks

                Raises
                ------
                AssertionError
                        If `raw_baseline` has not been recorded prior to this step.
                """
                assert hasattr(self, "raw_baseline"), "Baseline recording should be done prior to this step."
                ## extracting the aperiodic components from the baseline recording
                ap_params, _ = estimate_aperiodic_component(
                                                                raw_baseline=self.raw_baseline,
                                                                picks=self.picks,
                                                                method=self.params["method"]
                                                                )
                n_samples = int(self.winsize * self._sfreq)
                fft_window = np.hanning(n_samples)
                freqs = np.fft.rfftfreq(self.window_size_s, d=1/self.rec_info["sfreq"])
                freq_band_mask = (freqs >= self.params["frange"][0]) & (freqs <= self.params["frange"][1])
                freqs_band = freqs[freq_band_mask]
                
                ap_model = (10 ** ap_params[0]) / (freqs_band ** ap_params[1])
                gaussian = lambda x, a, mu, sigma: a * np.exp(-(x - mu)**2 / (2 * sigma**2))
                precomp = {
                        "fft_window": fft_window,
                        "freq_band_mask": freq_band_mask,
                        "freqs_band": freqs_band,
                        "ap_model": ap_model,
                        "gaussian": gaussian
                        }
                return precomp

        def _band_ratio_prep(self):
                """
                Prepare parameters for the 'band_ratio' neural feature modality.

                Returns
                -------
                dict
                        Dictionary containing precomputed parameters for the modality:
                        - sfreq : float, sampling frequency
                        - frange_1 : list or tuple, frequency range for numerator band
                        - frange_2 : list or tuple, frequency range for denominator band
                        - method : str, method for computing power in each band
                """
                precomp = dict(
                                sfreq=self.rec_info["sfreq"],
                                frange_1=self.params["frange_1"],
                                frange_2=self.params["frange_2"],
                                method=self.params["method"]
                                )
                return precomp

        def _individual_peak_power_prep(self):
                """
                Prepare parameters for the 'individual_peak_power' modality.

                Computes the individual peak frequency from baseline data and FFT indices
                for power extraction around the peak.

                Returns
                -------
                dict
                        Dictionary containing:
                        - fft_window : array, FFT windows
                        - individual_freq_band_idxs : array, indices around the identified peak frequency

                Notes
                -----
                - If multiple peaks are found in the selected frequency range, the center
                frequency is set to the middle of the range and a warning is issued.
                - Requires baseline recording to be done prior to calling this method.
                """
                _, peak_params_ = estimate_aperiodic_component(
                                                                raw_baseline=self.raw_baseline,
                                                                picks=self.picks,
                                                                method=self.params["method"]
                                                                )
                peak_params = [peak_param[0] for peak_param in peak_params_ if self.params["frange"][0] < peak_param[0] < self.params["frange"][1]]
                
                if len(peak_params) == 1:
                        cf = peak_params[0]
                else:
                        cf = (self.params["frange"][0] + self.params["frange"][1]) / 2
                        warn(f"center frequency was set to the middle frequency in the selected frequency range.")
                
                ## compute power in a small range around individual peak
                fft_window, _, _, frequencies = compute_fft(
                                                                sfreq=self._sfreq,
                                                                winsize=self.winsize,
                                                                freq_range=self.params["frange"],
                                                                freq_res=1
                                                                )
                freq_var = 1
                individual_freq_band_idxs = np.where(np.logical_and(cf - freq_var <= frequencies,
                                                        frequencies <= cf + freq_var))[0]
                precomp = {
                                "fft_window": fft_window,
                                "individual_freq_band_idxs": individual_freq_band_idxs
                                }
                return precomp

        def _entropy_prep(self):
                """
                Prepare parameters for the 'entropy' neural feature modality.

                Returns
                -------
                dict
                        Precomputed parameters including:
                        - sos : array, second-order sections of the Butterworth bandpass filter
                """
                sos = butter_bandpass(
                                        self.params["frange"][0],
                                        self.params["frange"][1],
                                        self._sfreq,
                                        order=5
                                        )
                precomp = {"sos": sos}
                return precomp

        def _source_power_prep(self):
                """
                Prepare parameters for the 'source_power' modality.

                Returns
                -------
                dict
                        Precomputed parameters including:
                        - fft_window : array, FFT windows
                        - freq_band_idxs : array, indices of the selected frequency band
                        - brain_label : instance of mne.Label, the target brain label
                        - inverse_operator : instance of mne.minimum_norm.InverseOperator, for source localization

                Notes
                -----
                - Reads the inverse operator from the subject's visit folder.
                - Requires the subject FreeSurfer ID and directory.
                """
                fft_window, _, freq_band_idxs, _ = compute_fft(
                                                                sfreq=self._sfreq,
                                                                winsize=self.winsize,
                                                                freq_range=self.params["frange"],
                                                                )
                bls = read_labels_from_annot(
                                                subject=self.subject_fs_id,
                                                parc=self.params["atlas"],
                                                subjects_dir=self.subjects_fs_dir
                                                )
                bl_names = [bl.name for bl in bls]
                bl_idx = bl_names.index(self.params["brain_label"])
                brain_label = bls[bl_idx]
                inverse_operator = read_inverse_operator(fname=self.subject_dir / "inv" / f"visit_{self.visit}-inv.fif")
                precomp = {
                                "fft_window": fft_window,
                                "freq_band_idxs": freq_band_idxs,
                                "brain_label": brain_label,
                                "inverse_operator": inverse_operator
                        }
                return precomp

        def _sensor_connectivity_prep(self):
                """
                Prepare parameters for the 'sensor_connectivity' modality.

                Returns
                -------
                dict
                        Precomputed parameters including:
                        - indices : tuple of arrays, channel index pairs for connectivity computation
                        - freqs : array, frequencies to compute connectivity
                """
                ch_names = self.rec_info["ch_names"]
                chs = self.params["channels"]
                indices = [(np.array([ch_names.index(ch1), ch_names.index(ch2)])) for ch1, ch2 in zip(chs[0], chs[1])]
                indices = tuple(indices)
                freq_res = 6 
                freqs = np.linspace(self.params["frange"][0], self.params["frange"][1], freq_res)
                precomp = {
                                "indices": indices,
                                "freqs": freqs
                        }
                return precomp

        def _source_connectivity_prep(self):
                """
                Prepare parameters for the 'source_connectivity' modality.

                Returns
                -------
                dict
                        Precomputed parameters including:
                        - merged_label : mne.Label, combination of left and right hemisphere labels
                        - inverse_operator : mne.minimum_norm.InverseOperator, for source localization
                        - freqs : array, frequencies to compute connectivity

                Raises
                ------
                AssertionError
                        If brain labels are not from the correct hemispheres ('lh' for left, 'rh' for right)
                """
                assert self.params["brain_label_1"][-2:] == "lh", "first brain label should be selected from left hemisphere."
                assert self.params["brain_label_2"][-2:] == "rh", "second brain label should be selected from right hemisphere."
                
                ## initiating the source space
                bls = read_labels_from_annot(
                                                subject=self.subject_fs_id,
                                                parc=self.params["atlas"],
                                                subjects_dir=self.subjects_fs_dir
                                                )
                bl_names = [bl.name for bl in bls]
                merged_label = bls[bl_names.index(self.params["brain_label_1"])] + \
                                bls[bl_names.index(self.params["brain_label_2"])]
                inverse_operator = read_inverse_operator(fname=self.subject_dir / "inv" / f"visit_{self.visit}-inv.fif")
                freq_res = 6
                freqs = np.linspace(self.params["frange"][0], self.params["frange"][1], freq_res)
                
                precomp = {
                                "merged_label": merged_label,
                                "inverse_operator": inverse_operator,
                                "freqs": freqs
                        }
                return precomp

        def _sensor_graph_prep(self):
                """
                Prepare parameters for the 'sensor_graph' modality.

                Returns
                -------
                dict
                        Precomputed parameters including:
                        - indices : tuple of arrays, channel index pairs for graph computation
                        - sos : array, second-order sections of Butterworth bandpass filter
                """
                ch_names = self.rec_info["ch_names"]
                chs = self.params["channels"]
                indices = [(np.array([ch_names.index(ch1), ch_names.index(ch2)])) for ch1, ch2 in zip(chs[0], chs[1])]
                indices = tuple(indices)

                ## initiating the filter
                sos = butter_bandpass(
                                        self.params["frange"][0],
                                        self.params["frange"][1],
                                        self._sfreq,
                                        order=5
                                        )
                precomp = {"indices": indices, "sos": sos}
                return precomp

        def _source_graph_prep(self):
                """
                Prepare parameters for the 'source_graph' neural feature modality.

                Returns
                -------
                dict
                        Precomputed parameters including:
                        - bls : list of mne.Label, all labels from the atlas
                        - bl_idxs : tuple of int, indices of selected brain labels for graph computation
                        - inverse_operator : mne.minimum_norm.InverseOperator, for source localization
                        - sos : array, second-order sections of Butterworth bandpass filter
                """
                ## initiating the source space
                bls = read_labels_from_annot(
                                                subject=self.subject_fs_id,
                                                parc=self.params["atlas"],
                                                subjects_dir=self.subjects_fs_dir
                                                )
                bl_names = [bl.name for bl in bls]
                bl_idxs = (bl_names.index(self.params["brain_label_1"]), bl_names.index(self.params["brain_label_2"])) 
                inverse_operator = read_inverse_operator(fname=self.subject_dir / "inv" / f"visit_{self.visit}-inv.fif")

                sos = butter_bandpass(
                                        self.params["frange"][0],
                                        self.params["frange"][1],
                                        self._sfreq,
                                        order=5
                                        )
                precomp = {
                                "bls": bls,
                                "bl_idxs": bl_idxs,
                                "inverse_operator": inverse_operator,
                                "sos": sos,
                        }
                return precomp

        ## --------------------------- Neural Feature Extraction Methods (main) --------------------------- ##

        @timed
        def _sensor_power(self, data, sfreq, frange, method, relative):
                """
                Compute sensor-level power in a given frequency band.

                Parameters
                ----------
                data : np.ndarray
                        EEG data array (channels x samples).
                sfreq : float
                        Sampling frequency.
                frange : list or tuple
                        Frequency range to compute power [fmin, fmax].
                method : str
                        Method for power computation: 'fft', 'periodogram', 'welch', or 'multitaper'.
                relative : bool
                        If True, return relative power within the frequency band.

                Returns
                -------
                float
                        Mean power across channels in the specified frequency band.
                """
                if method == "fft":
                        n_channels, n_samples = data.shape
                        n_fft = int(2 ** np.ceil(np.log2(n_samples)))
                        win = get_window(window="hann", Nx=n_samples, fftbins=True)
                        data_win = data * win
                        freqs = np.fft.rfftfreq(n_fft, d=1/sfreq)
                        psd = (np.abs(np.fft.rfft(data_win, n=n_fft)) ** 2) / (sfreq * np.sum(win**2))
                
                elif method == "periodogram":
                        freqs, psd = periodogram(data, sfreq, axis=1)
                
                elif method == "welch":
                        freqs, psd = welch(data, sfreq, axis=1)
                
                elif method == "multitaper":
                        psd, freqs = psd_array_multitaper(data, sfreq, axis=1)

                # Frequency band selection
                mask = (freqs >= frange[0]) & (freqs <= frange[1])
                freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
                bp = simpson(psd[:, mask], dx=freq_res, axis=1)
                if relative:
                        bp /= simpson(psd, dx=freq_res, axis=1)
                return bp.mean()

        @timed
        def _argmax_freq(self, data, fft_window, freq_band_mask, freqs_band, ap_model, gaussian):
                """
                Compute the individual peak frequency using FFT and a Gaussian fit.

                Parameters
                ----------
                data : np.ndarray
                        EEG data array (channels x samples).
                fft_window : np.ndarray
                        Precomputed FFT window.
                freq_band : np.ndarray
                        Frequencies corresponding to FFT bins.
                freq_band_idxs : np.ndarray
                        Indices of the frequency band of interest.
                ap_model : np.ndarray
                        Estimated aperiodic (1/f) component to subtract.
                gaussian : callable
                        Gaussian function to fit the peak.

                Returns
                -------
                float
                        Estimated individual peak frequency. Returns 0 if Gaussian fit fails.
                """
                data = np.multiply(data, fft_window)
                fftval = np.abs(np.fft.rfft(data, axis=1) / data.shape[-1])
                freqs = np.fft.rfftfreq(data.shape[-1], d=1/self._sfreq)
                freq_band_mask = (freqs >= self.params["frange"][0]) & (freqs <= self.params["frange"][1])
                freqs_band = freqs[freq_band_mask]
                total_power_band = np.mean(np.square(fftval[:, freq_band_mask]), axis=0)
                periodic_power = total_power_band - ap_model

                p0 = [periodic_power.max(), 10.5, 1.0]
                
                try:
                        popt, _ = curve_fit(gaussian, freqs_band, periodic_power, p0=p0)
                        individual_peak = popt[1]
                        print(f"individual peak : {individual_peak}")
                except RuntimeError:
                        individual_peak = 0 
                        warn(f"fitting failed and individual peak value is set to 0.")
                return individual_peak
        
        @timed
        def _band_ratio(self, data, sfreq, frange_1, frange_2, method):
                """
                Compute the ratio of band power between two frequency bands.

                Parameters
                ----------
                data : np.ndarray
                        EEG data array (channels x samples).
                sfreq : float
                        Sampling frequency.
                frange_1 : list or tuple
                        Frequency range for numerator band [fmin, fmax].
                frange_2 : list or tuple
                        Frequency range for denominator band [fmin, fmax].
                method : str
                        Method for power computation: 'fft', 'periodogram', 'welch', or 'multitaper'.

                Returns
                -------
                float
                        Ratio of mean power in `frange_1` to mean power in `frange_2`.
                """
                if method == "fft":
                        n_channels, n_samples = data.shape
                        n_fft = int(2 ** np.ceil(np.log2(n_samples)))
                        win = get_window(window="hann", Nx=n_samples, fftbins=True)
                        data_win = data * win
                        freqs = np.fft.rfftfreq(n_fft, d=1/sfreq)
                        psd = (np.abs(np.fft.rfft(data_win, n=n_fft)) ** 2) / (sfreq * np.sum(win**2))
                
                elif method == "periodogram":
                        freqs, psd = periodogram(data, sfreq, axis=1)
                
                elif method == "welch":
                        freqs, psd = welch(data, sfreq, axis=1)
                
                elif method == "multitaper":
                        psd, freqs = psd_array_multitaper(data, sfreq, axis=1)

                # Frequency band selection
                mask_1 = (freqs >= frange_1[0]) & (freqs <= frange_1[1])
                mask_2 = (freqs >= frange_2[0]) & (freqs <= frange_2[1])
                freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

                bp_1 = simpson(psd[:, mask_1], dx=freq_res, axis=1)
                bp_2 = simpson(psd[:, mask_2], dx=freq_res, axis=1)
                return bp_1.mean() / bp_2.mean()

        @timed
        def _individual_peak_power(self, data, fft_window, individual_freq_band_idxs):
                """
                Compute power around the individual peak frequency.

                Parameters
                ----------
                data : np.ndarray
                        EEG data array (channels x samples).
                fft_window : np.ndarray
                        Precomputed FFT window to apply on the data.
                individual_freq_band_idxs : np.ndarray
                        Indices of the frequency band around the individual peak.

                Returns
                -------
                float
                        Mean power across channels in the selected individual frequency band.
                """
                data = np.multiply(data, fft_window)
                fftval = np.abs(np.fft.rfft(data, axis=1) / data.shape[-1])
                power = np.average(np.square(fftval[:, individual_freq_band_idxs]).T)
                return power

        @timed
        def _entropy(self, data, sos):
                """
                Compute entropy of EEG signals.

                Parameters
                ----------
                data : np.ndarray
                        EEG data array (channels x samples).
                sos : np.ndarray
                        Second-order sections of the bandpass filter to apply.

                Returns
                -------
                float
                        Mean entropy value across channels.
                """
                data_filt = sosfiltfilt(sos, data)
                match self.params["method"]:
                        case "AppEn":
                                ents = compute_app_entropy(data_filt)
                        case "SampEn":
                                ents = compute_samp_entropy(data_filt)
                        case "Spectral":
                                ents = compute_spect_entropy(sfreq=self._sfreq, data=data_filt, psd_method=self.params["psd_method"])
                        case "SVD":
                                ents = compute_svd_entropy(data_filt)
                return ents.mean()

        @timed
        def _source_power(self, data, fft_window, freq_band_idxs, brain_label, inverse_operator):
                """
                Compute source-level power for a specific brain label.

                Parameters
                ----------
                data : np.ndarray
                        EEG data array (channels x samples).
                fft_window : np.ndarray
                        Precomputed FFT window.
                freq_band_idxs : np.ndarray
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
                
                ## compute source activation and then power
                stc_data = apply_inverse_raw(
                                                raw_data,
                                                inverse_operator,
                                                lambda2=1.0 / 9,
                                                pick_ori="normal",
                                                label=brain_label,
                                                ).data
                stc_data = np.multiply(stc_data, fft_window)
                fftval = np.abs(np.fft.rfft(stc_data, axis=1) / stc_data.shape[-1])
                stc_power = np.average(np.square(fftval[:, freq_band_idxs]).T)
                return stc_power

        @timed
        def _sensor_connectivity(self, data, indices, freqs):
                """
                Compute sensor-level connectivity between channel pairs.

                Parameters
                ----------
                data : np.ndarray
                        EEG data array (channels x samples).
                indices : tuple
                        Pairs of channel indices for connectivity calculation.
                freqs : np.ndarray
                        Frequencies at which to compute connectivity.

                Returns
                -------
                float
                        Mean connectivity across selected channel pairs.
                """
                con = spectral_connectivity_time(
                                                data=data[np.newaxis,:],
                                                freqs=freqs,
                                                indices=indices,
                                                average=False,
                                                sfreq=self._sfreq, 
                                                fmin=self.params["frange"][0],
                                                fmax=self.params["frange"][1], 
                                                faverage=True,
                                                mode=self.params["mode"],
                                                method=self.params["method"],
                                                n_cycles=5
                                                )
                con_data = np.squeeze(con.get_data(output='dense'))[indices].mean()
                return con_data

        @timed
        def _source_connectivity(self, data, merged_label, inverse_operator, freqs):
                """
                Compute source-level connectivity between two brain regions.

                Parameters
                ----------
                data : np.ndarray
                        EEG data array (channels x samples).
                merged_label : mne.Label
                        Merged label covering the two brain regions of interest.
                inverse_operator : mne.minimum_norm.InverseOperator
                        Inverse operator for source reconstruction.
                freqs : np.ndarray
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
                                        label=merged_label
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
                                                n_cycles=5
                                                )
                con_data = np.squeeze(con.get_data(output='dense'))[1][0]
                return con_data

        @timed
        def _sensor_graph(self, data, indices, sos):
                """
                Compute graph-theoretical metrics from sensor-level EEG data.

                Parameters
                ----------
                data : np.ndarray
                        EEG data array (channels x samples).
                indices : tuple
                        Pairs of channel indices to compute graph edges.
                sos : np.ndarray
                        Second-order sections of Butterworth bandpass filter.

                Returns
                -------
                float
                        Average edge value of the computed graph across selected channel pairs.
                """
                data_filt = sosfiltfilt(sos, data)
                graph_matrix = log_degree_barrier(
                                                data_filt,
                                                dist_type=self.params["dist_type"],
                                                alpha=self.params["alpha"],
                                                beta=self.params["beta"]
                                                )
                avg_edge = np.array([graph_matrix[idxs] for idxs in indices]).mean()
                return avg_edge
        
        @timed
        def _source_graph(self, data, bls, bl_idxs, inverse_operator, sos):
                """
                Compute graph-theoretical metrics from source-level EEG data.

                Parameters
                ----------
                data : np.ndarray
                        EEG data array (channels x samples).
                bls : list of mne.Label
                        Labels of brain regions used to extract time courses.
                bl_idxs : tuple of int
                        Indices of the brain labels to compute the graph edge between.
                inverse_operator : mne.minimum_norm.InverseOperator
                        Inverse operator for source reconstruction.
                sos : np.ndarray
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
                                        pick_ori="normal"
                                        )
                tcs = stcs.extract_label_time_course(
                                                        bls,
                                                        src=inverse_operator["src"],
                                                        mode='mean_flip',
                                                        allow_empty=True
                                                )
                tcs_filt = sosfiltfilt(sos, tcs)
                graph_matrix = log_degree_barrier(
                                                tcs_filt,
                                                dist_type=self.params["dist_type"],
                                                alpha=self.params["alpha"],
                                                beta=self.params["beta"]
                                                )
                avg_edge = graph_matrix[bl_idxs[0]][bl_idxs[1]]
                return avg_edge