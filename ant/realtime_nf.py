## don't enter here without a good guide!

import time
from warnings import warn
from pathlib import Path
import json
import uuid

import datetime
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import sosfiltfilt

from mne_lsl.stream import StreamLSL as Stream
from mne_lsl.player import PlayerLSL as Player
from mne_lsl.stream_viewer import StreamViewer as Viewer
from mne_lsl.lsl import local_clock

from mne import set_log_level, read_labels_from_annot, Report
from mne.io import RawArray
from mne.channels import get_builtin_montages, read_dig_captrak
from mne.minimum_norm import apply_inverse_raw, write_inverse_operator, read_inverse_operator
from mne_connectivity import spectral_connectivity_time
from mne_features.univariate import (
                                        compute_app_entropy,
                                        compute_samp_entropy,
                                        compute_spect_entropy,
                                        compute_svd_entropy
                                        )
from ant.tools import *
from ant.tools import _compute_inv_operator
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class NFRealtime:
        """
        Generate a real-time neurofeedback signal on M/EEG recordings.

        Parameters
        ----------
        subject_id : str
                Unique identifier of the subject.
        visit : int
                Visit number (must be >= 1).
        session : {"baseline", "main"}
                Session name; must be either "baseline" or "main".
        subjects_dir : str
                Path to the directory containing subject data.
        mri : bool
                Whether the subject has structural MRI data.
        artifact_rejection : bool | {"autoregressive", "noise_cov", "AJDC"}, default=False
                Method for artifact rejection. If False, no rejection is applied.
        save_raw : bool, default=True
                If True, streamed EEG/MEG data will be saved in the subject directory.
        save_nf_signal : bool, default=True
                If True, the computed neurofeedback signal will be saved.
        verbose : bool | str | int | None, optional
                Control verbosity of the logging output.

        Raises
        ------
        ValueError
                If inputs are invalid (e.g., session not recognized, directory missing).
        """

        VALID_SESSIONS = {"baseline", "main"}
        VALID_ARTIFACT_METHODS = {False, "autoregressive", "noise_cov", "AJDC", "LMS"}

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
                                f"`montage` must be one of the built-in montages {builtin} "
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
                        config = PROJECT_ROOT / "config.yml"
                elif config_file.endswith(".yml") and Path(config_file).is_file():
                        config = config_file
                
                else:
                        raise ValueError("`config_file` must be None or Path to a config file.")
                        
                # --- Assign attributes ---
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

                # --- Setup logging ---
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
                Connect to the LSL stream.

                Parameters
                ----------
                connection_params : dict
                        Ms before and after peak to cut out. If float the cut is symmetric.
                mock_lsl : bool
                        If True a saved recording will be streamed.
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
                                        source_id=source_id
                                        )
                        stream.connect(acquisition_delay=acquisition_delay, timeout=timeout)
                        stream.set_montage(montage, on_missing="warn")
                        stream.pick("eeg")
                        stream.set_meas_date(datetime.datetime.now().replace(tzinfo=datetime.timezone.utc))
                        self.stream = stream
                        
                self.sfreq = stream.info["sfreq"]  
                self.rec_info = stream.info     

                ## copying attributes
                for attr_name in dir(self.stream):
                        attr_value = getattr(self.stream, attr_name)
                        if callable(attr_value) and not attr_name.startswith("__"):
                                setattr(self, attr_name, attr_value)
        

        def record_baseline(self, baseline_duration, winsize=3):
                """
                Start baseline recording to extract useful features.

                Parameters
                ----------
                baseline_duration : float
                        Desired duration of the baseline recording.
                winsize : float
                        Size of the window of data to view. The window will view the last winsize * sfreq samples (ceiled) from the buffer.
                        If None, the entire buffer is returned.
                """
                
                self.baseline_duration = baseline_duration
                #time.sleep(self.bufsize)
                print("start")
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
                # self.stream.disconnect()

        def record_main(
                        self,
                        duration,
                        modality="sensor_power",
                        picks=None,
                        winsize=1,
                        estimate_delays=False,
                        modality_params=None,
                        visualize_nf=True
                        ):
                """
                Start recording to extract neural features.

                Parameters
                ----------
                duration : float
                        Desired duration of the main recording.
                modality : str
                        The method to extract the neural feature used for neurofeedback (see notes).
                picks : str | list | None
                        Channel names to pick, if None, all will be selected.
                winsize : float
                        Size of the window of data to view. The window will view the last winsize * sfreq samples (ceiled) from the buffer.
                        If None, the entire buffer is returned.
                estimate_delays : bool
                        if True, the acquisition and method delays will be saved.
                modality_params : dict | None
                        dictionary of parameters to substitude default parameters. If None parameters from config file will be used.
                """     
                
                self.duration = duration
                self.modality = modality
                self.picks = picks
                self.modality_params = modality_params
                self.winsize = winsize
                self.window_size_s = self.winsize * self.rec_info["sfreq"]
                self.estimate_delays = estimate_delays
                self._sfreq = self.rec_info["sfreq"]
                self.visualize_nf = visualize_nf

                if self.artifact_correction == "LMS":
                        ref_ch_idx = self.rec_info["ch_names"].index(self.ref_channel)

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
                        method_delays = {mod: [] for mod in mods}
                        
                ## add vizualisation
                if self.visualize_nf:
                        self.app = QtWidgets.QApplication([])
                        self.plot_widget = pg.PlotWidget(title="Neurofeedback")
                        self.plot_widget.showGrid(x=True, y=True)
                        self.plot_widget.setLabel('bottom', 'Time', units='s')
                        self.plot_widget.setLabel('left', 'Signal')
                        self.plot_widget.setYRange(-1, len(mods)* 5 + 1)
                        self.plot_widget.addLegend()
                        self.plot_widget.resize(1000, 500)
                        self.plot_widget.show()
                        self.colors_list = ["#5DA5A4", "#9A7DFF", "#FFB085", "#8FBF87", "#D98BA3", "#E0C368"]
                        self.scales_dict = {
                                                "sensor_power": 1e-12,
                                                "band_ratio": 2,
                                                "entropy": 0.5,
                                                } # implement the others as well

                        # self.plot = self.win.addPlot(title=modality)
                        self.curve = self.plot_widget.plot(pen='y')
                        self.time_axis = np.linspace(0, 10, int(self._sfreq)) # show for 10 seconds
                        self.legend = None
                
                ## now the real part!
                nf_data = {mod: [] for mod in mods}
                t_start = local_clock()
                while local_clock() < t_start + self.duration:
                        
                        tic = time.time()
                        ## add filtering
                        if self.filtering:
                                self.stream.filter(l_freq=l_freq, h_freq=h_freq)

                        ## get data
                        data = self.stream.get_data(self.winsize, picks=self.picks)[0] # n_chs * n_times
                        if estimate_delays:
                                acq_delays.append(time.time() - tic)
                        if data.shape[1] != self.window_size_s: continue

                        ## add artifact correction
                        if self.artifact_correction:
                                data = remove_blinks_lms(data, ref_ch_idx=ref_ch_idx, n_taps=5, mu=0.01)
                                print(data.shape)

                        ## compute nf
                        for idx, mod in enumerate(mods):
                                nf_data_, method_delay = nf_mods[idx](data, **precomps[idx])
                                print(nf_data_)
                                nf_data[mod].append(nf_data_)
                                if estimate_delays:
                                        method_delays[mod].append(method_delay)

                        ## add vizualisation
                        if self.visualize_nf:
                                last_vals = [nf_data["sensor_power"][-1], nf_data["band_ratio"][-1]]
                                last_vals = [nf_data[key][-1] for key in mods]
                                self.update_nf_plot(last_vals, labels=mods)
                                self.app.processEvents()
                                time.sleep(0.01)

                self.nf_data = nf_data
                if estimate_delays:
                        self.acq_delays = acq_delays
                        self.method_delays = method_delays
                        self.save(nf_data=True, acq_delay=True, method_delay=True, format="json")
                else:
                        self.save(nf_data=True, acq_delay=False, method_delay=False, format="json")

                self.app.exec()
        
        @property
        def modality_params(self):
                return self._modality_params

        @modality_params.setter
        def modality_params(self, params):
                if params is not None and not isinstance(params, dict):
                        raise ValueError("Can only be a dictionary.")
                else:
                        self._modality_params = params or {}

        def get_default_params(self):
                """
                Return the default parameters in the .yml file.

                """     
                return self._default_params
        
        ## --------------------------- General Methods --------------------------- ##

        def update_nf_plot(self, new_vals, labels=None):
                """
                Update neurofeedback plot in real-time.

                Parameters
                ----------
                new_vals : list or np.ndarray
                        List of last values for each channel/modality.
                        Example: [sensor_power_last, band_ratio_last]
                labels : list of str, optional
                        Labels for each modality (e.g., ["Sensor Power", "Band Ratio"])
                """
                
                n_labels = len(new_vals)
                new_vals = np.array(new_vals, dtype=float)

                ## normalize
                shifts = arr = np.arange(0, n_labels * 5, 5)
                scales = [(0, self.scales_dict[k]) for k in self._mods]

                norm_vals = []
                for val, (min_val, max_val), shift in zip(new_vals, scales, shifts):
                        norm = (val - min_val) / (max_val - min_val) + shift
                        norm_vals.append(norm)
                norm_vals = np.array(norm_vals)
                
                # Initialize plot_data and curves on first call
                if not hasattr(self, "plot_data"):
                        self.plot_data = np.zeros(shape=(n_labels, len(self.time_axis)))
                        self.curves = []
                        colors = [pg.mkPen(color=color, width=2) for color in self.colors_list]
                        for lb in range(n_labels):
                                pen = colors[lb % len(colors)]
                                curve = self.plot_widget.plot(self.time_axis, self.plot_data[lb, :], pen=pen, name=labels[lb])
                                self.curves.append(curve)
                        
                        if self.legend is None:
                                self.legend = self.plot_widget.addLegend()
                                self.legend.setLabelTextSize('14pt')

                self.plot_data = np.roll(self.plot_data, -1, axis=1)
                self.plot_data[:, -1] = norm_vals

                # Update curves
                for lb, curve in enumerate(self.curves):
                        curve.setData(self.time_axis, self.plot_data[lb, :])


        
        def plot_rt(self, bufsize_view=0.2):
                """
                Visualize the signals coming from the LSL stream.
        
                Parameters
                ----------
                bufsize_view : int | float
                        Buffer/window size of the attached StreamReceiver.
                """  
                Viewer(stream_name=self.stream.name).start(bufsize=bufsize_view)
                self.bufsize_view = bufsize_view
        
        def plot_delays(self):
                """
                Plot the histogram of the acqusition/method delays

                """  
                acq_delays_ms = np.array(self.acq_delays) * 1e3 # ms
                method_delays_ms = np.array(self.method_delays) * 1e3 # ms
                colors = ['#1f77b4', '#d62728']
                fig_delays, axs = plt.subplots(1, 2, figsize=(13, 4))

                if self.modality == "sensor_power":
                        bins = 1000
                if self.modality == "source_power":
                        bins = 50 

                for i, data, title in zip(range(2), [acq_delays_ms, method_delays_ms], ["acquisition", "method"]):
                        sns.histplot(data=data, log_scale=False, fill=False, kde=False,
                                        bins=bins, color=colors[i], ax=axs[i])
                        axs[i].spines['top'].set_visible(False)
                        axs[i].spines['right'].set_visible(False)
                        axs[i].set_xlabel("delays (ms)")
                        axs[i].set_title(f"avg {title} delay: {round(data.mean(), 3)} ms")

                        if self.modality == "sensor_power":
                                axs[i].set_xlim([0, 0.1])
                
                if self.modality == "source_power":
                        axs[0].set_xlim([0, 1])
                        axs[1].set_xlim([100, 300])
                
                return fig_delays

        def get_methods_list(self):
                return list(self._default_params.NF_modality.keys())
        
        def compute_inv_operator(self):
                inv = _compute_inv_operator(self.raw_baseline,
                                                subject_fs_id=self.subject_fs_id,
                                                subjects_fs_dir=self.subjects_fs_dir)
                self.inv = inv
                write_inverse_operator(
                                        fname=self.subject_dir / "inv" / f"visit_{self.visit}-inv.fif",
                                        inv=inv,
                                        overwrite=True
                                        )

        def save(self, nf_data=True, acq_delay=True, method_delay=True, format="json"):
                self.stream.disconnect()
                for folder in ["neurofeedback", "delays", "main", "reports"]:
                        (self.subject_dir / folder).mkdir(parents=True, exist_ok=True)

                if format == "json":
                        if nf_data:
                                fname = self.subject_dir / "neurofeedback" / f"nf_data_visit_{self.visit}_{self.modality}.json"
                                with open(fname, "w") as file:
                                        json.dump(self.nf_data, file)
                        if acq_delay:
                                fname = self.subject_dir / "delays" / f"acq_delay_visit_{self.visit}_{self.modality}.json"
                                with open(fname, "w") as file:
                                        json.dump(self.acq_delays, file)
                        if method_delay:
                                fname = self.subject_dir / "delays" / f"method_delay_visit_{self.visit}_{self.modality}.json"
                                with open(fname, "w") as file:
                                        json.dump(self.method_delays, file)                                
        
        def create_report(self, overwrite=True):
                """
                Create a report in HTML format for the subject.
                
                Parameters
                ----------
                report_path : path-like | None
                        Directory to save the report.
                overwrite : bool
                        If True, overwrite the destination file if it exists.
                """  
                report = Report(title=f"Neurofeedback Session with {self.modality} modality")
                report.add_raw(self.raw_baseline, title="Baseline recording", psd=False, butterfly=False)

                methods_list = list(self._default_params.NF_modality.keys())
                if self.modality in methods_list[:5]:
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

                if self.modality == "source_power":
                        figure_brain = plot_glass_brain(bl1=self.params.src.bl, bl2=None)
                        report.add_figure(fig=figure_brain, title=f"selected brain labels")
                if self.modality in methods_list[6:]:
                        figure_brain = plot_glass_brain(bl1=self.params.src.bl_1,
                                                        bl2=self.params.src.bl_2)
                        report.add_figure(fig=figure_brain, title=f"selected brain labels")
                
                fig_delays = self.plot_delays()
                report.add_figure(fig=fig_delays, title=f"estimated delays for method {self.modality}")
                
                if report_path is None:
                        report_path = Path.cwd().parent.parent / "reports" 
                        if not report_path.exists(): report_path.mkdir()
                report_path = report_path / f"nf_report_subject_{self.subject_id}_modality_{self.modality}.html"
                report.save(report_path, overwrite=overwrite)

        ## --------------------------- Neural Feature Extraction Methods (preparation) --------------------------- ##

        def _sensor_power_prep(self):
                precomp = dict(
                                sfreq=self.rec_info["sfreq"],
                                frange=self.params["frange"],
                                method=self.params["method"],
                                relative=self.params["relative"]
                                )
                return precomp


        def _argmax_freq_prep(self):
                
                assert hasattr(self, "raw_baseline"), "Baseline recording should be done prior to this step."
                ## extracting the aperiodic components from the baseline recording
                ap_params, _ = estimate_aperiodic_component(
                                                                raw_baseline=self.raw_baseline,
                                                                picks=self.picks,
                                                                method=self.params["method"]
                                                                )
                fft_window, freq_band, freq_band_idxs, _ = \
                                                        compute_fft(
                                                                sfreq=self._sfreq,
                                                                winsize=self.winsize,
                                                                freq_range=self.params["frange"],
                                                                freq_res=1
                                                                )
                
                ap_model = (10 ** ap_params[0]) / (freq_band ** ap_params[1])
                gaussian = lambda freq_band, amplitude, mean, stddev: \
                                amplitude * np.exp(-(freq_band - mean)**2 / (2 * stddev**2))
                precomp = {
                        "fft_window": fft_window,
                        "freq_band": freq_band,
                        "freq_band_idxs": freq_band_idxs,
                        "ap_model": ap_model,
                        "gaussian": gaussian
                        }
                return precomp

        def _band_ratio_prep(self):
                precomp = dict(
                                sfreq=self.rec_info["sfreq"],
                                frange_1=self.params["frange_1"],
                                frange_2=self.params["frange_2"],
                                method=self.params["method"]
                                )
                return precomp

        def _individual_peak_power_prep(self):
                ## extracting the periodic components from the baseline recording
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
                sos = butter_bandpass(
                                        self.params["frange"][0],
                                        self.params["frange"][1],
                                        self._sfreq,
                                        order=5
                                        )
                precomp = {"sos": sos}
                return precomp

        def _source_power_prep(self):
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
        def _argmax_freq(self, data, fft_window, freq_band, freq_band_idxs, ap_model, gaussian):
                data = np.multiply(data, fft_window)
                fftval = np.abs(np.fft.rfft(data, axis=1) / data.shape[-1])
                total_power = np.average(np.square(fftval[:, freq_band_idxs]).T)
                periodic_power = (total_power - ap_model).mean(axis=0)
                try:
                        popt, _ = curve_fit(gaussian, freq_band, periodic_power)
                        individual_peak = popt[1]
                except RuntimeError:
                        individual_peak = 0 
                        warn(f"fitting failed and individual peak value is set to 0.")
                return individual_peak
        
        @timed
        def _band_ratio(self, data, sfreq, frange_1, frange_2, method):
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
                data = np.multiply(data, fft_window)
                fftval = np.abs(np.fft.rfft(data, axis=1) / data.shape[-1])
                power = np.average(np.square(fftval[:, individual_freq_band_idxs]).T)
                return power

        @timed
        def _entropy(self, data, sos):
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