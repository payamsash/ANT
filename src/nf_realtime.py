## don't enter here without a good guide!

import time
from warnings import warn
from pathlib import Path
import json
import pickle

import datetime
from benedict import benedict
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import sosfiltfilt
import seaborn as sns
import matplotlib.pyplot as plt

# from bsl import StreamPlayer, StreamReceiver, StreamRecorder, StreamViewer
# from bsl.utils import Timer
from mne_lsl.stream import StreamLSL as Stream
from mne_lsl.player import PlayerLSL as Player
from mne_lsl.stream_viewer import StreamViewer as Viewer
from mne_lsl.lsl import local_clock


import mne
from mne import set_log_level
from mne.minimum_norm import apply_inverse_raw
from mne_connectivity import spectral_connectivity_time
from mne_features.univariate import (compute_app_entropy,
                                compute_samp_entropy,
                                compute_spect_entropy,
                                compute_svd_entropy)

from tools import (canonical_frange_str_to_list, update_params, timed, compute_fft,
                estimate_aperiodic_component, log_degree_barrier,
                create_inverse_operator, butter_bandpass, plot_glass_brain)

set_log_level("WARNING")

class NF_Realtime:
        """
        Class for generating a real time neurofeedback signal on M/EEG recordings.
        The list of parameters is in default_params.yml.

        Parameters
        ----------
        subject_id : str 
                Subject name.
        freq_range : tuple | list | str
                frequency range of the interest.
        picks : str | array_like | slice | None
                Channels to include. Slices and lists of integers will be interpreted as channel indices.
        artifact_correction : None | str
                if None, no artifact rejection will be applied.
                The possible options for artifact corrections are "autoregressive", "noise_cov" and "AJDC". 
        params_path: None | path-like
                Path to a .yml file including dictionary with parameters.
                Default values can be retrieved with nf.get_default_params().
                for each class method, a dictionary could be passed which will update the default params.
        save_data : bool
                If True, the streamed data will be saved in the current working directory.
        verbose : bool | str | int | None
                Control verbosity of the logging output.
        """

        def __init__(
                self,
                subject_id,
                freq_range,
                picks=None,
                artifact_correction=False,
                params_path=None,
                verbose=None,
        ):
                
                self.subject_id = subject_id
                self.freq_range = canonical_frange_str_to_list(freq_range) if isinstance(freq_range, str) else freq_range
                self.picks = picks 
                self.artifact_correction = artifact_correction
                self.params_path = Path.cwd().parent.parent / "default_params" / "default_params.yml" if params_path is None else params_path
                self.verbose = verbose
                self._default_params = benedict.from_yaml(self.params_path)


        @property
        def default_params(self):
                return self._default_params
        
        @default_params.setter
        def default_params(self, params_path):
                if not params_path.endswith(".yml"):
                        raise ValueError("Can only be assigned to a .yml path containing the parameter values.")
                self._default_params = benedict.from_yaml(self.params_path)

        def connect_lsl(self, session, connection_params=None, mock_lsl=False):    
                """
                Connect to the LSL stream.

                Parameters
                ----------
                connection_params : dict
                        Ms before and after peak to cut out. If float the cut is symmetric.
                mock_lsl : bool
                        If True a saved recording will be streamed.
                """
                ## update the parameters
                self._connection_params = update_params(connection_params, self._default_params, "LSL")
                self.mock_lsl = mock_lsl

                ## disconnect the previous streaming if any
                if hasattr(self, "stream"): 
                        if self.stream.connected:
                                self.stream.disconnect()
                
                ## get the recording info and create the stream
                if self.mock_lsl:
                        self._player = Player(fname=self._connection_params.Mock_connection.fname,
                                        chunk_size=self._connection_params.Mock_connection.chunk_size).start()
                else:
                        _stream = Stream(bufsize=self.connection_params.Baseline_acquisition.bufsize,
                                                name=self.connection_params.Recorder.stream_name)

                if session == "baseline":
                        stream = Stream(self._connection_params.Baseline_acquisition.bufsize).connect(**self._connection_params.Connection)
                if session == "main":
                        stream = Stream(self._connection_params.Acquisition.bufsize).connect(**self._connection_params.Connection)

                stream.pick("eeg")  # now only eeg
                # stream.set_eeg_reference("average")
                self._sfreq = stream.info["sfreq"]  
                self.rec_info = stream.info
                self.base_winsize = self._connection_params.Baseline_acquisition.winsize
                self.winsize = self._connection_params.Acquisition.winsize
                self.stream = stream
                self.stream.set_meas_date(datetime.datetime.now().replace(tzinfo=datetime.timezone.utc))

                ## copying attributes
                for attr_name in dir(self.stream):
                        attr_value = getattr(self.stream, attr_name)
                        if callable(attr_value) and not attr_name.startswith("__") and attr_name != "plot_sensors":
                                setattr(self, attr_name, attr_value)
                
        
        @property
        def connection_params(self):
                return self._connection_params

        @connection_params.setter
        def connection_params(self, params):
                if not isinstance(params, dict):
                        raise ValueError("Can only be a dictionary.")
                else:
                        self._connection_params = params
        
        
        def record_baseline(self, baseline_duration):
                """
                Start baseline recording to extract useful features.

                Parameters
                ----------
                baseline_duration : float
                        Desired duration of the baseline recording.
                """
                
                self.baseline_duration = baseline_duration
                time.sleep(self._connection_params.Baseline_acquisition.bufsize)
                print("start")
                data = [] 
                t_start = local_clock()
                while local_clock() < t_start + self.baseline_duration:
                        
                        data.append(self.stream.get_data(self.base_winsize)[0])
                        time.sleep(self._connection_params.Baseline_acquisition.winsize)

                data = np.concatenate(np.array(data), axis=1)
                self.raw_baseline = mne.io.RawArray(data, self.rec_info)
                self.stream.disconnect()
                if self.mock_lsl:
                        self._player.stop()

        def record_nf(self, duration, modality="sensor_power", modality_params=None):
                """
                Start baseline recording to extract useful features.

                Parameters
                ----------
                duration : float
                        Desired duration of the main recording.
                modality : str
                        The method to extract the neural feature used for neurofeedback (see notes).
                modality_params : dict | None
                        dictionary of parameters to substitude default parameters.
                """     
                ## update the parameters
                modality_params = update_params(modality_params, self._default_params, "NF_modality")
                self.duration = duration
                self.modality = modality
                self._modality_params = modality_params[modality]
                self._window_size = self._connection_params.Acquisition.winsize * self.rec_info["sfreq"]
                nf_mod_prep = getattr(self, f"_{modality}_prep", None)
                nf_mod = getattr(self, f"_{modality}", None)
                
                if not callable(nf_mod):
                        raise NotImplementedError(f"{modality} modality not implemented yet.")
                if "source" in self.modality:
                        assert self.picks is None, "picks should be None for source methods." 

                # compute the necessary stuff for the modality
                outputs = nf_mod_prep()

                nf_data, acq_delays, method_delays = ([], [], [])
                t_start = local_clock()
                while local_clock() < t_start + self.duration:
                
                        # record
                        tic = time.time()
                        data = self.stream.get_data(self.winsize, picks=self.picks)[0] # n_chs * n_times
                        acq_delays.append(time.time() - tic)

                        ## print(data.shape)
                        
                        # check shape
                        if data.shape[1] != self._window_size: continue

                        # add artifact correction
                        
                        #compute nf
                        ## print("hh")
                        nf_data_, method_delay = nf_mod(data, *outputs)

                        # append
                        nf_data.append(nf_data_)
                        method_delays.append(method_delay)

                self.nf_data = nf_data
                self.acq_delays = acq_delays
                self.method_delays = method_delays
        
        
        @property
        def modality_params(self):
                return self._modality_params

        @modality_params.setter
        def modality_params(self, params):
                if not isinstance(params, dict):
                        raise ValueError("Can only be a dictionary.")
                else:
                        self._modality_params = params
        

        def _sensor_power_prep(self):

                fft_window, _, freq_band_idxs, _ = compute_fft(sfreq=self._sfreq,
                                                        winsize=self._connection_params.Acquisition.winsize,
                                                        freq_range=self.freq_range,
                                                        freq_res=self._modality_params.fft.freq_res)
                
                return fft_window, freq_band_idxs
        
        @timed
        def _sensor_power(self, data, fft_window, freq_band_idxs):
                
                data = np.multiply(data, fft_window)
                fftval = np.abs(np.fft.rfft(data, axis=1) / data.shape[-1])
                power = np.average(np.square(fftval[:, freq_band_idxs]).T)

                return power


        def _argmax_freq_prep(self):

                assert hasattr(self, "raw_baseline"), "Baseline recording should be done prior to this step."

                ## extracting the aperiodic components from the baseline recording
                ap_params, _ = estimate_aperiodic_component(raw_baseline=self.raw_baseline,
                                        picks=self.picks, psd_params=self._modality_params.psd,
                                        fitting_params=self._modality_params.psd_fitting,
                                        verbose=self.verbose)
                fft_window, freq_band, freq_band_idxs, _ = \
                                compute_fft(sfreq=self._sfreq,
                                                winsize=self._connection_params.Acquisition.winsize,
                                                freq_range=self.freq_range,
                                                freq_res=self._modality_params.fft.freq_res)
                
                ap_model = (10 ** ap_params[0]) / (freq_band ** ap_params[1])
                gaussian = lambda freq_band, amplitude, mean, stddev: \
                                amplitude * np.exp(-(freq_band - mean)**2 / (2 * stddev**2))

                return fft_window, freq_band, freq_band_idxs, ap_model, gaussian

        @timed
        def _argmax_freq(self, data, fft_window, freq_band, freq_band_idxs, ap_model, gaussian):
        
                data = np.multiply(data, fft_window)
                fftval = np.abs(np.fft.rfft(data, axis=1) / data.shape[-1])
                total_power = np.average(np.square(fftval[:, freq_band_idxs]).T)
                periodic_power = (total_power - ap_model).mean(axis=0)
                try:
                        popt, _ = curve_fit(gaussian, freq_band, periodic_power,
                                                **self._modality_params.curve_fit)
                        individual_peak = popt[1]
                except RuntimeError:
                        individual_peak = 0 
                        warn(f"fitting failed and individual peak value is set to 0.")
                
                return individual_peak
        

        def _band_ratio_prep(self):

                fft_windows, _, freq_band_idxss, _ = zip(*(compute_fft(sfreq=self._sfreq,
                                                        winsize=self._connection_params.Acquisition.winsize,
                                                        freq_range=self._modality_params.fft[frange],
                                                        freq_res=self._modality_params.fft.freq_res) \
                                                        for frange in ["frange_1", "frange_2"]))
                                
                return fft_windows, freq_band_idxss
        
        @timed
        def _band_ratio(self, data, fft_windows, freq_band_idxss):
                
                data1 = np.multiply(data, fft_windows[0])
                fftval = np.abs(np.fft.rfft(data1, axis=1) / data.shape[-1])
                power1 = np.average(np.square(fftval[:, freq_band_idxss[0]]).T)
                data2 = np.multiply(data, fft_windows[1])
                fftval = np.abs(np.fft.rfft(data2, axis=1) / data.shape[-1])
                power2 = np.average(np.square(fftval[:, freq_band_idxss[1]]).T)
                ratio = power1 / power2

                return ratio

        
        
        def _individual_peak_power_prep(self):

                ## extracting the periodic components from the baseline recording
                _, peak_params_ = estimate_aperiodic_component(raw_baseline=self.raw_baseline,
                                                                picks=self.picks, psd_params=self._modality_params.psd,
                                                                fitting_params=self._modality_params.psd_fitting,
                                                                verbose=self.verbose)
                peak_params = [peak_param[0] for peak_param in peak_params_ if self.freq_range[0] < peak_param[0] < self.freq_range[1]]
                if len(peak_params) == 1:
                        cf = peak_params[0]
                else:
                        cf = (self.freq_range[0] + self.freq_range[1]) / 2
                        warn(f"center frequency was set to the middle frequency in the selected frequency range.")
                
                ## compute power in a small range around individual peak
                fft_window, _, _, frequencies = compute_fft(sfreq=self._sfreq,
                                winsize=self.connection_params.Acquisition.winsize,
                                freq_range=self.freq_range,
                                freq_res=self._modality_params.fft.freq_res)
                freq_var = self._modality_params.fft.freq_var
                individual_freq_band_idxs = np.where(np.logical_and(cf - freq_var <= frequencies,
                                                        frequencies <= cf + freq_var))[0]
                                
                return fft_window, individual_freq_band_idxs
        
        @timed
        def _individual_peak_power(self, data, fft_window, individual_freq_band_idxs):
                
                data = np.multiply(data, fft_window)
                fftval = np.abs(np.fft.rfft(data, axis=1) / data.shape[-1])
                power = np.average(np.square(fftval[:, individual_freq_band_idxs]).T)
                
                return power



        def _entropy_prep(self):

                entropy_method = self._modality_params.entropy_method
                
                return entropy_method
        
        @timed
        def _entropy(self, data, entropy_method):
                
                match entropy_method:
                        case "AppEn":
                                ents = compute_app_entropy(data, self._modality_params.ent.emb_app,
                                                        self._modality_params.ent.metric_app)
                        case "SampEn":
                                ents = compute_samp_entropy(data, self._modality_params.ent.emb_sample,
                                                        self._modality_params.ent.metric_sample)
                        case "Spectral":
                                ents = compute_spect_entropy(sfreq=self._sfreq, data=data,
                                                        psd_method=self._modality_params.ent.psd_method)
                        case "SVD":
                                ents = compute_svd_entropy(data, self._modality_params.ent.tau,
                                                        self._modality_params.ent.emb_svd)

                return ents.mean()

        def _source_power_prep(self):

                fft_window, _, freq_band_idxs, _ = compute_fft(sfreq=self._sfreq,
                                winsize=self._modality_params.fft.winsize,
                                freq_range=self.freq_range,
                                freq_res=self._modality_params.fft.freq_res)
                bls = mne.read_labels_from_annot(subject=self._modality_params.src.subject,
                                                parc=self._modality_params.src.atlas,
                                                subjects_dir=self._modality_params.src.subjects_dir,
                                                verbose=self.verbose)
                bl_names = [bl.name for bl in bls]
                bl_idx = bl_names.index(self._modality_params.src.bl)
                brain_label = bls[bl_idx]
                lambda2 = 1.0 / self._modality_params.src.snr ** 2
                inverse_operator = create_inverse_operator(self.raw_baseline, self._modality_params.src,
                                                        verbose=self.verbose)
                
                return fft_window, freq_band_idxs, lambda2, brain_label, inverse_operator
        
        @timed
        def _source_power(self, data, fft_window, freq_band_idxs, lambda2, brain_label, inverse_operator):
                
                raw_data = mne.io.RawArray(data, self.rec_info, verbose=self.verbose)
                raw_data.set_eeg_reference("average", projection=True)
                ## compute source activation and then power
                stc_data = apply_inverse_raw(raw_data, inverse_operator, lambda2=lambda2,
                                        label=brain_label, **self._modality_params.inv_modeling,
                                        verbose=self.verbose).data
                stc_data = np.multiply(stc_data, fft_window)
                fftval = np.abs(np.fft.rfft(stc_data, axis=1) / stc_data.shape[-1])
                stc_power = np.average(np.square(fftval[:, freq_band_idxs]).T)

                return stc_power


        def _sensor_connectivity_prep(self):

                ch_names = self.rec_info["ch_names"]
                chs = self._modality_params.channels.ch_names
                indices = [(np.array([ch_names.index(ch1), ch_names.index(ch2)])) for ch1, ch2 in zip(chs[0], chs[1])]
                indices = tuple(indices)
                freqs = np.linspace(self.freq_range[0], self.freq_range[1],
                                        self._modality_params.channels.freq_res)
                
                return indices, freqs
        
        @timed
        def _sensor_connectivity(self, data, indices, freqs):
                
                con = spectral_connectivity_time(data=data[np.newaxis,:], freqs=freqs,
                                                indices=indices, average=False,
                                                sfreq=self._sfreq, 
                                                fmin=self.freq_range[0],
                                                fmax=self.freq_range[1], 
                                                faverage=True,
                                                **self._modality_params.con,
                                                verbose=self.verbose)
                con_data = np.squeeze(con.get_data(output='dense'))[indices].mean()
                
                return con_data



        def _source_connectivity_prep(self):

                assert self._modality_params.src.bl_1[-2:]=="lh", "first brain label should be selected from left hemisphere."
                assert self._modality_params.src.bl_2[-2:]=="rh", "second brain label should be selected from right hemisphere."
                
                ## initiating the source space
                bls = mne.read_labels_from_annot(subject=self._modality_params.src.subject,
                                                parc=self._modality_params.src.atlas,
                                                verbose=self.verbose)
                bl_names = [bl.name for bl in bls]
                merged_label = bls[bl_names.index(self._modality_params.src.bl_1)] + \
                                bls[bl_names.index(self._modality_params.src.bl_2)]
                
                lambda2 = 1.0 / self._modality_params.src.snr ** 2
                inverse_operator = create_inverse_operator(self.raw_baseline, self._modality_params.src,
                                                        verbose=self.verbose)
                freqs = np.linspace(self.freq_range[0], self.freq_range[1],
                                        self._modality_params.fft.freq_res)
                
                sos = butter_bandpass(self.freq_range[0], self.freq_range[1], self._sfreq,
                                        order=self._modality_params.channels.order)
                
                return merged_label, lambda2, inverse_operator, freqs, sos
        
        @timed
        def _source_connectivity(self, data, merged_label, lambda2, inverse_operator, freqs, sos):
                
                raw_data = mne.io.RawArray(data, self.rec_info, verbose=self.verbose)
                
                stcs = apply_inverse_raw(raw_data, inverse_operator, lambda2=lambda2,
                        label=merged_label, **self._modality_params.inv_modeling,
                        verbose=self.verbose)
                stc_lh_data = stcs.lh_data.mean(axis=0)
                stc_rh_data = stcs.rh_data.mean(axis=0)
                
                if self._modality_params.con.method == "corr":

                        data_filt = sosfiltfilt(sos, np.array([stc_lh_data, stc_rh_data]))
                        con_data = np.corrcoef(data_filt[0], data_filt[1])[0][1]
                else:
                        con = spectral_connectivity_time(data=np.array([[stc_lh_data, stc_rh_data]]),
                                                        freqs=freqs,
                                                        indices=None,
                                                        average=False,
                                                        sfreq=self._sfreq,
                                                        fmin=self.freq_range[0],
                                                        fmax=self.freq_range[1],
                                                        faverage=True,
                                                        **self._modality_params.con,
                                                        verbose=self.verbose)
                        con_data = np.squeeze(con.get_data(output='dense'))[1][0]

                return con_data

        def _sensor_graph_prep(self):

                ch_names = self.rec_info["ch_names"]
                chs = self._modality_params.channels.ch_names
                indices = [(np.array([ch_names.index(ch1), ch_names.index(ch2)])) for ch1, ch2 in zip(chs[0], chs[1])]
                indices = tuple(indices)

                ## initiating the filter
                sos = butter_bandpass(self.freq_range[0], self.freq_range[1], self._sfreq,
                                        order=self._modality_params.channels.order)
                return indices, sos
        
        @timed
        def _sensor_graph(self, data, indices, sos):
                
                data_filt = sosfiltfilt(sos, data)
                graph_matrix = log_degree_barrier(data_filt, **self._modality_params.graph)
                avg_edge = np.array([graph_matrix[idxs] for idxs in indices]).mean()

                return avg_edge

        
        def _source_graph_prep(self):

                ## initiating the source space
                bls = mne.read_labels_from_annot(subject=self._modality_params.src.subject,
                                                parc=self._modality_params.src.atlas,
                                                verbose=self.verbose)[:-1] # should work
                bl_names = [bl.name for bl in bls]
                (bl_idx1, bl_idx2) = (bl_names.index(self._modality_params.src.bl_1),
                                        bl_names.index(self._modality_params.src.bl_2)) 
                lambda2 = 1.0 / self._modality_params.src.snr ** 2
                inverse_operator, src = create_inverse_operator(self.raw_baseline, self._modality_params.src,
                                                                return_src=True, verbose=self.verbose)

                sos = butter_bandpass(self.freq_range[0], self.freq_range[1],
                                        self._sfreq, order=self._modality_params.src.order)
                
                return bls, bl_idx1, bl_idx2, lambda2, inverse_operator, src, sos
        
        @timed
        def _source_graph(self, data, bls, bl_idx1, bl_idx2, lambda2, inverse_operator, src, sos):
                
                raw_data = mne.io.RawArray(data, self.rec_info, verbose=self.verbose)
                stcs = apply_inverse_raw(raw_data, inverse_operator, lambda2=lambda2, 
                                        **self._modality_params.inv_modeling,
                                        verbose=self.verbose)
                tcs = stcs.extract_label_time_course(bls, src=src,
                                                **self._modality_params.label_extraction,
                                                verbose=self.verbose)
                tcs_filt = sosfiltfilt(sos, tcs)
                graph_matrix = log_degree_barrier(tcs_filt, **self._modality_params["graph"])
                avg_edge = graph_matrix[bl_idx1][bl_idx2]

                return avg_edge

        


        
        def save(self, eeg=False, nf_data=True, acq_delay=True, method_delay=True, format="json"):

                self.stream.disconnect() 
                main_path = Path.cwd().parent.parent / "subjects" / f"{self.subject_id}"
                if not main_path.exists(): main_path.mkdir()
                
                if eeg:
                        raise NotImplementedError
                if nf_data:
                        fname = main_path / f"{self.modality}_nf_data.json"
                        if format == "json":
                                with open(fname, "w") as file:
                                        json.dump(self.nf_data, file)
                        if format == "pkl":
                                with open(fname, 'wb') as file:
                                        pickle.dump(self.nf_data, file)

                if acq_delay:
                        fname = main_path / f"{self.modality}_acq_delay.json"
                        if format == "json":
                                with open(fname, "w") as file:
                                        json.dump(self.acq_delays, file)
                        if format == "pkl":
                                with open(fname, 'wb') as file:
                                        pickle.dump(self.acq_delays, file)

                if method_delay:
                        fname = main_path / f"{self.modality}_method_delay.json"
                        if format == "json":
                                with open(fname, "w") as file:
                                        json.dump(self.method_delays, file)
                        if format == "pkl":
                                with open(fname, 'wb') as file:
                                        pickle.dump(self.method_delays, file)


        def get_push_interval(self):
        
                interval = self._connection_params.Mock_connection.chunk_size / self._sfreq
                return interval
        
        def get_default_params(self):
                """
                Return the default parameters in the .yml file.

                """     
                return self._default_params
        
        def plot_rt(self, bufsize_view=0.2):
                """
                Visualize the signals coming from the LSL stream.
        
                Parameters
                ----------
                bufsize_view : int | float
                        Buffer/window size of the attached StreamReceiver.
                """  
                warn("Has some issues for now, will be fixed ....")
                Viewer(stream_name=self.stream.name).start(bufsize=bufsize_view)
                self.bufsize_view = bufsize_view

        def plot_electrodes(self, kind="topomap"):
                """
                Plot sensors positions. Used sensors will be colored in red.
                
                Parameters
                ----------
                kind : str
                        Whether to plot the sensors as 3d, topomap or as an interactive sensor selection dialog. 
                        Available options 'topomap', '3d', 'select'.
                """  
                if self.picks is not None:
                        self.rec_info["bads"].extend(self.picks)
                else:
                        self.rec_info["bads"].extend(self.rec_info["ch_names"])
                self.fig_sensors = mne.viz.plot_sensors(info=self.rec_info, kind=kind, verbose=self.verbose)
                self.rec_info["bads"] = []
        
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
        
        
        def create_report(self, report_path=None, overwrite=True):
                """
                Create a report in HTML format for the subject.
                
                Parameters
                ----------
                report_path : path-like | None
                        Directory to save the report.
                overwrite : bool
                        If True, overwrite the destination file if it exists.
                """  
                report = mne.Report(title=f"Neurofeedback Session with {self.modality} modality")
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
                        mne.viz.plot_sensors(info=self.rec_info, kind="topomap", axes=ax1, show=False, verbose=False)
                        mne.viz.plot_sensors(info=self.rec_info, kind="3d", axes=ax2, show=False, verbose=False)
                        ax2.axis("off")
                        self.rec_info["bads"] = []
                        report.add_figure(fig=fig_sensors, title="Sensors")

                if self.modality == "source_power":
                        figure_brain = plot_glass_brain(bl1=self._modality_params.src.bl, bl2=None)
                        report.add_figure(fig=figure_brain, title=f"selected brain labels")
                if self.modality in methods_list[6:]:
                        figure_brain = plot_glass_brain(bl1=self._modality_params.src.bl_1,
                                                        bl2=self._modality_params.src.bl_2)
                        report.add_figure(fig=figure_brain, title=f"selected brain labels")
                
                fig_delays = self.plot_delays()
                report.add_figure(fig=fig_delays, title=f"estimated delays for method {self.modality}")
                
                if report_path is None:
                        report_path = Path.cwd().parent.parent / "reports" 
                        if not report_path.exists(): report_path.mkdir()
                report_path = report_path / f"nf_report_subject_{self.subject_id}_modality_{self.modality}.html"
                report.save(report_path, overwrite=overwrite)

        
        
        

