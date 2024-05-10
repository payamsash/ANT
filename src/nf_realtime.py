# don't enter here without a good guide!

import yaml
import time
import datetime
from warnings import warn
from pathlib import Path

import numpy as np
import mne
from scipy.optimize import curve_fit
from mne_features.univariate import (compute_app_entropy,
                                compute_samp_entropy,
                                compute_spect_entropy,
                                compute_svd_entropy)

from bsl import StreamPlayer, StreamReceiver, StreamRecorder, StreamViewer
from bsl.utils import Timer
from mne_lsl.stream import StreamLSL 
from mne_lsl.player import PlayerLSL 
from mne.minimum_norm import apply_inverse_raw
from mne_connectivity import spectral_connectivity_time
from tools import (update_params, receive_data, compute_fft,
                estimate_aperiodic_component, log_degree_barrier,
                create_inverse_operator, butter_bandpass,
                bandpass_filter)





class NF_Realtime:
        """
        Class for generating a real time neurofeedback on M/EEG recordings.
        The list of parameters is in default_params/nf_realtime_params.yaml.


        Parameters
        ----------
        stream_name : str
                Name of the LSL stream.
        freq_range : tuple | list | str
                Frequency range in EEG recordings.
        modality : str
                Should be either "individual_peak_tracking" to track individual frequency where the peak occurs
                or "power" to estimate amount of periodic power present in a selected frequency range or
                "individual_peak_power" to detect individual peak and return power around the individual peak.
        duration: float
                Duration of the recording.
        baseline_duration : float
                Duration of the recording used to estimate aperiodic model of the subjects PSD.
        artifact_rejection : bool
                online artifact rejection using pyriemann method.
        return_delay : bool
                If "True", returns the delay values (in seconds) per iteration.
        params : dict
                Parameters for LSL inlet.
        verbose : bool | str | int | None
                Control verbosity of the logging output.

        """

        def __init__(
                self,
                freq_range,
                picks=None,
                artifact_rejection=False,
                return_delay=True,
                params=None, # should be a path to yaml file
                verbose=False,
        ):

                self.freq_range = freq_range 
                self.picks = picks 
                self.artifact_rejection = artifact_rejection
                self.return_delay = return_delay
                self.verbose = verbose

                params_path = "/Users/payamsadeghishabestari/codes/mne-nf/src/default_params/default_params.yml" # should fix here
                
                with open(params_path, "r") as f:
                        self.params = yaml.safe_load(f)
        

        def connect(self, connection_params=None, mock_lsl=False):    
                
                ## update the parameters
                connection_params = update_params(connection_params, self.params, "LSL")
                self.connection_params = connection_params

                ## disconnect the previous streaming if any
                if hasattr(self, "stream"):
                        self.stream.stop()
                
                ## get the recording info and create the stream
                if mock_lsl:
                        mne_stream = PlayerLSL(fname=self.connection_params["Mock_connection"]["fif_file"])
                        stream = StreamPlayer(**self.connection_params["Mock_connection"]) 
                else:
                        mne_stream = StreamLSL(bufsize=self.connection_params["Baseline_acquisition"]["bufsize"],
                                                name=self.connection_params["Recorder"]["stream_name"])
                        stream = StreamRecorder(**self.connection_params["Recorder"], verbose=self.verbose)

                stream.start()
                self.rec_info = mne_stream.info
                self.stream = stream
        
        def record_baseline(self, baseline_duration):
                
                self.baseline_duration = baseline_duration
                
                receiver = StreamReceiver(stream_name=self.stream.stream_name, **self.connection_params["Baseline_acquisition"])
                time.sleep(self.connection_params["Baseline_acquisition"]["bufsize"]) 
                t0 = Timer()
                raws_baseline_list = [] 
                while t0.sec() < self.baseline_duration:
                        print(t0.sec())
                        receiver.acquire()
                        raws_baseline_list.append(receiver.get_window(return_raw=True)[0])
                        time.sleep(self.connection_params["Baseline_acquisition"]["winsize"])
                raw_baseline = mne.concatenate_raws(raws_baseline_list)

                ## montaging the recording
                raw_baseline.drop_channels(ch_names="TRIGGER", on_missing="warn")
                raw_baseline.set_montage(**self.connection_params["montage"])
                self.raw_baseline = raw_baseline

                if self.picks is None:
                        self.pick_idxs = range(len(self.raw_baseline.info["ch_names"]))
                else:
                        self.pick_idxs = np.array([self.raw_baseline.info["ch_names"].index(pick) for pick in self.picks])

                receiver.disconnect(stream_name=self.stream.stream_name)

        def record_main(self,
                duration,
                modality="sensor_power",
                modality_params=None,
                ):        

                ## update the parameters
                modality_params = update_params(modality_params, self.params, "NF_modality")
                self.duration = duration
                self.modality = modality
                self.modality_params = modality_params[modality]
                self.window_size = self.connection_params["Acquisition"]["winsize"] * self.rec_info["sfreq"]
                acq_delays, method_delays  = ([], [])

                match modality:
                        case "individual_peak_tracking":
                                assert hasattr(self, "raw_baseline"), "Baseline recording should be done prior to this step."

                                ## extracting the aperiodic components from the baseline recording
                                ap_params, _ = estimate_aperiodic_component(raw_baseline=self.raw_baseline,
                                                        picks=self.picks, psd_params=self.modality_params["psd"],
                                                        fitting_params=self.modality_params["psd_fitting"],
                                                        verbose=self.verbose)
                                fft_window, freq_band, freq_band_idxs, frequencies = compute_fft(sfreq=self.raw_baseline.info["sfreq"],
                                                winsize=self.connection_params["Acquisition"]["winsize"],
                                                freq_range=self.freq_range,
                                                freq_res=self.modality_params["fft"]["freq_res"])
                                ap_model = (10 ** ap_params[0]) / (freq_band ** ap_params[1])
                                gaussian = lambda freq_band, amplitude, mean, stddev: \
                                                amplitude * np.exp(-(freq_band - mean)**2 / (2 * stddev**2))


                                ## start the real recording 
                                receiver = StreamReceiver(stream_name=self.stream.stream_name,
                                                           **self.connection_params["Acquisition"])
                                t0 = Timer()
                                while t0.sec() < self.duration:
                                        
                                        ## acquiring the data
                                        data, acq_delay = receive_data(receiver, self.pick_idxs)
                                        acq_delays.append(acq_delay)
                                        
                                        ## skip incomplete windows at the beginning of the recording
                                        if data.shape[0] != self.window_size:
                                                continue  

                                        ## FFT + fitting the Gaussian function to the data
                                        start_method_time = time.time()
                                        data *= fft_window[:,np.newaxis]
                                        fftval = np.abs(np.fft.rfft(data, axis=1) / data.shape[-1])
                                        total_power = np.square(fftval[:, freq_band_idxs])
                                        periodic_power = (total_power - ap_model).mean(axis=0)

                                        try:
                                                popt, _ = curve_fit(gaussian, freq_band, periodic_power,
                                                                        **self.modality_params["curve_fit"])
                                                individual_peak = popt[1]
                                        except RuntimeError:
                                                individual_peak = 0 
                                                warn(f"fitting failed and individual peak value is set to 0.")
                        
                                        method_delays.append(time.time() - start_method_time)
                                        
                                        return individual_peak 
                                        
                
                        case "sensor_power":

                                fft_window, freq_band, freq_band_idxs, _ = compute_fft(sfreq=self.rec_info["sfreq"],
                                                winsize=self.connection_params["Acquisition"]["winsize"],
                                                freq_range=self.freq_range,
                                                freq_res=self.modality_params["fft"]["freq_res"])
                                
                                ## start the real recording 
                                receiver = StreamReceiver(stream_name=self.stream.stream_name,
                                                           **self.connection_params["Acquisition"])
                                
                                t0 = Timer()
                                while t0.sec() < self.duration:
                                        
                                        ## acquiring the data
                                        data, acq_delay = receive_data(receiver, self.pick_idxs)
                                        acq_delays.append(acq_delay)

                                        ## skip incomplete windows at the beginning of the recording
                                        if data.shape[0] != self.window_size:
                                                continue
                                        
                                        ## FFT
                                        start_method_time = time.time()
                                        data *= fft_window[:,np.newaxis]
                                        fftval = np.abs(np.fft.rfft(data, axis=1) / data.shape[-1])
                                        power = np.square(fftval[:, freq_band_idxs]).mean()
                                        method_delays.append(time.time() - start_method_time)

                                        return power

                        case "individual_peak_power":
                                
                                ## extracting the periodic components from the baseline recording
                                _, peak_params_ = estimate_aperiodic_component(raw_baseline=self.raw_baseline,
                                                        picks=self.picks, psd_params=self.modality_params["psd"],
                                                        fitting_params=self.modality_params["psd_fitting"],
                                                        verbose=self.verbose)
                                peak_params = [peak_param[0] for peak_param in peak_params_ if self.freq_range[0] < peak_param[0] < self.freq_range[1]]
                                
                                if len(peak_params) == 1:
                                        cf = peak_params[0]
                                else:
                                        cf = (self.freq_range[0] + self.freq_range[1]) / 2
                                        warn(f"center frequency was set to the middle frequency in the selected frequency range.")
                                
                                ## compute power in a small range around individual peak
                                fft_window, freq_band, freq_band_idxs, frequencies = compute_fft(sfreq=self.rec_info["sfreq"],
                                                winsize=self.connection_params["Acquisition"]["winsize"],
                                                freq_range=self.freq_range,
                                                freq_res=self.modality_params["fft"]["freq_res"])
                                
                                freq_var = self.modality_params["fft"]["freq_var"]
                                individual_freq_band_idxs = np.where(np.logical_and(cf - freq_var <= frequencies,
                                                                                frequencies <= cf + freq_var))[0]

                                ## start the real recording 
                                receiver = StreamReceiver(stream_name=self.stream.stream_name,
                                                           **self.connection_params["Acquisition"])
                        
                                t0 = Timer()
                                while t0.sec() < self.duration:
                                        ## acquiring the data
                                        data, acq_delay = receive_data(receiver, self.pick_idxs)
                                        acq_delays.append(acq_delay)

                                        ## skip incomplete windows at the beginning of the recording
                                        if data.shape[0] != self.window_size:
                                                continue
                                        
                                        ## FFT
                                        start_method_time = time.time()
                                        data *= fft_window[:,np.newaxis]
                                        fftval = np.abs(np.fft.rfft(data, axis=1) / data.shape[-1])
                                        power = np.square(fftval[:, individual_freq_band_idxs]).mean()
                                        method_delays.append(time.time() - start_method_time)
                                        
                                        return power


                        case "entropy":
                                ## start the real recording 
                                receiver = StreamReceiver(stream_name=self.stream.stream_name,
                                                           **self.connection_params["Acquisition"])

                                t0 = Timer()
                                if self.modality_params["entropy_method"] == "AppEn":

                                        while t0.sec() < self.duration:
                                                ## acquiring the data
                                                data, acq_delay = receive_data(receiver, self.pick_idxs)
                                                acq_delays.append(acq_delay)
                                                
                                                ## skip incomplete windows at the beginning of the recording
                                                if data.shape[0] != self.window_size:
                                                        continue
                                                
                                                ## compute spectral entropy
                                                start_method_time = time.time()
                                                ent = compute_app_entropy(data.T, self.modality_params["emb_app"],
                                                                                self.modality_params["metric_app"]).mean()
                                                method_delays.append(time.time() - start_method_time)

                                                return ent

                                if self.modality_params["entropy_method"] == "SampEn":

                                        while t0.sec() < self.duration:
                                                ## acquiring the data
                                                data, acq_delay = receive_data(receiver, self.pick_idxs)
                                                acq_delays.append(acq_delay)
                                                
                                                ## skip incomplete windows at the beginning of the recording
                                                if data.shape[0] != self.window_size:
                                                        continue
                                                
                                                ## compute spectral entropy
                                                start_method_time = time.time()
                                                ent = compute_samp_entropy(data.T, self.modality_params["emb_sample"],
                                                                                self.modality_params["metric_sample"]).mean()
                                                method_delays.append(time.time() - start_method_time)

                                                return ent

                                if self.modality_params["entropy_method"] == "Spectral":

                                        while t0.sec() < self.duration:
                                                ## acquiring the data
                                                data, acq_delay = receive_data(receiver, self.pick_idxs)
                                                acq_delays.append(acq_delay)
                                                
                                                ## skip incomplete windows at the beginning of the recording
                                                if data.shape[0] != self.window_size:
                                                        continue
                                                
                                                ## compute spectral entropy
                                                start_method_time = time.time()
                                                ent = compute_spect_entropy(sfreq=self.rec_info["sfreq"], data=data.T,
                                                                                psd_method=self.modality_params["psd_method"]).mean()
                                                method_delays.append(time.time() - start_method_time)

                                                return ent

                                if self.modality_params["entropy_method"] == "SVD":

                                        while t0.sec() < self.duration:
                                                ## acquiring the data
                                                data, acq_delay = receive_data(receiver, self.pick_idxs)
                                                acq_delays.append(acq_delay)
                                                
                                                ## skip incomplete windows at the beginning of the recording
                                                if data.shape[0] != self.window_size:
                                                        continue
                                                
                                                ## compute spectral entropy
                                                start_method_time = time.time()
                                                ent = compute_svd_entropy(data.T, self.modality_params["tau"],
                                                                        self.modality_params["emb_svd"]).mean()
                                                method_delays.append(time.time() - start_method_time)

                                                return ent


                        case "source_power":

                                fft_window, freq_band, freq_band_idxs, _ = compute_fft(sfreq=self.rec_info["sfreq"],
                                                winsize=self.modality_params["fft"]["winsize"],
                                                freq_range=self.freq_range,
                                                freq_res=self.modality_params["fft"]["freq_res"])
                                
                                bls = mne.read_labels_from_annot(subject=self.modality_params["src"]["subject"],
                                                                parc=self.modality_params["src"]["atlas"],
                                                                verbose=self.verbose)
                                bl_names = [bl.name for bl in bls]
                                bl_idx = bl_names.index(self.modality_params["src"]["bl"])
                                lambda2 = 1.0 / self.modality_params["src"]["snr"] ** 2
                                self.inv = create_inverse_operator(self.raw_baseline, self.modality_params["src"],
                                                                        verbose=self.verbose)
                                
                                ## start the real recording 
                                receiver = StreamReceiver(stream_name=self.stream.stream_name,
                                                           **self.connection_params["Acquisition"])
                                t0 = Timer()
                                while t0.sec() < self.duration:
                                        
                                        ## acquiring the data
                                        raw_data, acq_delay = receive_data(receiver, self.picks, return_raw=True)
                                        acq_delays.append(time.time() - acq_delay)
                                        
                                        ## skip incomplete windows at the beginning of the recording
                                        if len(raw_data) != self.window_size:
                                                continue
                                        
                                        ## compute source activation and then power
                                        start_method_time = time.time()
                                        stc_data = apply_inverse_raw(raw_data, self.inv, lambda2=lambda2,
                                                                label=bls[bl_idx], **self.modality_params["inv_modeling"],
                                                                verbose=self.verbose).data.T
                                        
                                        stc_data *= fft_window[:,np.newaxis]
                                        fftval = np.abs(np.fft.rfft(stc_data, axis=1) / stc_data.shape[-1])
                                        stc_power = np.square(fftval[:, freq_band_idxs]).mean()
                                        method_delays.append(time.time() - start_method_time)

                                        return stc_power
                
                        case "sensor_connectivity":
                                
                                ch_names = self.rec_info["ch_names"]
                                chs = self.modality_params["channels"]["ch_names"]
                                indices = [(np.array([ch_names.index(ch1), ch_names.index(ch2)])) for ch1, ch2 in zip(chs[0], chs[1])]
                                indices = tuple(indices)
                                freqs = np.linspace(self.freq_range[0], self.freq_range[1],
                                                        self.modality_params["channels"]["freq_res"])

                                ## start the real recording 
                                receiver = StreamReceiver(stream_name=self.stream.stream_name,
                                                           **self.connection_params["Acquisition"])
                                t0 = Timer()
                                while t0.sec() < self.duration:
                                        ## acquiring the data
                                        data, acq_delay = receive_data(receiver, self.pick_idxs)
                                        acq_delays.append(acq_delay)

                                        ## skip incomplete windows at the beginning of the recording
                                        if data.shape[0] != self.window_size:
                                                continue
                                        
                                        ## compute connectivity between sensors
                                        self.data = data
                                        start_method_time = time.time()
                                        con = spectral_connectivity_time(data=data.T[np.newaxis,:], freqs=freqs,
                                                                        indices=indices, average=False,
                                                                        sfreq=self.rec_info["sfreq"], 
                                                                        fmin=self.freq_range[0],
                                                                        fmax=self.freq_range[1], 
                                                                        faverage=True,
                                                                        **self.modality_params["con"],
                                                                        verbose=self.verbose)
                                        con_data = np.squeeze(con.get_data(output='dense'))[indices].mean()
                                        method_delays.append(time.time() - start_method_time)

                                        return con_data


                        case "source_connectivity":

                                ## initiating the source space
                                bls = mne.read_labels_from_annot(subject=self.modality_params["src"]["subject"],
                                                                parc=self.modality_params["src"]["atlas"],
                                                                verbose=self.verbose)
                                bl_names = [bl.name for bl in bls]
                                (bl_idx1, bl_idx2) = (bl_names.index(self.modality_params["src"]["bl_1"]),
                                                        bl_names.index(self.modality_params["src"]["bl_2"]))
                                lambda2 = 1.0 / self.modality_params["src"]["snr"] ** 2
                                self.inv = create_inverse_operator(self.raw_baseline, self.modality_params["src"],
                                                                        verbose=self.verbose)
                                freqs = np.linspace(self.freq_range[0], self.freq_range[1],
                                                        self.modality_params["fft"]["freq_res"])
                                
                                ## start the real recording 
                                receiver = StreamReceiver(stream_name=self.stream.stream_name,
                                                           **self.connection_params["Acquisition"])
                                t0 = Timer()
                                while t0.sec() < self.duration:
                                        ## acquiring the data
                                        raw_data, acq_delay = receive_data(receiver, self.picks, return_raw=True)
                                        acq_delays.append(time.time() - acq_delay)

                                        ## skip incomplete windows at the beginning of the recording
                                        if len(raw_data) != self.window_size:
                                                continue

                                        ## compute the activation in labels (avoid for loop for speed)
                                        start_method_time = time.time()
                                        stc1 = apply_inverse_raw(raw_data, self.inv, lambda2=lambda2,
                                                                label=bls[bl_idx1], **self.modality_params["inv_modeling"],
                                                                verbose=self.verbose)
                                        stc2 = apply_inverse_raw(raw_data, self.inv, lambda2=lambda2,
                                                                label=bls[bl_idx2], **self.modality_params["inv_modeling"],
                                                                verbose=self.verbose)
                                        stc1_data = np.mean(stc1.data, axis=0)
                                        stc2_data = np.mean(stc2.data, axis=0)
                                        con = spectral_connectivity_time(data=np.array([[stc1_data, stc2_data]]),
                                                                        freqs=freqs,
                                                                        indices=None,
                                                                        average=False,
                                                                        sfreq=self.rec_info["sfreq"],
                                                                        fmin=self.freq_range[0],
                                                                        fmax=self.freq_range[1],
                                                                        faverage=True,
                                                                        **self.modality_params["con"],
                                                                        verbose=self.verbose)
                                        con_data = np.squeeze(con.get_data(output='dense'))[1][0]
                                        method_delays.append(time.time() - start_method_time)
                                        
                                        return con_data

                        case "sensor_graph":

                                ch_names = self.rec_info["ch_names"]
                                chs = self.modality_params["channels"]["ch_names"]
                                indices = [(np.array([ch_names.index(ch1), ch_names.index(ch2)])) for ch1, ch2 in zip(chs[0], chs[1])]
                                indices = tuple(indices)
                                ## initiating the filter
                                self.sos = butter_bandpass(self.freq_range[0], self.freq_range[1],
                                                self.rec_info["sfreq"], order=self.modality_params["channels"]["order"])
                                
                                ## start the real recording 
                                receiver = StreamReceiver(stream_name=self.stream.stream_name,
                                                           **self.connection_params["Acquisition"])
                                t0 = Timer()
                                while t0.sec() < self.duration:
                                        ## acquiring the data
                                        data, acq_delay = receive_data(receiver, self.pick_idxs)
                                        acq_delays.append(acq_delay)

                                        ## skip incomplete windows at the beginning of the recording
                                        if data.shape[0] != self.window_size:
                                                continue

                                        ## filter the data and learn the graph
                                        start_method_time = time.time()
                                        data_filt = bandpass_filter(data.T, self.freq_range[0], self.freq_range[1],
                                                                self.rec_info["sfreq"], self.modality_params["channels"]["order"])
                                        graph_matrix = log_degree_barrier(data_filt, **self.modality_params["graph"])
                                        avg_edge = np.array([graph_matrix[idxs] for idxs in indices]).mean()
                                        method_delays.append(time.time() - start_method_time)
                                        
                                        return avg_edge

                        
                        case "source_graph":

                                ## initiating the source space
                                bls = mne.read_labels_from_annot(subject=self.modality_params["src"]["subject"],
                                                                parc=self.modality_params["src"]["atlas"],
                                                                verbose=self.verbose)
                                bl_names = [bl.name for bl in bls]
                                (bl_idx1, bl_idx2) = (bl_names.index(self.modality_params["src"]["bl_1"]),
                                                        bl_names.index(self.modality_params["src"]["bl_2"])) 
                                
                                lambda2 = 1.0 / self.modality_params["src"]["snr"] ** 2
                                self.inv, self.src = create_inverse_operator(self.raw_baseline, self.modality_params["src"],
                                                                        return_src=True, verbose=self.verbose)
                                ## initiating the filter
                                self.sos = butter_bandpass(self.freq_range[0], self.freq_range[1],
                                                self.rec_info["sfreq"], order=self.modality_params["src"]["order"])
                                ## start the real recording 
                                receiver = StreamReceiver(stream_name=self.stream.stream_name,
                                                           **self.connection_params["Acquisition"])
                                
                                t0 = Timer()
                                while t0.sec() < self.duration:
                                        ## acquiring the data
                                        raw_data, acq_delay = receive_data(receiver, self.picks, return_raw=True)
                                        acq_delays.append(time.time() - acq_delay)

                                        ## skip incomplete windows at the beginning of the recording
                                        if len(raw_data) != self.window_size:
                                                continue

                                        ## compute the activation in labels -> filter it -> learn the graph
                                        start_method_time = time.time()
                                        stcs = apply_inverse_raw(raw_data, self.inv, lambda2=lambda2, 
                                                                **self.modality_params["inv_modeling"],
                                                                verbose=self.verbose)
                                        tcs = stcs.extract_label_time_course(bls, src=self.src,
                                                                        **self.modality_params["label_extraction"],
                                                                        verbose=self.verbose)
                                        tcs_filt = bandpass_filter(tcs, self.freq_range[0], self.freq_range[1],
                                                                self.rec_info["sfreq"], self.modality_params["src"]["order"])
                                        graph_matrix = log_degree_barrier(tcs_filt, **self.modality_params["graph"])
                                        avg_edge = graph_matrix[bl_idx1][bl_idx2]
                                        method_delays.append(time.time() - start_method_time)
                                        
                                        return avg_edge
                                        

                return acq_delays, method_delays

        
        def plot(self, bufsize_view=0.2):
                StreamViewer(stream_name=self.stream.stream_name).start(bufsize=bufsize_view)
                self.bufsize_view = bufsize_view

        def plot_sensors(self, kind="topomap"):
                if self.picks is not None:
                        self.rec_info["bads"].extend(self.picks)
                mne.viz.plot_sensors(info=self.rec_info, kind=kind, verbose=self.verbose)
                self.rec_info["bads"] = []

        def set_meas_date(self):
                self.rec_info.set_meas_date(datetime.datetime.now().replace(tzinfo=datetime.timezone.utc))
        
#        def create_report(self):

                        # plot alignment in source space
                #         mne.viz.plot_alignment(
                #     nf.raw_baseline.info,
                #     src=src,
                #     eeg=["original", "projected"],
                #     trans="fsaverage",
                #     show_axes=True,
                #     mri_fiducials=True,
                #     dig="fiducials",
                # )

        # def disconnect():

        # def save_raw_data():

        # def save_nf_data():

        # def plot_delays():
