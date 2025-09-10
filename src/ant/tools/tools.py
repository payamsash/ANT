import os.path as op
import time
from copy import deepcopy
from functools import wraps

import yaml
import numpy as np
from scipy import sparse
from scipy.integrate import simpson
from scipy.signal import butter, welch, periodogram, get_window
from scipy.spatial.distance import pdist, squareform
from fooof import FOOOF
from pyunlocbox import functions, solvers
from padasip.filters import FilterLMS

from mne import make_forward_solution, compute_raw_covariance
from mne.io import read_raw
from mne.preprocessing import ICA
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator
from mne.time_frequency import psd_array_multitaper
from mne.viz import create_3d_figure, get_brain_class

from mne_icalabel import label_components

def timed(func):
        """Decorator to measure execution time of a function.

        Parameters
        ----------
        func : callable
                Function to be timed.

        Returns
        -------
        wrapper : callable
                Wrapped function that returns both the function's output
                and its execution time in seconds.

        Notes
        -----
        The decorated function returns a tuple ``(value, elapsed_time)``,
        where ``value`` is the original return value of the function,
        and ``elapsed_time`` is the runtime measured with
        :func:`time.perf_counter`.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
                tic = time.perf_counter()
                value = func(*args, **kwargs)
                toc = time.perf_counter()
                return value, toc - tic
        return wrapper

def get_canonical_freqs(frange_name):
        """
        Map a canonical frequency band name to its frequency range.

        Parameters
        ----------
        frange_name : str
                Name of the frequency band. Supported values include:
                'delta', 'theta', 'alpha', 'lower_alpha', 'upper_alpha',
                'smr', 'beta', 'lower_beta', 'upper_beta',
                'gamma', 'lower_gamma', 'upper_gamma'.

        Returns
        -------
        List[float]
                Two-element list containing [low_freq, high_freq] in Hz.

        Raises
        ------
        ValueError
                If the provided frequency band name is not defined.
        """
        
        freq_bands = {
                        "delta": [0.5, 4],
                        "theta": [4, 8],
                        "alpha": [8, 13],
                        "lower_alpha": [8, 10],
                        "upper_alpha": [10, 13],
                        "smr": [12, 15],
                        "beta": [15, 30],
                        "lower_beta": [15, 20],
                        "upper_beta": [20, 30],
                        "gamma": [30, 80],
                        "lower_gamma": [30, 50],
                        "upper_gamma" : [50, 80],
                        }
        try:
                return freq_bands[frange_name]
        except KeyError:
                raise ValueError(
                                f"Frequency range '{frange_name}' is not defined. "
                                f"Available options: {list(freq_bands.keys())}"
                                )

def get_params(config_file, modality, modality_params):
        """Load and update parameters for a given neurofeedback (NF) modality.

        Parameters
        ----------
        config_file : str | path-like
                Path to the YAML configuration file containing modality parameters.
        modality : str
                Name of the neurofeedback modality. Must be present in the
                ``NF_modality`` section of the config file.
        modality_params : dict | None
                Optional dictionary with parameter overrides. The keys should be method
                names (e.g., "fft", "welch"), and the values should be dictionaries
                with parameter key-value pairs to update.

        Returns
        -------
        params : dict
                Dictionary containing updated parameters for the specified modality.

        Raises
        ------
        ValueError
                If the modality is not found in the config file.
        ValueError
                If a provided method in ``modality_params`` is not available
                for the given modality.

        Notes
        -----
        This function first loads default parameters for the requested modality
        from the YAML file. If ``modality_params`` is provided, the defaults
        are updated with the user-specified overrides.
        """
        with open(config_file, "r") as f:
                config = yaml.safe_load(f)

        if modality not in config["NF_modality"]:
                raise ValueError(f"Unknown modality {modality!r}, must be one of {list(config['NF_modality'].keys())}")

        # get params for this modality
        params = deepcopy(config["NF_modality"][modality])
        if modality_params is not None:
                for method, overrides in modality_params.items():
                        if method not in params:
                                raise ValueError(f"Unknown method {method!r} for modality {modality!r}. Available: {list(params.keys())}")
                        params[method].update(overrides)   
        return params

def compute_bandpower(
        data,
        sfreq,
        band,
        method="fft",
        relative=True,
        window="hann",
        n_fft=None,
        **kwargs
        ):
        """
        Compute the bandpower of EEG/MEG channels using various PSD estimation methods.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_samples)
                Input time series data.
        sfreq : float
                Sampling frequency in Hz.
        band : tuple of float
                Frequency band of interest (low, high) in Hz.
        method : {'fft', 'periodogram', 'welch', 'multitaper'}, default='fft'
                PSD estimation method.
        relative : bool, default=True
                Normalize by total power if True.
        window : str, default='hann'
                Window type (used only for 'fft').
        n_fft : int | None
                Number of FFT points (used only for 'fft').
        **kwargs : additional keyword arguments
                Passed to the respective PSD functions.

        Returns
        -------
        bandpower : ndarray, shape (n_channels,)
                Bandpower of each channel in the requested frequency band.
        """
        assert data.ndim == 2, "Input must be 2D: (n_channels, n_samples)"
        assert len(band) == 2 and band[0] <= band[1], "Band must be (low, high)"

        n_channels, n_samples = data.shape
        n_fft = n_fft or int(2 ** np.ceil(np.log2(n_samples)))

        if method == "fft":
                win = get_window(window, n_samples, fftbins=True)
                data_win = data * win
                freqs = np.fft.rfftfreq(n_fft, d=1/sfreq)
                psd = (np.abs(np.fft.rfft(data_win, n=n_fft)) ** 2) / (sfreq * np.sum(win**2))

        elif method == "periodogram":
                freqs, psd = periodogram(data, sfreq, axis=1, **kwargs)

        elif method == "welch":
                freqs, psd = welch(data, sfreq, axis=1, **kwargs)

        elif method == "multitaper":
                psd, freqs = psd_array_multitaper(data, sfreq, axis=1, verbose="ERROR", **kwargs)

        else:
                raise ValueError(f"Unsupported method '{method}'.")

        # Frequency band selection
        mask = (freqs >= band[0]) & (freqs <= band[1])
        freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        bp = simpson(psd[:, mask], dx=freq_res, axis=1)
        if relative:
                bp /= simpson(psd, dx=freq_res, axis=1)

        return bp

def compute_fft(sfreq, winsize, freq_range, freq_res=1):
        """Compute FFT-related quantities for spectral analysis.

        Parameters
        ----------
        sfreq : float
                Sampling frequency in Hz.
        winsize : float
                Window size in seconds.
        freq_range : tuple of float
                Frequency range of interest (low, high) in Hz.
        freq_res : int, optional
                Frequency resolution factor. Default is 1.

        Returns
        -------
        fft_window : ndarray, shape (n_times,)
                Hanning window of length ``winsize * sfreq`` in samples.
        freq_band : ndarray, shape (n_freqs,)
                Array of frequencies within the specified range.
        freq_band_idxs : ndarray, shape (n_freqs,)
                Indices of ``frequencies`` corresponding to ``freq_band``.
        frequencies : ndarray, shape (n_freqs_total,)
                Array of frequencies from the FFT across the full spectrum.
        """
        winsize_in_samples = sfreq * winsize
        frequencies = np.fft.rfftfreq(n=int(winsize_in_samples)*freq_res, d=1/sfreq) 
        freq_band_idxs = np.where(np.logical_and(freq_range[0]<=frequencies, frequencies<=freq_range[1]))[0]
        freq_band = frequencies[freq_band_idxs]
        fft_window = np.hanning(winsize_in_samples)

        return fft_window, freq_band, freq_band_idxs, frequencies

def butter_bandpass(l_freq, h_freq, sfreq, order):
        """Design a Butterworth band-pass filter.

        Parameters
        ----------
        l_freq : float
                Low cut-off frequency in Hz.
        h_freq : float
                High cut-off frequency in Hz.
        sfreq : float
                Sampling frequency in Hz.
        order : int
                Order of the Butterworth filter.

        Returns
        -------
        sos : ndarray
                Second-order sections representation of the band-pass filter,
                suitable for use with :func:`scipy.signal.sosfiltfilt`.
        """
        nyq = 0.5 * sfreq
        (low, high) = (l_freq / nyq, h_freq / nyq) 
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def estimate_aperiodic_component(
                raw_baseline,
                picks,
                method,
                freq_range=(1, 20),
                verbose=None
                ):
        """
        Estimate the aperiodic (1/f) component of a PSD using FOOOF.

        Parameters
        ----------
        raw_baseline : mne.io.BaseRaw
                MNE Raw object containing baseline data.
        picks : str | list
                Channels to include in PSD computation.
        psd_params : dict
                Parameters passed to `raw_baseline.compute_psd()`.

        freq_range : tuple of float, optional (default=(1, 20))
                Frequency range in Hz for fitting the model.
        verbose : bool, optional
                Whether to print FOOOF fitting progress.

        Returns
        -------
        aperiodic_params : ndarray
                Parameters of the aperiodic component (offset, slope, knee if used).
        peak_params : ndarray
                Parameters of oscillatory peaks identified in the spectrum.
        """
        spectrum = raw_baseline.compute_psd(picks=picks, method=method, fmax=80)
        fm = FOOOF(verbose=verbose)

        fm.fit(
                spectrum.freqs,
                spectrum.get_data().mean(axis=0),
                freq_range=freq_range
        )
        return fm.aperiodic_params_, fm.peak_params_


def _compute_inv_operator(
                raw_baseline,
                subject_fs_id="fsaverage",
                subjects_fs_dir=None
                ):
        """
        Compute the inverse operator for EEG source localization.

        This function sets up the forward and inverse models required
        to project sensor-level EEG data into source space. Depending
        on whether the subject is ``fsaverage`` or an individual, the
        function handles source space, BEM, and coregistration steps
        differently.

        Parameters
        ----------
        raw_baseline : mne.io.Raw
                The baseline raw EEG recording used to estimate the noise
                covariance and forward model.
        subject : str, default='fsaverage'
                The subject identifier. If 'fsaverage', a template anatomy
                provided by MNE is used. Otherwise, a subject-specific model
                is built using individual MRI and digitization data.

        Returns
        -------
        inverse_operator : dict
                The inverse operator object created with MNE-Python. This can
                be passed to functions such as ``apply_inverse`` or
                ``apply_inverse_epochs`` to estimate source time courses.

        Notes
        -----
        - If ``subject='fsaverage'``:
                * Fetches the template FreeSurfer subject using
                :func:`mne.datasets.fetch_fsaverage`.
                * Uses precomputed fsaverage source space and BEM solution.
                * Uses the default ``'fsaverage'`` trans file.

        - If an individual subject is specified:
                * Requires subject-specific source space and BEM setup.
                * Runs an automatic coregistration using fiducials and ICP
                alignment (with multiple iterations and nasion weighting).
                * Builds a BEM model and solution with :func:`mne.make_bem_model`
                and :func:`mne.make_bem_solution`.
                * Estimates the head-MRI transform from the coregistration.
        """

        if subject_fs_id == "fsaverage":
                fs_dir = fetch_fsaverage()
                src = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
                bem = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")
                trans = "fsaverage"

        else:
                bem = sl_params["bem"]
                src = sl_params["source"]

                src = setup_source_space(subject=subject_fs_id, subjects_dir=subjects_fs_dir)
                bem_model = make_bem_model(subject=subject_fs_id, subjects_dir=subjects_fs_dir)  
                bem = make_bem_solution(bem_model)
                coreg = Coregistration(raw_baseline.info, subject=subject_fs_id, subjects_dir=subjects_fs_dir, fiducials='auto')
                coreg.fit_fiducials()
                coreg.fit_icp(n_iterations=40, nasion_weight=2.0) 
                coreg.omit_head_shape_points(distance=5.0 / 1000)
                coreg.fit_icp(n_iterations=40, nasion_weight=10)
                trans = coreg.trans

        fwd = make_forward_solution(
                                raw_baseline.info,
                                trans=trans,
                                src=src,
                                bem=bem,
                                meg=False,
                                eeg=True
                                )
        noise_cov = compute_raw_covariance(raw_baseline, method="empirical")
        inverse_operator = make_inverse_operator(raw_baseline.info, fwd, noise_cov)

        return inverse_operator

def weight_to_degree_map(n_nodes):
        """
        Construct linear mappings between edge weights and node degrees.

        Given a graph with `n_nodes`, this function builds a sparse matrix
        that maps edge weights to node degrees and its transpose. The
        resulting linear operators are useful in graph optimization problems.

        Parameters
        ----------
        n_nodes : int
                Number of nodes in the graph.

        Returns
        -------
        k : callable
                Function mapping edge weights (1D array of length n_edges) to
                node degrees (1D array of length n_nodes).
        kt : callable
                Function mapping node degrees back to edge weights.

        Notes
        -----
        - The number of edges is ``n_edges = n_nodes * (n_nodes - 1) / 2``.
        - Internally, constructs a sparse COO matrix of shape
        ``(n_nodes, n_edges)``.
        """

        n_edges = n_nodes * (n_nodes - 1) // 2

        row_idx1 = np.repeat(np.arange(n_nodes - 1), np.arange(n_nodes - 1, 0, -1))
        row_idx2 = np.concatenate([np.arange(i + 1, n_nodes) for i in range(n_nodes - 1)])

        row_idx = np.concatenate((row_idx1, row_idx2))
        col_idx = np.concatenate((np.arange(n_edges), np.arange(n_edges)))
        vals = np.ones(len(row_idx))

        coo = sparse.coo_matrix((vals, (row_idx, col_idx)), shape=(n_nodes, n_edges))

        return lambda w: coo.dot(w), lambda d: coo.T.dot(d)


def log_degree_barrier(
                signals,
                dist_type,
                alpha,
                beta,
                step=0.5,
                max_iter=10000,
                rtol=1.0e-16,
                w0=None
                ):
        """
        Graph learning with a log-barrier degree constraint.

        Builds a weighted graph from smooth signals by solving a convex
        optimization problem with:
        - distance fitting,
        - log-barrier enforcing non-degenerate node degrees,
        - quadratic regularization on edge weights.

        Parameters
        ----------
        signals : array of shape (n_nodes, n_samples)
                Input data (node-wise signals).
        dist_type : str
                Distance metric to use (passed to ``scipy.spatial.distance.pdist``).
        alpha : float
                Weight of the log-barrier degree penalty.
        beta : float
                Weight of the L2 regularization term.
        step : float
                Initial step size for the optimization.
        w0 : array or None
                Initial edge weights. If None, initialized to zeros.
        max_iter : int
                Maximum number of iterations for the solver.
        rtol : float
                Relative tolerance for convergence.

        Returns
        -------
        graph_matrix : ndarray of shape (n_nodes, n_nodes)
                Learned adjacency matrix of the graph.

        Notes
        -----
        - Uses MLF-BF primal-dual solver (external `solvers` package).
        - Relies on `functions.func` objects with custom eval/prox/grad.
        - The optimization problem solved is of the form::

                min_w  2 <w, z> - α Σ log(k(w)) + β ||w||²

        where z are normalized pairwise distances and k(w) maps edge
        weights to node degrees.

        """
        n_nodes = signals.shape[0]

        # Compute pairwise distances and normalize
        z = pdist(signals, dist_type)
        z /= np.max(z)
        w0 = np.zeros_like(z) if w0 is None else w0

        # Degree operator
        k, kt = weight_to_degree_map(n_nodes)
        norm_k = np.sqrt(2 * (n_nodes - 1))

        # Objective terms
        f1 = functions.func()
        f1._eval = lambda w: 2 * np.dot(w, z)
        f1._prox = lambda w, gamma: np.maximum(0, w - 2 * gamma * z)

        f2 = functions.func()
        f2._eval = lambda w: -alpha * np.sum(np.log(np.maximum(np.finfo(float).eps, k(w))))
        f2._prox = lambda d, gamma: np.maximum(0, 0.5 * (d + np.sqrt(d**2 + 4 * alpha * gamma)))

        f3 = functions.func()
        f3._eval = lambda w: beta * np.sum(w**2)
        f3._grad = lambda w: 2 * beta * w
        lipg = 2 * beta

        # Solver setup
        stepsize = step / (1 + lipg + norm_k)
        solver = solvers.mlfbf(L=k, Lt=kt, step=stepsize)
        problem = solvers.solve([f1, f2, f3], x0=w0, solver=solver, maxit=max_iter,
                                rtol=rtol, verbosity="NONE")

        return squareform(problem["sol"])


def plot_glass_brain(bl1, bl2=None):
        """Plot one or two brain labels on an fsaverage glass brain.

        Parameters
        ----------
        bl1 : str
                Name of the first brain label (must exist in the 'aparc' parcellation).
        bl2 : str | None, optional
                Name of the second brain label. If None (default), only ``bl1`` is
                displayed.

        Returns
        -------
        fig_brain : matplotlib.figure.Figure
                Figure containing the glass brain plots with highlighted labels.

        Notes
        -----
        This function uses the ``fsaverage`` subject and the ``aparc`` parcellation
        from FreeSurfer. The labels are displayed on a cortical surface rendering
        using four different views (frontal, dorsal, frontal, lateral).
        """
        brain_kwargs = dict(alpha=0.15, background="white", cortex="low_contrast", size=(800, 600))
        brain_labels = read_labels_from_annot(subject='fsaverage', parc='aparc')
        bl_names = [bl.name for bl in brain_labels]
        views = ["frontal", "dorsal", "frontal", "frontal"]
        azimuths = [180, 0, 0, -90]
        fig_brain, axs = plt.subplots(1, 4, figsize=(12, 3))

        for view, azimuth, ax in zip(views, azimuths, axs):

                figure = create_3d_figure(size=(100, 100), bgcolor=(0, 0, 0))
                Brain = get_brain_class()
                brain = Brain("fsaverage", hemi="both", surf="pial", **brain_kwargs)

                if bl2 is None:
                        idx = bl_names.index(bl1)
                        brain.add_label(brain_labels[idx], hemi="both", color='#d62728', borders=False, alpha=0.8)
                
                if bl2 is not None:

                        for bl, color in zip([bl1, bl2], ['#1f77b4', '#d62728']):
                                idx = bl_names.index(bl)
                                brain.add_label(brain_labels[idx], hemi="both", color=color, borders=False, alpha=0.8)

                brain.show_view(view=view, azimuth=azimuth)
                img = brain.screenshot()  
                ax.imshow(img)
                ax.axis("off")
        fig_brain.tight_layout()

        return fig_brain


def remove_blinks_lms(data, ref_ch_idx=0, n_taps=5, mu=0.01):
        """
        Remove blink artifacts from EEG using adaptive regression (LMS).
        
        Parameters
        ----------
        data : np.ndarray
                Shape (n_channels, n_times), EEG data.
        ref_ch : int
                Index of reference channel (Fp1/Fp2/EOG).
        n_taps : int
                Number of filter taps (adaptive filter length).
        mu : float
                Learning rate (adaptation speed).
        
        Returns
        -------
        cleaned : np.ndarray
                Shape (n_channels, n_times), cleaned EEG.
        """
        n_channels, n_times = data.shape
        cleaned = data.copy()
        ref = data[ref_ch_idx, :]
        X = _make_tapped_delay(ref, n_taps=n_taps)
        
        # LMS adaptive filter per channel
        filters = [FilterLMS(n=n_taps, mu=mu) for _ in range(n_channels)]

        for ch in range(n_channels):
                if ch == ref_ch_idx:
                        continue
                
                target = data[ch, :]
                y, e, w = filters[ch].run(target, X)
                cleaned[ch, :] = e  # target - estimated blink contribution
        
        return cleaned

def _make_tapped_delay(ref, n_taps):
        """Build a tapped-delay matrix from a reference signal.

        Parameters
        ----------
        ref : ndarray, shape (n_times,)
                The reference signal.
        n_taps : int
                Number of delay taps to include.

        Returns
        -------
        X : ndarray, shape (n_times, n_taps)
                The tapped-delay matrix where each column contains the reference
                signal delayed by a given number of samples.
        """
        n_times = len(ref)
        X = np.zeros((n_times, n_taps))
        for k in range(n_taps):
                X[k:, k] = ref[:n_times-k]
        
        return X

def create_blink_template(raw, max_iter=800, method="infomax"):
        """
        Create a template for eye blink ICA component from raw EEG data.

        Parameters
        ----------
        raw : mne.io.Raw
                The raw EEG data.
        max_iter : int, optional
                Maximum number of iterations for the ICA fitting. Default is 800.
        method : str, optional
                ICA method to use. Default is "infomax".

        Returns
        -------
        template_blink_comp : np.ndarray
                The spatial topography of the ICA component corresponding to the blink
                (shape: n_channels x 1). If no blink component is detected, returns None.
        """
        ica = ICA(n_components=0.95, max_iter=max_iter, method=method, fit_params=dict(extended=True), random_state=0)
        
        try:
                ica.fit(raw)
        except Exception:
                ica = ICA(n_components=5, max_iter=max_iter, method=method, fit_params=dict(extended=True), random_state=0)
                ica.fit(raw)

        ic_dict = label_components(raw, ica, method="iclabel")
        ic_labels = ic_dict["labels"]
        ic_probs = ic_dict["y_pred_proba"]

        # Find indices of components labeled as "eye blink"
        eye_indices = [i for i, lbl in enumerate(ic_labels) if lbl == "eye blink"]

        template_blink_comp = None
        if eye_indices:
                blink_idx = max(eye_indices, key=lambda i: ic_probs[i])
                template_blink_comp = ica.get_components()[:, blink_idx]

        return template_blink_comp