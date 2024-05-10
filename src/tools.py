import os.path as op
import numpy as np
from warnings import warn
from scipy.signal import butter, sosfiltfilt
from fooof import FOOOF
import time
from scipy import sparse
from mne.io import RawArray
from scipy.spatial.distance import pdist, squareform
from pyunlocbox import functions, solvers
from mne import create_info, make_forward_solution, compute_raw_covariance
from mne.minimum_norm import make_inverse_operator
from mne.datasets import fetch_fsaverage
                                



def update_params(new_params, old_params, module):

        if new_params is None:
                new_params = old_params[module]
        else:
                for mod in list(new_params.keys()):
                        old_params[module][mod].update(new_params[mod]) 
                new_params = old_params[module]
        return new_params


def receive_data(receiver, pick_idxs, return_raw=False):

        start_acq_time = time.time()
        receiver.acquire()
        if return_raw:
                data = receiver.get_window(return_raw=True)[0].pick(pick_idxs).set_eeg_reference(projection=True)
        else:
                data = receiver.get_window(return_raw=False)[0][:,pick_idxs]
                data -= np.mean(data, axis=0, keepdims=True)
        acq_delay = time.time() - start_acq_time
        
        return data, acq_delay



def compute_fft(sfreq, winsize, freq_range, freq_res):
        
        winsize_in_samples = sfreq * winsize
        frequencies = np.fft.rfftfreq(n=int(winsize_in_samples)*freq_res, d=1/sfreq) 
        freq_band_idxs = np.where(np.logical_and(freq_range[0]<=frequencies, frequencies<=freq_range[1]))[0]
        freq_band = frequencies[freq_band_idxs]
        fft_window = np.hanning(winsize_in_samples)

        return fft_window, freq_band, freq_band_idxs, frequencies



def estimate_aperiodic_component(raw_baseline, picks, psd_params, fitting_params, verbose=None):
        
        spectrum = raw_baseline.compute_psd(picks=picks, **psd_params)
        fm = FOOOF(**fitting_params, verbose=verbose)
        fm.fit(spectrum.freqs, spectrum.get_data().mean(axis=0),
                freq_range=[1, 20]) # maybe I should fix it some where
        
        return fm.aperiodic_params_, fm.peak_params_


def create_inverse_operator(raw_baseline, sl_params, return_src=False, verbose=None):

        if sl_params["bem"] == "fsaverage" and sl_params["source"] == "fsaverage":
                fs_dir = fetch_fsaverage(verbose=verbose)
                src = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
                bem = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")
        else:
                bem = sl_params["bem"]
                src = sl_params["source"]

        
        fwd = make_forward_solution(raw_baseline.info, trans=sl_params["trans"],
                                        src=src, bem=bem,
                                        eeg=True, mindist=sl_params["mindist"],
                                        n_jobs=sl_params["n_jobs"],
                                        verbose=verbose)
        noise_cov = compute_raw_covariance(raw_baseline, method=sl_params["noise_cov_method"],
                                        n_jobs=sl_params["n_jobs"], verbose=verbose)
        inverse_operator = make_inverse_operator(raw_baseline.info, fwd, noise_cov,
                                                loose=sl_params["loose"], depth=sl_params["depth"],
                                                fixed=sl_params["fixed"], use_cps=sl_params["use_cps"],
                                                verbose=verbose)

        if return_src:
                return inverse_operator, src
        else:
                return inverse_operator


def butter_bandpass(l_freq, h_freq, sfreq, order):

        nyq = 0.5 * sfreq
        (low, high) = (l_freq / nyq, h_freq / nyq) 
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos


def bandpass_filter(data, l_freq, h_freq, sfreq, order):

        sos = butter_bandpass(l_freq, h_freq, sfreq, order=order)
        filtered_data = sosfiltfilt(sos, data)
        return filtered_data


def weight2degmap(n_nodes):


        n_edges = int(n_nodes * (n_nodes - 1) / 2)
        row_idx1 = np.zeros((n_edges, ))
        row_idx2 = np.zeros((n_edges, ))
        count = 0
        for i in np.arange(1, n_nodes):
                row_idx1[count: (count + (n_nodes - i))] = i - 1
                row_idx2[count: (count + (n_nodes - i))] = np.arange(i, n_nodes)
                count = count + n_nodes - i
        row_idx = np.concatenate((row_idx1, row_idx2))
        col_idx = np.concatenate((np.arange(0, n_edges), np.arange(0, n_edges)))
        vals = np.ones(len(row_idx))
        coo_matrix = sparse.coo_matrix((vals, (row_idx, col_idx)), shape=(n_nodes, n_edges))
        
        return lambda w: coo_matrix.dot(w), lambda d: coo_matrix.transpose().dot(d)


def log_degree_barrier(smooth_signals, dist_type, alpha, beta, step, w0, maxit, rtol):


        ## compute distances and normalize them
        n_nodes = smooth_signals.shape[0]
        z = pdist(smooth_signals, dist_type)  
        z = z / np.max(z) 
        w0 = np.zeros(z.shape) if w0 is None else w0

        ## get primal-dual linear map
        k, kt = weight2degmap(n_nodes)
        norm_k = np.sqrt(2 * (n_nodes - 1))

        ## assemble functions in the objective
        f1 = functions.func()
        f1._eval = lambda w: 2 * np.dot(w, z)
        f1._prox = lambda w, gamma: np.maximum(0, w - (2 * gamma * z))

        f2 = functions.func()
        f2._eval = lambda w: - alpha * np.sum(np.log(np.maximum(np.finfo(np.float64).eps, k(w))))
        f2._prox = lambda d, gamma: np.maximum(0, 0.5 * (d + np.sqrt(d**2 + (4 * alpha * gamma))))

        f3 = functions.func()
        f3._eval = lambda w: beta * np.sum(w**2)
        f3._grad = lambda w: 2 * beta * w
        lipg = 2 * beta

        ## rescale stepsize and solve the problem
        stepsize = step / (1 + lipg + norm_k)
        solver = solvers.mlfbf(L=k, Lt=kt, step=stepsize)
        problem = solvers.solve([f1, f2, f3], x0=w0, solver=solver, maxit=maxit,
                                rtol=rtol, verbosity=None)
        graph_matrix = squareform(problem['sol'])

        return graph_matrix

