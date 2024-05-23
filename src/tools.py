import os.path as op
from copy import deepcopy
from warnings import warn
import time

import numpy as np
from scipy.signal import butter
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

from fooof import FOOOF
from pyunlocbox import functions, solvers
from mne.minimum_norm import make_inverse_operator
from mne.datasets import fetch_fsaverage
from mne.viz import create_3d_figure, get_brain_class
from mne import (make_forward_solution, compute_raw_covariance,
                read_source_spaces, read_labels_from_annot)
                                

def canonical_frange_str_to_list(frange_name):
        
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
        
        if frange_name in list(freq_bands.keys()):
                return freq_bands[frange_name]
        else:
                raise ValueError(f"frequency range {frange_name} is not defined.")


def update_params(new_params, old_params, module):
        
        if new_params is None:
                new_params = old_params[module]
        else:
                old_params_copy = deepcopy(old_params)
                assert isinstance(new_params, dict), "Input params should be a dictionary."
                assert module in ["LSL", "NF_modality"], "module should be either LSL or NF_modality."

                if module == "LSL":
                        for mod in list(new_params.keys()):
                                old_params_copy[module][mod].update(new_params[mod]) 
                if module == "NF_modality":
                        for mod1 in list(new_params.keys()):
                                for mod2 in list(new_params[mod1].keys()):
                                        old_params_copy[module][mod1][mod2].update(new_params[mod1][mod2]) 

                new_params = old_params_copy[module]

        return new_params


def timed(function):
        def wrapper(*args, **kwargs):
                tic = time.time()
                value = function(*args, **kwargs)
                toc = time.time()
                return value, (toc - tic)
        return wrapper


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
                src_obj = read_source_spaces(src, verbose=verbose)
                return inverse_operator, src_obj
        else:
                return inverse_operator


def butter_bandpass(l_freq, h_freq, sfreq, order):

        nyq = 0.5 * sfreq
        (low, high) = (l_freq / nyq, h_freq / nyq) 
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos


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
                                rtol=rtol, verbosity="NONE")
        graph_matrix = squareform(problem['sol'])

        return graph_matrix


def plot_glass_brain(bl1, bl2=None):

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