import os.path as op
from pathlib import Path

import numpy as np
from mne.io import read_raw_brainvision
from mne.label import select_sources
from mne.datasets import fetch_fsaverage
from mne import (
                rename_channels,
                make_forward_solution,
                read_labels_from_annot,
                make_ad_hoc_cov)
from mne.simulation import (
                            SourceSimulator,
                            simulate_raw,
                            add_noise
                            )

def simulate_eeg_raw(brain_label, frequency, amplitude, duration,
                    gap_duration, n_repetition, start, verbose=None):
    """
    Generate simulated source estimates and raw data by adding an artificial signal in a desired brain label.
    
    Parameters
    ----------
    brain_label : str
            The new measurement date. If datetime object, it must be timezone-aware and in UTC.
            A tuple of (seconds, microseconds) or float (alias for (meas_date, 0)) can also be passed and a datetime object will be automatically created.
            If None, the time of executing code will be added.
    frequency : float
        frequency of simulated sine signal.
    amplitude : float
        amplitude of simulated sine signal.
    duration : float
        duration of the simulated signal.
    gap_duration: float
        interval between two consequent simulated signals
    n_repetition : int
        number of repetition of the signal.
    start : str
        start time of the first simulated signal.
    verbose : bool | str | int | None
        Control verbosity of the logging output.

    Saves the simulated signal.
    """  

    ## load the data
    data_dir = Path.cwd().parent.parent / "template_data" 
    fname_vhdr = data_dir / "template_data.vhdr"
    raw = read_raw_brainvision(fname_vhdr, preload=True, verbose=verbose)

    ## montaging and removing ecg channels
    new_ch_names = raw.info['ch_names'].copy()
    new_ch_names[58] = 'Fpz' # change FPz to Fpz
    mapping = dict(zip(raw.info['ch_names'], new_ch_names))
    rename_channels(raw.info, mapping)
    raw.drop_channels(ch_names=["HRli", "HRre"], on_missing='raise')
    raw.set_montage('standard_1020', on_missing='warn', verbose=verbose)

    ## download fsaverage files
    fs_dir = fetch_fsaverage(verbose=True)
    subjects_dir = op.dirname(fs_dir)
    subject = "fsaverage"
    trans = "fsaverage"  # MNE has a built-in fsaverage transformation
    src = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
    bem = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")

    ## create forward solution
    fwd = make_forward_solution(raw.info, trans=trans, src=src, bem=bem)
    src = fwd["src"]

    ## put source activation
    tstep = 1.0 / raw.info["sfreq"]
    selected_label = read_labels_from_annot(subject,
                                            regexp=brain_label,
                                            subjects_dir=subjects_dir)[0]
    
    label = select_sources(subject, selected_label, location="center",
                                    extent=1, grow_outside=True, 
                                    subjects_dir=subjects_dir)
    source_time_series = np.sin(2.0 * np.pi * frequency * np.arange(int(duration * raw.info["sfreq"])) * tstep) * 10e-9 * amplitude

    gap_duration_s = gap_duration * raw.info["sfreq"]
    start_s = start * raw.info["sfreq"]
    events = np.zeros((n_repetition, 3), int)
    events[:, 0] = start_s + gap_duration_s * np.arange(n_repetition)
    events[:, 2] = 1  

    source_simulator = SourceSimulator(fwd["src"], tstep=tstep)
    source_simulator.add_data(label, source_time_series, events)
    raw = simulate_raw(raw.info, source_simulator, forward=fwd)
    cov = make_ad_hoc_cov(raw.info, verbose=verbose)
    add_noise(raw, cov, iir_filter=[0.2, -0.2, 0.04])

    ## save
    saving_dir = Path.cwd().parent.parent / "simulated_data"
    raw.save(fname=f"{saving_dir}/{brain_label}_{frequency}_{amplitude}-raw.fif")
    