"""
Simulating EEG and running a Neurofeedback session
==================================================

This example demonstrates how to simulate EEG data with sinusoidal sources 
in a specific brain region and use it for a neurofeedback session using the
ANT package.

We cover the following steps:

1. Simulate raw EEG data with sinusoidal activity in a cortical label.
2. Record a baseline session to extract blink templates and the inverse operator.
3. Run a main neurofeedback session using multiple modalities.
"""

###############################################################################
# Step 1: Simulate EEG recording
# ------------------------------
# First, we simulate EEG data by adding sinusoidal activity in the
# pericalcarine region of the left hemisphere. This will generate a raw 
# MNE-Python object that can be used in our neurofeedback session.

import os
from pathlib import Path
import numpy as np
from mne.io import read_raw_brainvision
from mne.label import select_sources
from mne.datasets import fetch_fsaverage
from mne import (
    set_log_level,
    rename_channels,
    make_forward_solution,
    read_labels_from_annot,
    make_ad_hoc_cov
)
from mne.simulation import (
    SourceSimulator,
    simulate_raw,
    add_noise,
    add_eog
)

def simulate_eeg_raw(
    brain_label,
    frequency,
    amplitude,
    duration,
    gap_duration,
    n_repetition,
    start,
    iir_filter=[0.2, -0.2, 0.04],
    fname_save=None,
    verbose=None
):
    """Simulate EEG data with a sinusoidal source in a given brain label.

    Parameters
    ----------
    brain_label : str
        Name (regexp) of the cortical label in which to simulate the source.
    frequency : float
        Frequency of the simulated sine wave (Hz).
    amplitude : float
        Amplitude scaling factor of the simulated signal.
    duration : float
        Duration of each simulated signal epoch in seconds.
    gap_duration : float
        Interval (in seconds) between consecutive signal epochs.
    n_repetition : int
        Number of signal epochs to simulate.
    start : float
        Start time of the first simulated signal, in seconds.
    iir_filter : array_like
        IIR filter coefficients (denominator) used when adding noise.
    verbose : bool | str | int | None
        Control verbosity of the logging output.

    Returns
    -------
    raw : instance of mne.io.Raw
        The simulated raw EEG object.
    """

    set_log_level(verbose=verbose)

    # Load example data
    data_dir = Path.cwd().parent / "data" 
    fname_vhdr = data_dir / "sample" / "sample_data.vhdr" 
    raw = read_raw_brainvision(fname_vhdr, preload=True)

    # Montage and drop ECG channels
    new_ch_names = raw.info['ch_names'].copy()
    new_ch_names[58] = 'Fpz'  # rename FPz
    mapping = dict(zip(raw.info['ch_names'], new_ch_names))
    rename_channels(raw.info, mapping)
    raw.drop_channels(ch_names=["HRli", "HRre"], on_missing='raise')
    raw.set_montage("easycap-M1", on_missing='warn')

    # FSaverage files
    fs_dir = fetch_fsaverage()
    subjects_dir = os.path.dirname(fs_dir)
    subject = "fsaverage"
    trans = "fsaverage"
    src = os.path.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
    bem = os.path.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")

    # Forward solution
    fwd = make_forward_solution(raw.info, trans=trans, src=src, bem=bem)
    src = fwd["src"]

    # Source activation
    tstep = 1.0 / raw.info["sfreq"]
    selected_label = read_labels_from_annot(
        subject, regexp=brain_label, subjects_dir=subjects_dir
    )[0]

    label = select_sources(
        subject,
        selected_label,
        location="center",
        extent=1,
        grow_outside=True,
        subjects_dir=subjects_dir
    )

    source_time_series = np.sin(
        2.0 * np.pi * frequency * np.arange(int(duration * raw.info["sfreq"])) * tstep
    ) * 10e-9 * amplitude

    gap_duration_s = gap_duration * raw.info["sfreq"]
    start_s = start * raw.info["sfreq"]
    events = np.zeros((n_repetition, 3), int)
    events[:, 0] = start_s + gap_duration_s * np.arange(n_repetition)
    events[:, 2] = 1  

    source_simulator = SourceSimulator(fwd["src"], tstep=tstep)
    source_simulator.add_data(label, source_time_series, events)
    raw = simulate_raw(raw.info, source_simulator, forward=fwd)
    cov = make_ad_hoc_cov(raw.info)
    add_noise(raw, cov, iir_filter=iir_filter)
    add_eog(raw)

    # Save simulated raw
    sim_dir = data_dir / "simulated"
    os.makedirs(sim_dir, exist_ok=True)
    if fname_save is None:
        raw.save(fname=sim_dir / f"{brain_label}_{frequency}Hz_{amplitude}-raw.fif", overwrite=True)
    else:
        raw.save(fname=fname_save)

    return raw

###############################################################################
# Step 2: Record baseline session
# --------------------------------
# We create a `NFRealtime` object to record a baseline session. This step is
# required to extract the blink template and the inverse operator for the 
# subject.

from ant import NFRealtime
import time

kwargs_sim = {
    "brain_label": "pericalcarine-lh",
    "frequency": 10,
    "amplitude": 1,
    "duration": 1,
    "gap_duration": 6,
    "n_repetition": 8,
    "start": 5,
    "fname_save": None
}
raw = simulate_eeg_raw(**kwargs_sim)

brain_label = "pericalcarine-lh"
frequency = 10
amplitude = 2
fname = Path.cwd().parent / "data" / "simulated" / f"{brain_label}_{frequency}Hz_{amplitude}-raw.fif"
kwargs = {
    "subject_id": "bert",
    "visit": 6,
    "subjects_dir": Path.cwd().parent / "data" / "subjects",
    "montage": "easycap-M1",
    "mri": False,
    "artifact_correction": False,
    "verbose": False
}

nf = NFRealtime(session="baseline", **kwargs)
# Connect to a mock LSL stream (we are using our simulated data)

nf.connect_to_lsl(mock_lsl=True, fname=fname)
time.sleep(4)

# Record baseline for 6 seconds
nf.record_baseline(baseline_duration=6)

# Extract blink template for artifact correction
nf.get_blink_template()

###############################################################################
# Step 3: Run the main neurofeedback session
# ------------------------------------------
# Now we run the main neurofeedback session using multiple modalities.
# The results will be saved in the subject's directory.

mods = [
    "sensor_power",
    "band_ratio",
    "entropy",
    "sensor_connectivity",
    "sensor_graph",
    "individual_peak_power"
]

nf.record_main(
    duration=60, 
    modality=mods,
    picks=None,
    winsize=1,
    estimate_delays=True,
    modality_params=None,
    show_raw_signal=False,
    show_nf_signal=True,
    time_window=20,
    show_design_viz=False,
    design_viz="VisualRorschach",
    show_brain_activation=False
)

###############################################################################
# Step 4: Results
# ----------------
# The neurofeedback session is complete. Results are automatically saved 
# in the subject directory specified during the creation of the NFRealtime
# object. You can now visualize or analyze the feedback signals as needed.
