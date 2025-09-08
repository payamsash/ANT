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
                            add_noise
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
                        verbose=None
                    ):
    """Simulate EEG data with a sinusoidal source in a given brain label.

    A forward solution is created for ``fsaverage`` and a sinusoidal source
    is injected into the specified brain label. The simulated signal is then
    projected to sensor space, optionally repeated with gaps, and saved as
    an MNE Raw object.

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

    Notes
    -----
    The simulated raw file is also written to disk in the ``data/simulated``
    subdirectory, with a filename that includes the brain label, frequency,
    and amplitude.

    Examples
    --------
    >>> raw = simulate_eeg_raw('bankssts-lh', frequency=10, amplitude=1e-9,
    ...                        duration=2., gap_duration=1., n_repetition=5,
    ...                        start=0.)
    >>> raw.plot()
    """

    set_log_level(verbose=verbose)
    
    ## load the data
    data_dir = Path.cwd().parent / "data" 
    fname_vhdr = data_dir / "sample" / "sample_data.vhdr" 
    raw = read_raw_brainvision(fname_vhdr, preload=True)

    ## montaging and removing ecg channels
    new_ch_names = raw.info['ch_names'].copy()
    new_ch_names[58] = 'Fpz' # change FPz to Fpz
    mapping = dict(zip(raw.info['ch_names'], new_ch_names))
    rename_channels(raw.info, mapping)
    raw.drop_channels(ch_names=["HRli", "HRre"], on_missing='raise')
    raw.set_montage("easycap-M1", on_missing='warn')

    ## download fsaverage files
    fs_dir = fetch_fsaverage()
    subjects_dir = os.path.dirname(fs_dir)
    subject = "fsaverage"
    trans = "fsaverage"
    src = os.path.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
    bem = os.path.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")

    ## create forward solution
    fwd = make_forward_solution(raw.info, trans=trans, src=src, bem=bem)
    src = fwd["src"]

    ## put source activation
    tstep = 1.0 / raw.info["sfreq"]
    selected_label = read_labels_from_annot(
                                            subject,
                                            regexp=brain_label,
                                            subjects_dir=subjects_dir
                                            )[0]
    label = select_sources(
                            subject,
                            selected_label,
                            location="center",
                            extent=1,
                            grow_outside=True, 
                            subjects_dir=subjects_dir
                            )
    source_time_series = np.sin(2.0 * np.pi * frequency \
                                                * np.arange(int(duration * raw.info["sfreq"])) * tstep) \
                                                                                        * 10e-9 * amplitude

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

    ## save
    sim_dir = data_dir / "simulated"
    os.makedirs(sim_dir, exist_ok=True)
    raw.save(fname=sim_dir / f"{brain_label}_{frequency}Hz_{amplitude}-raw.fif")

    return raw