"""Shared pytest fixtures and configuration."""

import sys
from pathlib import Path

import pytest

# Ensure the package source is on sys.path for all tests
_src = Path(__file__).parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

# test_rt_epochs_erp.py is a standalone script rather than a set of test cases
# -- everything in it runs at import.  Collecting it therefore downloads the
# MNE sample dataset and runs a live 60 s LSL replay on every session, and
# contributes no tests in return; its own assertions would surface as
# collection *errors*, not failures.  It stays runnable by hand, as its
# docstring describes.  The wiring it covers is tested in test_rt_epochs.py.
collect_ignore = ["test_rt_epochs_erp.py"]


@pytest.fixture(scope="module")
def meg_info():
    """Return MEG info from MNE sample data, or skip if unavailable."""
    pytest.importorskip("mne")
    try:
        import mne

        data_path = mne.datasets.sample.data_path(download=False, verbose=False)
        fname = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
        if not fname.exists():
            pytest.skip("MNE sample data not downloaded")
        raw = mne.io.read_raw_fif(fname, preload=False, verbose=False)
        return raw.info
    except Exception:
        pytest.skip("MNE sample data not available")
