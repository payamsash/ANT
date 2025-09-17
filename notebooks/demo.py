from pathlib import Path
from ant import NFRealtime
import time


brain_label = "pericalcarine-lh"
frequency = 10
amplitude = 2
fname = Path.cwd().parent / "data" / "simulated" / f"{brain_label}_{frequency}Hz_{amplitude}-raw.fif"

kwargs = {
            "subject_id": "bert",
            "visit": 14,
            "subjects_dir": Path.cwd().parent / "data" / "subjects",
            "montage": "easycap-M1",
            "mri": False,
            "artifact_correction": False,
            "verbose": False
        }
nf = NFRealtime(session="baseline", **kwargs)
nf.connect_to_lsl(mock_lsl=True, fname=fname)
time.sleep(4)
nf.record_baseline(baseline_duration=6)
nf.get_blink_template()


nf.record_main(
                duration=40, 
                modality=["band_ratio", "entropy"],# , "sensor_connectivity", "sensor_graph", "individual_peak_power"],
                picks=None,
                winsize=1,
                estimate_delays=True,
                modality_params=None,
                show_raw_signal= False,
                show_nf_signal= False,
                time_window=20,
                show_design_viz= False,
                design_viz= "VisualRorschach",
                show_brain_activation=True
                )

