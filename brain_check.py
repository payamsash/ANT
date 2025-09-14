import time
from pathlib import Path

import numpy as np
import pyvista as pv
from mne.io import read_raw_fif
from mne.minimum_norm import read_inverse_operator, apply_inverse_raw
from mne.datasets import fetch_fsaverage
from mne.surface import read_surface
import mne
from nibabel.freesurfer import read_morph_data


class RealTimeBrainVisualizer:
    """
    Real-time visualization of source activity on a cortical surface using MNE-Python and PyVista.
    Supports both left and right hemispheres.
    """

    def __init__(
        self,
        raw_fif_path,
        inv_fif_path,
        lambda2=1.0 / 9,
        pick_ori="normal",
        eeg_ref="average",
    ):
        """
        Initialize the visualizer.

        Parameters
        ----------
        raw_fif_path : str or Path
            Path to the raw FIF file.
        inv_fif_path : str or Path
            Path to the inverse operator FIF file.
        lambda2 : float
            Regularization parameter for inverse solution.
        pick_ori : str
            Orientation to pick ("normal", "vector", etc.).
        eeg_ref : str
            EEG reference method.
        """
        self.raw_fif_path = Path(raw_fif_path)
        self.inv_fif_path = Path(inv_fif_path)
        self.lambda2 = lambda2
        self.pick_ori = pick_ori
        self.eeg_ref = eeg_ref

        self.raw = None
        self.inverse_operator = None
        self.stc = None
        self.mesh = None
        self.scalars_full = None
        self.plotter = None
        self.actor = None
        self.verts_stc = {}
        self.data_hemi = {}
        self.hemi_offsets = {}

        mne.viz.set_3d_backend("pyvistaqt")

        self._load_data()
        self._setup_surface()
        self._setup_plotter()

    def _load_data(self):
        """Load raw data, set reference, and compute inverse solution."""
        self.raw = read_raw_fif(self.raw_fif_path, preload=True)
        self.raw.set_eeg_reference(self.eeg_ref, projection=True)
        self.inverse_operator = read_inverse_operator(self.inv_fif_path)
        self.stc = apply_inverse_raw(
            self.raw, self.inverse_operator, lambda2=self.lambda2, pick_ori=self.pick_ori
        )

        self.verts_stc["lh"] = self.stc.vertices[0]
        self.verts_stc["rh"] = self.stc.vertices[1]
        self.data_hemi["lh"] = self.stc.lh_data
        self.data_hemi["rh"] = self.stc.rh_data

    def _setup_surface(self, hemi_distance=100.0):
        """Fetch fsaverage surfaces and prepare PyVista mesh for both hemispheres."""
        fs_dir = fetch_fsaverage(verbose=False)

        verts_all = []
        faces_all = []
        offset = 0
        self.hemi_offsets = {}

        # Load sulcal depth
        lh_sulc = read_morph_data(Path(fs_dir) / 'surf' / 'lh.sulc')
        rh_sulc = read_morph_data(Path(fs_dir) / 'surf' / 'rh.sulc')
        sulc_all = np.hstack([lh_sulc, rh_sulc])

        # Load surfaces and separate hemispheres along x-axis
        for hemi in ["lh", "rh"]:
            surf_path = Path(fs_dir) / "surf" / f"{hemi}.inflated"
            verts_surf, faces_surf = read_surface(surf_path)

            # Apply translation: shift right hemisphere
            if hemi == "rh":
                verts_surf[:, 0] += hemi_distance

            self.hemi_offsets[hemi] = offset
            verts_all.append(verts_surf)

            # Convert faces to PyVista format
            faces_pv = np.hstack([
                np.full((faces_surf.shape[0], 1), 3),
                faces_surf + offset
            ]).astype(np.int64).ravel()
            faces_all.append(faces_pv)

            offset += verts_surf.shape[0]

        # Combine hemispheres
        verts_all = np.vstack(verts_all)
        faces_all = np.hstack(faces_all)

        # Create PyVista mesh
        self.mesh = pv.PolyData(verts_all, faces_all)
        self.mesh["base"] = sulc_all

        # Initialize activity scalars
        self.scalars_full = np.zeros(self.mesh.n_points)
        for hemi in ["lh", "rh"]:
            verts = self.verts_stc[hemi] + self.hemi_offsets[hemi]
            data = self.data_hemi[hemi][:, 0]
            self.scalars_full[verts] = data
        self.mesh["activity"] = self.scalars_full

    def _setup_plotter(self):
        """Initialize PyVista plotter, add mesh, and set camera."""
        self.plotter = pv.Plotter(window_size=(1800, 1200), lighting="three lights")
        self.plotter.set_background("black")

        # Add base mesh (sulcal depth)
        self.plotter.add_mesh(
            self.mesh,
            scalars="base",
            cmap="hot",
            smooth_shading=True,
            show_scalar_bar=False
        )

        # Determine activity range automatically
        max_activity = max(self.data_hemi["lh"].max(), self.data_hemi["rh"].max())

        # Add activity overlay as semi-transparent layer
        self.actor = self.plotter.add_mesh(
            self.mesh,
            scalars="activity",
            cmap="seismic",
            opacity=0.6,
            clim=[3, 10],
            smooth_shading=True,
            show_scalar_bar=False,
            interpolate_before_map=True
        )
        self.plotter.add_scalar_bar(
            title="Activity",
            n_labels=5,
            vertical=True,
            position_x=0.85,  # fraction of window width
            position_y=0.1,   # fraction of window height
            height=0.8
        )
        self.plotter.enable_eye_dome_lighting()

        # Set camera
        self.plotter.camera_position = "yz"
        self.plotter.camera.azimuth = 45

        # Show the interactive window
        self.plotter.show(interactive_update=True, auto_close=False)

    def run(self, interval=0.05):
        """
        Run the real-time visualization loop.

        Parameters
        ----------
        interval : float
            Time in seconds to wait between frames.
        """
        n_times = self.data_hemi["lh"].shape[1]
        for t in range(1, n_times):
            for hemi in ["lh", "rh"]:
                verts = self.verts_stc[hemi] + self.hemi_offsets[hemi]
                data = self.data_hemi[hemi][:, t]
                self.scalars_full[verts] = data

            self.mesh["activity"] = self.scalars_full
            self.mesh.Modified()
            self.plotter.update_scalars(self.mesh["activity"], render=True)
            time.sleep(interval)

        # Keep window open at the end
        self.plotter.show(auto_close=False)


if __name__ == "__main__":
    visualizer = RealTimeBrainVisualizer(
        raw_fif_path="/Users/payamsadeghishabestari/ANT/data/subjects/bert/baseline/visit_1-raw.fif",
        inv_fif_path="/Users/payamsadeghishabestari/ANT/data/subjects/bert/inv/visit_1-inv.fif",
    )
    visualizer.run(interval=0.05)
