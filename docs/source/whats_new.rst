.. _whats_new:

What's new
==========

.. _changes_1_1_0:

Version 1.1.0
--------------

*2026-07-29*

New features
^^^^^^^^^^^^

- **Feature combiners are now wired into the live loop.** The four
  :class:`~mne_rt.FeatureCombiner` subclasses existed and were unit-tested, but
  nothing ever called them: :meth:`~mne_rt.RTStream.record_main` had no way to
  use one, despite the base class documenting otherwise. It now accepts
  ``combiner=`` (and ``combined_name=``), reducing the per-modality values to a
  single scalar once per window, after z-scoring and smoothing. The result is an
  additional trace — the per-modality ones are kept — and is treated as a
  modality throughout: plotted, saved, broadcast over OSC/LSL, and able to drive
  its own protocol. Mixing features in native units without
  ``zscore_normalize=True`` warns, since the largest-scale feature would
  otherwise dominate — except for :class:`~mne_rt.ZScoredNormCombiner`, which
  normalises internally. :class:`~mne_rt.GeometricMeanCombiner` warns about the
  opposite combination: it takes a logarithm, so the negative half of a z-score
  distribution is floored and collapses the result.
  The combined trace is held at ``0.0`` until every feature's z-score has warmed
  up, so it never mixes native units and then jumps by orders of magnitude when
  normalisation engages.

- **Volume source spaces and subcortical ROIs.**
  :class:`~mne_rt.RTStream` accepts ``source_space="volume"``, building a
  volumetric grid instead of a cortical surface. This makes subcortical
  structures — hippocampus, amygdala, thalamus — reachable for the first
  time; they do not exist on the cortical surface at all. FreeSurfer's
  ``aparc+aseg`` atlas carries cortical parcels *and* subcortical structures,
  so one volume source space covers both without needing a mixed model.
  ``compute_inv_operator(volume_labels=[...])`` restricts the grid to just
  the labels you need, which takes a whole-brain 5 mm grid from ~14 600
  source points down to a few hundred.
- **New** :mod:`mne_rt.source` **module** with :class:`~mne_rt.SourceModel`,
  :class:`~mne_rt.ROI`, :func:`~mne_rt.resolve_rois` and
  :func:`~mne_rt.list_rois`. ROIs are named groups of atlas labels, so
  ``"Broca"`` can expand to pars opercularis ∪ pars triangularis and still
  behave as one region.
  :meth:`~mne_rt.SourceModel.roi_kernel` collapses the whole
  beamformer-plus-label-extraction chain into a single
  ``(n_roi, n_channels)`` matrix, built once. Applying it is a ~10 µs matmul
  instead of a ~650 ms round trip through
  :func:`~mne.extract_label_time_course`, and is numerically identical to the
  MNE route (~1e-15 relative error, pinned by a test).
- :meth:`~mne_rt.RTStream.compute_inv_operator` now also estimates and saves a
  **data covariance** (``*_desc-data_cov.fif``) and the **source space**
  (``*_src.fif``). The data covariance is what a beamformer adapts to and is a
  different quantity from the noise covariance; the source space is needed to
  map atlas labels onto source estimates and cannot be recovered from a
  beamformer afterwards. ``make_inverse=False`` skips the minimum-norm
  operator for beamformer-only sessions.
- :class:`~mne_rt.viz.NFPlot` now draws a live, toggleable dashed
  **threshold line** for the protocol driving each modality — fixed for
  :class:`~mne_rt.protocols.ThresholdProtocol`, adaptive (redrawn every
  push) for :class:`~mne_rt.protocols.ZScoreProtocol`.
- :class:`~mne_rt.viz.NFPlot` now shades a translucent green **reward
  span** over the time windows where the driving protocol is currently
  rewarding the subject, also independently toggleable. A 🟢 prefix marks
  reward-active updates in the status bar.
- :class:`~mne_rt.viz.EpochPlot` supports interactive **click-to-reject**
  bad-epoch marking: left-click a shaded epoch span to mark it bad
  (rendered in red), click again to restore it. Marked epochs are tracked
  via :attr:`~mne_rt.viz.EpochPlot.bad_epoch_ids` and
  :meth:`~mne_rt.viz.EpochPlot.is_epoch_bad`.
- CLI: ``--protocol {threshold,zscore}`` explicitly selects the reward
  protocol, with automatic inference from ``--threshold`` / ``--zscore-*``
  flags when omitted; new ``--zscore-min-std`` option avoids the default
  standard-deviation floor swamping small-magnitude features (e.g.
  ``sensor_power``).
- New :meth:`~mne_rt.RTStream.connect_to_array` connects a session to a
  plain in-memory numpy array instead of an LSL stream, driving the exact
  same :meth:`~mne_rt.RTStream.record_baseline` /
  :meth:`~mne_rt.RTStream.record_main` pipeline with no LSL networking or
  recorded file required — useful for offline analysis, unit tests, and
  demos. Backed by the new :class:`~mne_rt.ArrayStream`.
- New :class:`~mne_rt.RTDecode` wraps an :mod:`mne.decoding`
  (:class:`~mne.decoding.CSP` or :class:`~mne.decoding.Scaler` +
  :class:`~mne.decoding.Vectorizer`) + scikit-learn classifier pipeline for
  real-time single-trial decoding. Fit offline on labelled calibration
  epochs, then attach with :meth:`~mne_rt.RTStream.set_decoder` and query
  once per window as the new ``"decode"`` NF modality, alongside any other
  modality in :meth:`~mne_rt.RTStream.record_main`.

Bug fixes
^^^^^^^^^

- **Corrected several dependency lower bounds that were never satisfiable.**
  CI only ever installed the newest release of each dependency, so the minimum
  versions declared in ``pyproject.toml`` had never been tested. Installing
  them revealed four combinations that the metadata allowed but that fail on
  contact:

  * ``mne>=1.8`` → **1.9**: on 1.8, ``record_baseline()`` raises
    ``KeyError: 'type'`` when saving the baseline recording.
  * ``mne-connectivity>=0.7`` → **0.8**: ``spectral_connectivity_time`` gained
    ``"cohy"`` (how imaginary coherence is computed) only in 0.8, and native
    ``"imcoh"`` only in 0.9. On 0.7 both raise ``KeyError``.
  * ``nibabel>=5.0`` → **5.2**: 5.0 and 5.1 call ``np.sctypes``, removed in
    numpy 2.0, so importing mne-rt failed outright.
  * ``scikit-learn>=1.3`` → **1.4.2**: earlier versions import
    ``ComplexWarning`` from ``numpy.core.numeric``, also removed in numpy 2.0.

  The ``dev`` extra's ``pytest``/``pytest-cov`` are now lower-bounded too; with
  no bound at all a minimum-version resolution selected pytest 2.0.0 (2011),
  which does not build. A new **Minimum deps** CI job installs the oldest
  versions each bound allows, so these cannot drift back into fiction.
- **Source-space modalities could never run.** ``source_power``,
  ``source_connectivity`` and ``source_graph`` read the inverse operator from
  ``visit_{self.visit}-inv.fif``, but ``self.visit`` was never assigned and
  :meth:`~mne_rt.RTStream.compute_inv_operator` writes a BIDS-style
  ``sub-<id>_ses-<session>_task-baseline_inv.fif``. All three raised
  ``AttributeError`` on first use. They now take the operator from the session
  (falling back to the correct filename on disk) and report an actionable error
  when no baseline has been recorded.
- **``method: "imcoh"`` was broken in every connectivity modality.**
  :func:`~mne_connectivity.spectral_connectivity_time` has no ``"imcoh"`` in its
  bivariate dispatch table and raised ``KeyError``. Imaginary coherence is now
  computed as the imaginary part of ``"cohy"`` — an exact identity — in
  ``sensor_connectivity``, ``connectivity_ratio`` and ``source_connectivity``.
  Connectivity results are read from the raveled output rather than
  ``get_data(output="dense")``, which allocates a real array and silently
  discards the imaginary part.
- **Theta-band connectivity was impossible at the default window.** ``n_cycles``
  was hard-coded to 5; a 5-cycle wavelet at 4 Hz spans 1.25 s and does not fit a
  1 s window. ``n_cycles`` is now configurable per modality and accepts
  ``"auto"`` to scale with ``winsize``; wavelets that cannot fit are rejected
  up front with a message naming the offending frequency.
- ``record_main(modality_params=...)`` raised ``ValueError: Unknown method`` for
  the documented ``{modality: {param: value}}`` form, and ``AttributeError`` for
  the flat form (it called ``.update()`` on list-valued parameters). Both forms
  now work, and unknown parameter names are reported against the modality.
- ``source_connectivity`` and ``source_graph`` read ``self.params`` on every
  window, but :meth:`~mne_rt.RTStream.record_main` leaves it holding the *last*
  prepared modality's parameters — so running them alongside another modality
  silently used the wrong frequency band, metric or graph weights. All
  parameters are now captured at prep time. Same fix for ``argmax_freq``.
- ``sensor_connectivity`` raised ``IndexError`` when given a single channel pair
  and silently ignored the third and subsequent pairs. Unknown channel names now
  raise a clear error naming them.
- Fixed a crash when closing one plot window (e.g. :class:`~mne_rt.viz.RawPlot`
  or :class:`~mne_rt.viz.TopomapPlot`) while other plot windows remained open.
- :func:`mne.datasets.eegbci.load_data` is now called with
  ``update_path=True`` in the motor-imagery example, avoiding an
  interactive prompt that hung non-interactive/CI gallery builds.
- Fixed :func:`~mne_rt.tools.compute_bandpower` always raising ``TypeError``
  with ``method="multitaper"`` due to an invalid ``axis`` argument passed to
  MNE's :func:`~mne.time_frequency.psd_array_multitaper`.
- Fixed :meth:`~mne_rt.RTStream.create_report` raising ``FileNotFoundError``
  when called without a prior :meth:`~mne_rt.RTStream.record_baseline` or
  :meth:`~mne_rt.RTStream.record_main` call, since the session's output
  directories were never created in that path.
