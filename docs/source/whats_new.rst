.. _whats_new:

What's new
==========

.. _changes_1_2_0:

Version 1.2.0
--------------

*unreleased*

New features
^^^^^^^^^^^^

- **Every analysis window now carries its own onset.** Sessions previously saved
  only a start time, leaving a window's time to be inferred as
  ``index * winsize/2``. That inference is not sound. It drifts — measured at
  ~0.4 s over a 600 s run for a light configuration, and worse as the
  per-window computation approaches the hop — and any window dropped for being
  the wrong length shifted every later index silently, with no record that it
  had happened. Most importantly it cannot produce an absolute time at all,
  which is what aligning against a stimulus log requires.
  :meth:`~mne_rt.RTStream.record_main` now records the real onset and duration
  per window, exposed as
  :attr:`~mne_rt.RTStream.window_onsets` / ``window_durations`` and saved under
  a ``"windows"`` block in the session JSON: ``onset`` relative to the first
  window, ``duration``, and ``onset_lsl`` on the absolute LSL clock, which is
  the one that aligns directly against a stimulus log.
- ``record_main`` now writes the BIDS ``_beh.tsv`` table by default
  (``save_tsv=True``). It previously wrote only the session JSON, so the
  per-window table never reached disk unless ``save()`` was called by hand. The
  table leads with ``onset`` and ``duration`` as BIDS requires, and the session
  JSON gained a ``"columns"`` block describing every column — BIDS would put
  that in a ``_beh.json`` sidecar, but that filename is already the session
  payload's.
- A window dropped for having the wrong number of samples is now counted
  (``meta["n_short_windows"]``) and explained once in the log. This is a
  deterministic failure rather than an occasional one: ``winsize * sfreq`` is
  truncated when sizing the window but rounded up when fetching it, so a
  non-integral product silently discards *every* window and produces an empty
  session.
- **:class:`~mne_rt.RTEpochs` can now cut epochs on a separate LSL marker
  stream**, which is what a PsychoPy paradigm publishes. Until now the event
  codes had to arrive on a ``"stim"``-type channel *inside* the M/EEG stream
  itself, so an amplifier bridged to LSL without a stim channel could not
  trigger an epoch at all. Pass ``event_stream_name`` (or ``event_source_id``)
  to :meth:`~mne_rt.RTEpochs.connect_to_lsl`, and ``event_channels`` then names
  a channel of that stream.

  When an event stream is used, both streams are connected with
  ``processing_flags="all"``: two LSL streams share a time base only once their
  clocks are synchronised, and without it markers published from a second
  machine sit at an arbitrary offset from the data, so every epoch is cut in
  the wrong place and nothing reports it.

  The publisher has three requirements, each documented with a runnable snippet
  on :meth:`~mne_rt.RTEpochs.connect_to_lsl`: the outlet must be **numeric**
  (mne-lsl refuses string streams outright, and ``"string"`` is the format most
  PsychoPy marker examples use), it should **label its channel** — an
  unlabelled outlet is exposed as ``"0"`` — and the codes must be positive
  integers no greater than 32767.

Numeric output changes
^^^^^^^^^^^^^^^^^^^^^^

**Standardisation no longer depends on the units of your feature.** If you use
``zscore_normalize=True``, :class:`~mne_rt.protocols.ZScoreProtocol`,
:class:`~mne_rt.protocols.TransferProtocol`,
:class:`~mne_rt.protocols.RLProtocol`,
:class:`~mne_rt.protocols.ThresholdProtocol` or
:class:`~mne_rt.ZScoredNormCombiner` on a power-like modality, **the numbers
this version produces differ from 1.1.0 — the previous ones were wrong.**

Every running standard deviation was floored with an absolute constant
(``1e-6``, or ``1e-9`` in the combiners). EEG band power is ~1e-12 V²/Hz and MEG
~1e-24 T²/Hz, so for those features the floor did not *guard* the standard
deviation, it *replaced* it: measured on synthetic EEG band power, a z-score
that should have been 1.0 came out as 7.9e-9, and on MEG, 9.0e-21. Rewards
therefore almost never fired, and the adaptive threshold line sat
indistinguishably close to the mean. Features of order 1 — connectivity,
entropy, laterality, ``erd_ers`` — were unaffected, which is why this went
unnoticed.

There is now no floor. A standard deviation is used exactly as observed,
whatever its magnitude, and the only special case is a signal with *no* spread,
which has no z-score and yields ``0.0`` — no crossing, no reward. Affected
modalities: ``sensor_power``, ``individual_peak_power``, ``source_graph`` and,
marginally, ``scp``.

Two consequences worth expecting:

- ``current_threshold`` moves. On ``sensor_power`` it was ``mean ± 5e-7`` — far
  outside the data — and is now a real ``mean ± zscore_threshold × σ`` boundary,
  so the dashed line on :class:`~mne_rt.viz.NFPlot` finally sits where the
  rewards are.
- With ``zscore_normalize=True`` the plot's display scales are now ``1.0``,
  since the traces are in standard deviations rather than native units. The
  raw value is still passed through during the warmup windows, so the trace
  sits flat near the axis origin until normalisation engages.

``min_std`` (on :class:`~mne_rt.protocols.ZScoreProtocol` and ``--zscore-min-std``)
now defaults to ``None``, meaning no floor. Passing a float keeps its old
meaning exactly, so an existing script that set one explicitly is unaffected.
The CLI help and :doc:`cli` previously advised choosing a floor below your
signal's magnitude (``1e-15`` for ``sensor_power``); that advice is obsolete and
has been removed — passing a floor now *opts back into* the old behaviour.

Bug fixes
^^^^^^^^^

- **Every :class:`~mne_rt.RTEpochs` epoch carried a trailing all-zero sample.**
  The epoch buffer was sized as ``round((tmax - tmin) * sfreq) + 1``, but
  mne-lsl produces ``ceil((tmax - tmin) * sfreq)`` samples with
  ``endpoint=False`` — one fewer. The last column was therefore never written,
  so it stayed at zero: a step to zero at the end of every epoch, biasing any
  average taken over the full window, and a reported ``tmax`` one sample past
  the truth. The width and ``tmin`` now come from
  :attr:`~mne_lsl.stream.EpochsStream.times` rather than being re-derived.
- **:meth:`~mne_rt.RTEpochs.connect_to_lsl` left stale handles behind.**
  Reconnecting without an event stream after a session that used one passed the
  now-disconnected marker stream to ``EpochsStream``, which refuses it — so the
  second connection failed with an error about a stream the caller had not
  asked for. A failure during setup also stranded a running mock player and a
  connected stream, with ``_connected`` still ``False`` so the caller had no
  handle to clean up with; setup is now unwound on the way out.
- **A sensor-space baseline no longer downloads 700 MB of anatomy.**
  :meth:`~mne_rt.RTStream.record_baseline` ended with an unconditional
  :meth:`~mne_rt.RTStream.compute_inv_operator`, which for the default
  ``subject_fs_id="fsaverage"`` calls :func:`mne.datasets.fetch_fsaverage` —
  so *every* baseline reached across the network for a head model, including
  the many sessions that only ever compute sensor-space features. It made an
  OSF outage look like a library failure, and it was a recurring source of red
  CI on tests that never touch a source. The four sensor-space tests it
  affected go from 47.7 s to 17.3 s with the anatomy already cached; on a cold
  machine the difference is the download itself.

  The head model is now built on first use — by a source-space modality, by the
  brain-activation display, or by calling ``compute_inv_operator()`` directly,
  which is also how you pass it non-default arguments. Nothing about a
  source-space session changes except *when* the model is built; a sensor-space
  session never builds one at all. The ``inv/`` files therefore appear at that
  point rather than at the end of the baseline, and :attr:`~mne_rt.RTStream.inv`
  is no longer set by ``record_baseline`` alone.
  :meth:`~mne_rt.RTStream.fit_gedai` triggers the build too when
  ``use_leadfield=True``, since it reads the forward solution and would
  otherwise fall back to band-filter mode — a different denoising algorithm —
  with only a log line to say so.
- ``ArrayStream`` timestamped its samples with ``time.time()`` while the
  acquisition loop runs on ``local_clock()`` — the Unix epoch against
  seconds-since-boot, about 1.8e9 seconds apart. Any timing derived from the
  offline/test stream was therefore on a different clock from the live one.
- The two behavioural-TSV writers (:meth:`~mne_rt.RTStream.save` and
  :func:`~mne_rt.tools.save_as_bids`) were independent implementations that had
  already drifted apart on float formatting and on ragged-column padding. They
  now share one function, so a column written by both is written identically,
  and :func:`~mne_rt.tools.save_as_bids` gains the ``_beh.json`` sidecar it
  never wrote. Which columns appear still depends on what each caller supplies:
  ``save_as_bids`` passes feature values only, while a session written by
  ``record_main`` also carries timing, reward and SNR columns.
- **The adaptive z-score tracked the wrong statistic.** With ``zscore_alpha >
  0``, :meth:`~mne_rt.RTStream.record_main` exponentially averaged
  ``abs(value - mean)`` — the mean absolute deviation, ≈0.798σ for Gaussian
  input — into a variable seeded with a standard deviation, inflating every
  z-score by roughly 25% on top of the floor problem. It now uses the standard
  EMA-weighted variance recursion. The mean and the variance are also advanced
  from the same ``delta``; previously the variance used the pre-update mean
  while the returned z-score used the post-update one.
- Standard deviations are now sample (``ddof=1``) throughout. Two of the three
  estimators already were, so a :class:`~mne_rt.protocols.ZScoreProtocol`
  threshold line was drawn over a trace standardised by a different convention —
  a 5.4% mismatch at the default warmup, and 41% at ``zscore_warmup=2``.
- :class:`~mne_rt.GeometricMeanCombiner` clipped every input to ``floor`` before
  taking its logarithm. For band power (~1e-14) against the ``1e-9`` default
  that meant *all* inputs clipped, so the result was the constant ``1e-9``
  regardless of the data. ``floor`` now defaults to ``None``: positive values of
  any magnitude pass through, and a non-positive one — for which the geometric
  mean is undefined — is dropped from the product rather than clamped.
- :class:`~mne_rt.protocols.TransferProtocol` now rejects a prior recording with
  zero variance at construction, naming the file. Such a prior cannot seed a
  z-score, and under the new rule it would otherwise produce no reward for an
  entire session with nothing to indicate why.
- :class:`~mne_rt.ZScoredNormCombiner` now requires ``warmup >= 2``, which its
  sample-variance baseline needs.

.. _changes_1_1_0:

Version 1.1.0
--------------

*2026-07-29*

New features
^^^^^^^^^^^^

- **The same modality can now run several times in one session.** A modality
  name may carry an instance label after an ``"@"``, so
  ``modality=["source_connectivity@theta", "source_connectivity@alpha"]``
  computes one measure in two bands at once — previously impossible, because
  :meth:`~mne_rt.RTStream.record_main` used the modality name as both the
  dispatch key and the key for every piece of per-window state, so a repeated
  name collapsed the two into one. The base modality still selects the config
  entry and the compute function; the full name distinguishes the instances
  everywhere they surface: plot traces, protocol keys, combiner feature names,
  OSC addresses, LSL channels and saved columns. In ``modality_params``, a key
  naming a base modality applies to all its instances and an instance's own
  entry takes precedence, so shared settings need only be written once.
  Protocols are stateful, so each instance needs its own protocol object;
  passing one keyed by a base modality that has instances now raises rather
  than quietly firing a single protocol several times per window.
- ``LSLSender`` now publishes its channel names in the stream description.
  They were previously stored on the sender and dropped before the outlet was
  built, despite the documented behaviour, leaving subscribers to identify
  values by position — ambiguous as soon as two channels share a base modality
  and differ only by instance label.
- The sensor → ROI operator is cached alongside the forward model, so several
  instances of a source modality that share an ROI set build it once rather
  than once each.
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

- **Source-space connectivity between arbitrary ROI pairs.**
  ``source_connectivity`` previously *required* one left-hemisphere and one
  right-hemisphere label and rejected anything else, so a within-hemisphere
  pair — Broca ↔ Wernicke, say — was impossible, as was any pair involving a
  subcortical structure. It now takes a list of ``rois`` and explicit
  ``pairs``, in any combination of hemispheres and of cortical/subcortical
  regions, and reports the mean across pairs (``signed: true`` keeps the
  lead/lag sign instead of the magnitude).
- All three source modalities (``source_power``, ``source_connectivity``,
  ``source_graph``) now run on :class:`~mne_rt.SourceModel`'s cached operator
  and share one forward model and beamformer per ``(method, atlas)``, so
  running the same measure in several frequency bands no longer rebuilds the
  head model for each. ``source_graph`` gained ``pair`` for naming the edge to
  report, and ``source_connectivity`` gained ``inverse_method`` — its
  ``method`` is the *connectivity metric*, which was ambiguous before.
- Minimum-norm inverses now also use the cached-kernel path, recovered by
  pushing an identity "recording" through
  :func:`~mne.minimum_norm.apply_inverse_raw` (public API, exact for ``MNE``
  and to floating point for the noise-normalised methods). Surface source
  modalities get *faster* as a result: measured per window, dSPM drops from
  69 ms to 1.1 ms and MNE from 17 ms to 1.1 ms.
- Phase-based connectivity metrics are now refused on a magnitude source
  estimate. A free-orientation solution combines three orientations by norm,
  discarding phase, so ``imcoh``/``plv``/``wpli`` computed on it are
  meaningless; they raise instead of returning a plausible-looking number.
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

- **A single bad channel broke the session, and silently corrupted
  connectivity.** ``Stream.get_data()`` defaults to ``exclude="bads"``, but
  ``rec_info`` keeps those channels, and nine places indexed the acquired array
  by position in ``rec_info["ch_names"]`` — the LMS reference channel, ORICA's
  channel count, the raw viewer, GEDAI's dimension, and every connectivity and
  source modality's channel lookup. One bad channel shifted all of them:
  :meth:`~mne_rt.RTStream.record_baseline` raised ``len(data) does not match
  len(info["ch_names"])`` before a session could start, and
  ``sensor_connectivity`` would have computed on the wrong channel pair. Those
  lookups now use the acquired array's own channel list. Bad channels are
  deliberately still listed in ``rec_info``, since the forward model and
  beamformer exclude them themselves and
  :class:`~mne_rt.tools.RTMaxwellFilter` needs them marked to reconstruct them
  through the SSS expansion.
- ``instantaneous_phase`` and ``laterality_erd_ers`` had no entry in the
  display-scale table, so running either with ``show_nf_signal=True`` raised
  ``KeyError`` as soon as the first window was plotted. Both now have one, and
  an unknown name falls back to its base modality's scale rather than raising.
- OSC and LSL send failures in the neurofeedback loop were swallowed silently,
  so a session could run to completion delivering no feedback at all with no
  indication of why. The first failure of each is now logged.
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
- **Bad channels broke every source-space window.** The cached source operator's
  columns follow the forward model's channels, which exclude ``info["bads"]`` and
  any channel without a digitised position, but it was applied to the full data
  window. One bad channel — routine in real recordings — therefore raised a shape
  error on every window, and a same-count reordering would have silently produced
  wrong ROI time courses. Channels are now selected and reordered to match the
  operator, as :func:`~mne.beamformer.apply_lcmv_raw` does internally.
- **``compute_inv_operator(volume_labels=[...])`` produced an unusable source
  space.** :func:`~mne.setup_volume_source_space` returns one source space *per
  label* unless ``single_volume=True``, while label extraction reads a single
  interpolator — so the ROI-restricted path documented as recommended yielded
  rows of the wrong length, or silently all-zero ones.
- A volume session with default modality parameters failed with
  ``FileNotFoundError: Volumetric atlas 'aparc' not found``: the config shipped
  ``atlas: "aparc"`` for every source modality, so the session's
  ``source_atlas`` was never consulted. The config now defaults to ``null``,
  meaning "use the session's".
- The data covariance was estimated from ``raw_baseline`` rather than from the
  recording the forward model was built from, so :func:`~mne.beamformer.make_lcmv`
  received an average-reference-projected ``info`` alongside an unprojected
  covariance.
- Cached source models are now discarded when a new baseline is recorded;
  previously a second :meth:`~mne_rt.RTStream.record_baseline` left
  :meth:`~mne_rt.RTStream.record_main` using the beamformer built from the
  *previous* baseline.
- A multi-label ROI containing a label with no source points was silently
  attenuated: label sizes were counted in atlas voxels, so an empty label still
  took a share of the weighting. Sizes are now counted in source points, and the
  "ROI contains no source points" warning fires as intended.
- **EEG source models were built without an average reference.** MNE requires
  an average reference for EEG source modelling —
  :func:`~mne.minimum_norm.apply_inverse_raw` rejects data without one, and
  :meth:`~mne_rt.RTStream.record_main` duly applies the projection to every
  analysis window — but the forward model, the covariances and the beamformer
  were all built from an ``info`` without it, so MNE warned on every baseline
  that the covariance was adversely affected and the whitener did not match
  the data it was applied to. The projection is now set before the covariances
  are estimated. This changes source-space values slightly; it is a
  correctness fix, not a cosmetic one.
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
