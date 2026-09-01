"""Real-time single-trial decoding built from :mod:`mne.decoding` + scikit-learn.

:class:`RTDecode` wraps a scikit-learn ``sklearn.pipeline.Pipeline``
assembled from MNE's own decoding building blocks — :class:`mne.decoding.CSP`
or :class:`mne.decoding.Scaler` + :class:`mne.decoding.Vectorizer` — followed
by any scikit-learn classifier. It is fit **offline** on labelled epochs
(e.g. a calibration block or a prior session) and then queried **online**,
once per acquisition window. Unlike the built-in NF modalities in
:mod:`mne_rt.modalities`, which are pure functions of a window plus scalar
YAML config, a decoder is a stateful, externally-fitted object attached via
:meth:`~mne_rt.RTStream.set_decoder` — see the Notes section below.

Pipeline position::

    calibration epochs  →  RTDecode.fit(X, y)  →  RTStream.set_decoder()
    live window  →  RTStream._acquire loop  →  RTDecode.predict_proba()  →  nf_data["decode"]

Quick example::

    from mne_rt import RTDecode, RTStream

    # --- offline (calibration session) ---
    X_cal = ...  # shape (n_epochs, n_channels, n_times)
    y_cal = ...  # shape (n_epochs,), e.g. {"rest": 0, "motor_imagery": 1}
    decoder = RTDecode(info=epochs_info, spatial_filter="csp").fit(X_cal, y_cal)

    # --- real-time session ---
    nf = RTStream(subject_id="sub01", montage="easycap-M1", data_type="eeg")
    nf.connect_to_lsl(...)
    nf.set_decoder(decoder)
    nf.record_main(modality=["decode"], winsize=1.0)

Classes
-------
RTDecode
    Fit-offline / predict-online single-trial classifier for MNE-RT sessions.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from mne.decoding import CSP, Scaler, Vectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from mne_rt._logging import verbose

_SPATIAL_FILTERS = ("csp", "scaler")


class RTDecode:
    """Fit-offline / predict-online single-trial decoder for MNE-RT sessions.

    Wraps a scikit-learn ``sklearn.pipeline.Pipeline`` built from MNE's
    decoding primitives so that a classifier trained on labelled calibration
    epochs can be queried once per real-time acquisition window, wired into
    :meth:`~mne_rt.RTStream.record_main` via :meth:`~mne_rt.RTStream.set_decoder`
    as the ``"decode"`` modality (see :mod:`mne_rt.modalities`).

    Parameters
    ----------
    info : mne.Info
        Info describing the channels the decoder will be fit and queried on.
        Only used when ``spatial_filter="scaler"`` (passed to
        :class:`mne.decoding.Scaler` for channel-type-aware standardisation).
    estimator : sklearn-compatible classifier, default None
        Final pipeline step. Defaults to
        ``sklearn.linear_model.LogisticRegression()`` when ``None``. Must
        implement ``fit``/``predict``; ``predict_proba`` is additionally
        required for :meth:`predict_proba`.
    spatial_filter : "csp" | "scaler", default "csp"
        Feature-extraction step applied before *estimator*:

        * ``"csp"`` — :class:`mne.decoding.CSP`, log-variance of the
          ``n_components`` most discriminative spatial filters. Well suited
          to oscillatory/motor-imagery decoding; requires at least two
          classes.
        * ``"scaler"`` — :class:`mne.decoding.Scaler` (channel-type-aware
          standardisation) followed by :class:`mne.decoding.Vectorizer`
          (flattens to ``(n_epochs, n_channels * n_times)``). A simpler,
          filter-agnostic alternative when CSP's oscillatory-power
          assumption doesn't fit the decoding target (e.g. ERP decoding).
    n_components : int, default 4
        Number of CSP spatial filters. Only used when
        ``spatial_filter="csp"``; must not exceed the number of channels.
    scalings : "mean" | "median" | dict, default "mean"
        Passed to :class:`mne.decoding.Scaler`. Only used when
        ``spatial_filter="scaler"``.

    Attributes
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The assembled, not-yet-fit (until :meth:`fit` is called) pipeline.
    classes_ : ndarray | None
        Class labels seen during :meth:`fit`; ``None`` before fitting.
    n_channels_ : int | None
        Number of channels seen during :meth:`fit`; ``None`` before fitting.

    Notes
    -----
    As with :class:`~mne_rt.combiners.LearnedCombiner`, fitting happens
    **offline** against a full calibration recording — there is no
    incremental/online update during a live session. Re-fit and swap in a new
    :class:`RTDecode` instance between sessions instead.

    The fitted pipeline is sensitive to channel **count and order**: query
    windows must present the same channels, in the same order, as the ``X``
    passed to :meth:`fit`. When used as the ``"decode"`` modality, this means
    :meth:`~mne_rt.RTStream.record_main`'s ``picks`` must resolve to the same
    channel selection the decoder was fit on — a channel-*count* mismatch is
    caught at ``record_main()`` start, but a same-count reordering is not
    detected and will silently degrade predictions.

    The ``"decode"`` modality always reports :meth:`predict_proba` (a
    continuous class probability), not :meth:`predict` (a discrete label) —
    unlike the other NF modalities' outputs, a discrete label is not
    meaningful after the EMA smoothing / z-scoring every modality's value
    passes through in :meth:`~mne_rt.RTStream.record_main`. Use
    :meth:`predict` directly for offline/standalone decoding outside a live
    session.

    Examples
    --------
    >>> import numpy as np
    >>> from mne import create_info
    >>> from mne_rt import RTDecode
    >>> info = create_info(["C3", "Cz", "C4"], sfreq=256.0, ch_types="eeg")
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((20, 3, 256))
    >>> y = np.array([0, 1] * 10)
    >>> decoder = RTDecode(info=info, spatial_filter="csp", n_components=2)
    >>> _ = decoder.fit(X, y)
    >>> proba = decoder.predict_proba(rng.standard_normal((3, 256)))
    >>> proba.shape
    (2,)
    """

    def __init__(
        self,
        info: Optional[Any] = None,
        estimator: Optional[Any] = None,
        spatial_filter: str = "csp",
        n_components: int = 4,
        scalings: Any = "mean",
    ) -> None:
        if spatial_filter not in _SPATIAL_FILTERS:
            raise ValueError(
                f"`spatial_filter` must be one of {_SPATIAL_FILTERS}, got {spatial_filter!r}."
            )
        if spatial_filter == "scaler" and info is None:
            raise ValueError("`info` is required when spatial_filter='scaler'.")
        self.info = info
        self.estimator = estimator if estimator is not None else LogisticRegression()
        self.spatial_filter = spatial_filter
        self.n_components = n_components
        self.scalings = scalings
        self.classes_: Optional[np.ndarray] = None
        self.n_channels_: Optional[int] = None
        self._supports_proba: bool = hasattr(self.estimator, "predict_proba")
        self.pipeline: Pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Pipeline:
        if self.spatial_filter == "csp":
            steps = [
                ("csp", CSP(n_components=self.n_components, reg="ledoit_wolf", log=True)),
                ("clf", self.estimator),
            ]
        elif self.spatial_filter == "scaler":
            steps = [
                ("scaler", Scaler(self.info, scalings=self.scalings)),
                ("vectorizer", Vectorizer()),
                ("clf", self.estimator),
            ]
        return Pipeline(steps)

    @property
    def fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return self.classes_ is not None

    @verbose
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: Optional[Any] = None,
    ) -> "RTDecode":
        """Fit the pipeline on labelled calibration epochs.

        Parameters
        ----------
        X : array, shape (n_epochs, n_channels, n_times)
            Calibration epochs.
        y : array, shape (n_epochs,)
            Class label per epoch.
        verbose : bool | str | None, default None
            Control logging verbosity for this call, e.g. ``False`` to
            silence CSP's per-class covariance-estimation messages. See
            :func:`mne_rt.set_log_level`.

        Returns
        -------
        self : instance of RTDecode
            The fitted decoder, for chaining.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 3:
            raise ValueError(
                f"`X` must be a 3D (n_epochs, n_channels, n_times) array, got shape {X.shape}."
            )
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"`X` has {X.shape[0]} epochs but `y` has {y.shape[0]} labels.")
        self.pipeline.fit(X, y)
        self.classes_ = self.pipeline.classes_
        self.n_channels_ = X.shape[1]
        return self

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError(
                f"{type(self).__name__} must be fit() on calibration epochs "
                "before predict()/predict_proba() can be called."
            )

    def _prepare_window(self, window: np.ndarray) -> np.ndarray:
        window = np.asarray(window, dtype=np.float64)
        if window.ndim != 2:
            raise ValueError(
                f"`window` must be a 2D (n_channels, n_times) array, got shape {window.shape}."
            )
        return window[np.newaxis, ...]

    def predict(self, window: np.ndarray) -> Any:
        """Predict the class label for a single window.

        Parameters
        ----------
        window : array, shape (n_channels, n_times)
            One acquisition window, e.g. from ``stream.get_data(winsize)[0]``.

        Returns
        -------
        label : scalar
            The predicted class label (same dtype as the ``y`` passed to
            :meth:`fit`).
        """
        self._check_fitted()
        return self.pipeline.predict(self._prepare_window(window))[0]

    def predict_proba(self, window: np.ndarray) -> np.ndarray:
        """Predict per-class probabilities for a single window.

        Parameters
        ----------
        window : array, shape (n_channels, n_times)
            One acquisition window, e.g. from ``stream.get_data(winsize)[0]``.

        Returns
        -------
        proba : array, shape (n_classes,)
            Class probabilities in :attr:`classes_` order.
        """
        self._check_fitted()
        if not self._supports_proba:
            raise AttributeError(
                f"The estimator {type(self.estimator).__name__} does not implement "
                "predict_proba(); pass an estimator that does, e.g. "
                "LogisticRegression or SVC(probability=True)."
            )
        return self.pipeline.predict_proba(self._prepare_window(window))[0]

    def __repr__(self) -> str:
        status = "fitted" if self.fitted else "not fitted"
        return (
            f"<RTDecode | spatial_filter={self.spatial_filter!r}, "
            f"estimator={type(self.estimator).__name__}, {status}>"
        )
