import numpy as np


class ORICA:
    """Online Recursive ICA (ORICA) for EEG data.

    Parameters
    ----------
    n_channels : int
        Number of input EEG channels.
    learning_rate : float
        Learning rate for ICA updates.
    block_size : int
        Size of blocks for online updates.
    online_whitening : bool
        If True, perform online whitening. If False, assumes input is already whitened.
    calibrate_pca : bool
        If True, estimate whitening matrix from initial block and fix it.
        If False, update recursively.
    forgetfac : float
        Forgetting factor for online covariance (0 < forgetfac <= 1).
        Values < 1 allow adaptation to nonstationary signals.
    nonlinearity : str
        Nonlinearity function: 'tanh', 'pow3', or 'gauss'.
    random_state : int | None
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_channels,
        learning_rate=0.1,
        block_size=256,
        online_whitening=True,
        calibrate_pca=False,
        forgetfac=1.0,
        nonlinearity="tanh",
        random_state=None,
    ):
        self.n_channels = n_channels
        self.learning_rate = learning_rate
        self.block_size = block_size
        self.online_whitening = online_whitening
        self.calibrate_pca = calibrate_pca
        self.forgetfac = forgetfac
        self.nonlinearity = nonlinearity

        rng = np.random.default_rng(random_state)
        self.W = rng.standard_normal((n_channels, n_channels))
        self.W, _ = np.linalg.qr(self.W)

        self.mean_ = np.zeros((n_channels, 1))
        self.cov_ = np.eye(n_channels)
        self.whitening_ = np.eye(n_channels)
        self._calibrated = False

    def _nonlinear_func(self, Y):
        if self.nonlinearity == "tanh":
            gY = np.tanh(Y)
            gprime = 1.0 - gY**2
        elif self.nonlinearity == "pow3":
            gY = Y**3
            gprime = 3 * Y**2
        elif self.nonlinearity == "gauss":
            gY = Y * np.exp(-0.5 * Y**2)
            gprime = (1 - Y**2) * np.exp(-0.5 * Y**2)
        else:
            raise ValueError(f"Unknown nonlinearity {self.nonlinearity}")
        return gY, gprime

    def _update_whitening(self, X):
        """Update whitening matrix with forgetting factor."""
        # update mean
        self.mean_ = self.forgetfac * self.mean_ + (1 - self.forgetfac) * X.mean(axis=1, keepdims=True)
        Xc = X - self.mean_

        cov_block = (Xc @ Xc.T) / X.shape[1]
        self.cov_ = self.forgetfac * self.cov_ + (1 - self.forgetfac) * cov_block

        d, E = np.linalg.eigh(self.cov_)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-10))
        self.whitening_ = E @ D_inv_sqrt @ E.T

        return self.whitening_ @ Xc

    def partial_fit(self, X):
        """Update unmixing matrix with a new block of EEG data.

        Parameters
        ----------
        X : array, shape (n_channels, n_times)
            New data block (continuous EEG).
        """
        if X.shape[0] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {X.shape[0]}")

        # whitening
        if self.online_whitening:
            if self.calibrate_pca and not self._calibrated:
                # fix whitening from first block
                self._update_whitening(X)
                self._calibrated = True
            Xw = self._update_whitening(X)
        else:
            Xw = X - X.mean(axis=1, keepdims=True)

        Y = self.W @ Xw
        gY, gprime = self._nonlinear_func(Y)
        dW = (np.eye(self.n_channels) + (gY @ Y.T) / X.shape[1]) @ self.W
        self.W += self.learning_rate * dW
        self.W, _ = np.linalg.qr(self.W)

        return self

    # ---------- Transform ----------
    def transform(self, X):
        """Apply learned unmixing to data (without updating W)."""
        if self.online_whitening:
            Xw = self._update_whitening(X)
        else:
            Xw = X - X.mean(axis=1, keepdims=True)
        return self.W @ Xw

    def fit_transform(self, X):
        """Fit and return sources."""
        self.partial_fit(X)
        return self.transform(X)

    def find_blink_ic(self, template_map, threshold=0.8):
        """
        Find IC(s) that best match a template blink spatial map.

        Parameters
        ----------
        template_map : ndarray, shape (n_channels,)
            Spatial topography of a blink component (from prior ICA).
        threshold : float
            Correlation threshold to accept ICs as blink.

        Returns
        -------
        blink_idx : list of int
            Indices of components matching the template.
        corrs : ndarray, shape (n_components,)
            Correlation values for each IC.
        """
        if self.W is None:
            raise RuntimeError("Model has not been fitted yet.")

        # mixing matrix (channels × components)
        A = np.linalg.pinv(self.W)
        corrs = np.array([np.corrcoef(A[:, ic], template_map)[0, 1]
                            for ic in range(A.shape[1])])
        blink_idx = np.where(np.abs(corrs) > threshold)[0].tolist()
        return blink_idx, corrs 