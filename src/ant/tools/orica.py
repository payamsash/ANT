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

    # ---------- Nonlinearities ----------
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

    # ---------- Whitening ----------
    def _update_whitening(self, X):
        """Update whitening matrix with forgetting factor."""
        self.mean_ = self.forgetfac * self.mean_ + (1 - self.forgetfac) * X.mean(axis=1, keepdims=True)
        Xc = X - self.mean_

        cov_block = (Xc @ Xc.T) / X.shape[1]
        self.cov_ = self.forgetfac * self.cov_ + (1 - self.forgetfac) * cov_block

        d, E = np.linalg.eigh(self.cov_)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-10))
        self.whitening_ = E @ D_inv_sqrt @ E.T

        return self.whitening_ @ Xc

    def _apply_whitening(self, X):
        """Apply current whitening without updating (fix for #2)."""
        Xc = X - self.mean_
        return self.whitening_ @ Xc

    # ---------- Fit ----------
    def partial_fit(self, X):
        """Update unmixing matrix with a new block of EEG data."""
        if X.shape[0] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {X.shape[0]}")

        # fix for #1 (separate calibration vs online whitening)
        if self.online_whitening:
            if self.calibrate_pca and not self._calibrated:
                Xw = self._update_whitening(X)
                self._calibrated = True
            elif self.calibrate_pca and self._calibrated:
                Xw = self._apply_whitening(X)
            else:
                Xw = self._update_whitening(X)
        else:
            Xw = X - X.mean(axis=1, keepdims=True)

        # ICA update
        Y = self.W @ Xw
        gY, gprime = self._nonlinear_func(Y)

        # fix for #3 (include gprime term for stability)
        N = X.shape[1]
        dW = ((np.eye(self.n_channels) - np.mean(gprime, axis=1)[:, None]) @ self.W +
                (gY @ Y.T) / N @ self.W)

        self.W += self.learning_rate * dW
        self.W, _ = np.linalg.qr(self.W)  # keep W orthogonal

        return self

    # ---------- Transform ----------
    def transform(self, X):
        """Apply learned unmixing to data (without updating W)."""
        if self.online_whitening:
            if self.calibrate_pca and self._calibrated:
                Xw = self._apply_whitening(X)  # fix for #2
            else:
                Xw = self._apply_whitening(X)  # never update in transform
        else:
            Xw = X - X.mean(axis=1, keepdims=True)
        return self.W @ Xw

    def fit_transform(self, X):
        """Fit and return sources."""
        self.partial_fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        """ Reconstruct EEG from sources. """
        if self.W is None:
            raise RuntimeError("Model has not been fitted yet.")
        A = np.linalg.pinv(self.W)
        X_rec = A @ X
        X_rec += self.mean_
        return X_rec

    # ---------- Blink IC detection ----------
    def find_blink_ic(self, template_map, threshold=0.7):
        if self.W is None:
            raise RuntimeError("Model has not been fitted yet.")

        A = np.linalg.pinv(self.W)  # mixing matrix
        corrs = np.array([np.corrcoef(A[:, ic], template_map)[0, 1]
                            for ic in range(A.shape[1])])
        blink_idx = np.where(np.abs(corrs) > threshold)[0].tolist()
        return blink_idx, corrs