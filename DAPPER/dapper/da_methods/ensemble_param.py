"""The EnKF and other ensemble-based methods."""

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
from numpy import diag, eye, sqrt, zeros

import dapper.tools.multiproc as multiproc
from dapper.stats import center, inflate_ens, mean0
from dapper.tools.linalg import mldiv, mrdiv, pad0, svd0, svdi, tinv, tsvd
from dapper.tools.matrices import funm_psd, genOG_1
from dapper.tools.progressbar import progbar
from dapper.tools.randvars import GaussRV
from dapper.tools.seeding import rng

from . import da_method

try:
    from dapper.mods.NPZ import Np
except ImportError:
    Np = 3

@dataclass(kw_only=True)
class ens_method:
    """Default ensemble arguments (shared via inheritance)."""

    infl: float = 1.0
    rot: bool = False
    fnoise_treatm: str = "Stoch"


class EnKF_param(da_method, ens_method):
    """The ensemble Kalman filter.

    Refs: [evensen2009a][].
    """

    upd_a: str
    N: int

    log_transform: bool = False
    truncated: bool = False

    def assimilate(self, HMM, xx, yy):
        # Init
        E = HMM.X0.sample(self.N)
        self.stats.assess(0, E=E)

        # Cycle
        for k, ko, t, dt in progbar(HMM.tseq.ticker):
            E = HMM.Dyn(E, t - dt, dt)

            if Np > 0 and E.shape[1] > Np:
                E_phys = add_noise(E[:, :-Np], dt, HMM.Dyn.noise, self.fnoise_treatm)
                E = np.hstack([E_phys, E[:, -Np:]])
            else:
                E = add_noise(E, dt, HMM.Dyn.noise, self.fnoise_treatm)

            # Analysis update
            if ko is not None:
                self.stats.assess(k, ko, "f", E=E)
                E = EnKF_analysis(
                    E,
                    HMM.Obs(ko)(E),
                    HMM.Obs(ko).noise,
                    yy[ko],
                    self.upd_a,
                    self.stats,
                    ko,
                    self.log_transform
                )

                if self.truncated:
                    if Np > 0:
                        E[:, :-Np] = np.maximum(E[:, :-Np], 1e-20)
                    else:
                        E = np.maximum(E, 1e-20)

                E = post_process(E, self.infl, self.rot)

            self.stats.assess(k, ko, E=E)


def EnKF_analysis(E, Eo, hnoise, y, upd_a, stats=None, ko=None, log_transform=False):
    """Perform the EnKF analysis update.

    This implementation includes several flavours and forms,
    specified by `upd_a`.

    Main references: [sakov2008b][],
    [sakov2008a][], [hoteit2015a][]
    """

    # log transform -----------------
    if log_transform:
        E = np.log(np.maximum(E, 1e-20))
    # ---------------------------------

    R = hnoise.C  # Obs noise cov
    N, Nx = E.shape  # Dimensionality
    N1 = N - 1  # Ens size - 1

    mu = np.mean(E, 0)  # Ens mean
    A = E - mu  # Ens anomalies

    xo = np.mean(Eo, 0)  # Obs ens mean
    Y = Eo - xo  # Obs ens anomalies
    dy = y - xo  # Mean "innovation"

    if "PertObs" in upd_a:
        # Uses classic, perturbed observations (Burgers'98)
        C = Y.T @ Y + R.full * N1
        D = mean0(hnoise.sample(N))
        YC = mrdiv(Y, C)
        KG = A.T @ YC
        HK = Y.T @ YC
        dE = (KG @ (y - D - Eo).T).T
        E = E + dE

    elif "Sqrt" in upd_a:
        # Uses a symmetric square root (ETKF)
        # to deterministically transform the ensemble.

        # The various versions below differ only numerically.
        # EVD is default, but for large N use SVD version.
        if upd_a == "Sqrt" and N > Nx:
            upd_a = "Sqrt svd"

        if "explicit" in upd_a:
            # Not recommended due to numerical costs and instability.
            # Implementation using inv (in ens space)
            Pw = sla.inv(Y @ R.inv @ Y.T + N1 * eye(N))
            T = sla.sqrtm(Pw) * sqrt(N1)
            HK = R.inv @ Y.T @ Pw @ Y
            # KG = R.inv @ Y.T @ Pw @ A
        elif "svd" in upd_a:
            # Implementation using svd of Y R^{-1/2}.
            V, s, _ = svd0(Y @ R.sym_sqrt_inv.T)
            d = pad0(s**2, N) + N1
            Pw = (V * d ** (-1.0)) @ V.T
            T = (V * d ** (-0.5)) @ V.T * sqrt(N1)
            # docs/images/snippets/trHK.jpg
            trHK = np.sum((s**2 + N1) ** (-1.0) * s**2)
        elif "sS" in upd_a:
            # Same as 'svd', but with slightly different notation
            # (sometimes used by Sakov) using the normalization sqrt(N1).
            S = Y @ R.sym_sqrt_inv.T / sqrt(N1)
            V, s, _ = svd0(S)
            d = pad0(s**2, N) + 1
            Pw = (V * d ** (-1.0)) @ V.T / N1  # = G/(N1)
            T = (V * d ** (-0.5)) @ V.T
            # docs/images/snippets/trHK.jpg
            trHK = np.sum((s**2 + 1) ** (-1.0) * s**2)
        else:  # 'eig' in upd_a:
            # Implementation using eig. val. decomp.
            d, V = sla.eigh(Y @ R.inv @ Y.T + N1 * eye(N))
            T = V @ diag(d ** (-0.5)) @ V.T * sqrt(N1)
            Pw = V @ diag(d ** (-1.0)) @ V.T
            HK = R.inv @ Y.T @ (V @ diag(d ** (-1)) @ V.T) @ Y
        w = dy @ R.inv @ Y.T @ Pw
        E = mu + w @ A + T @ A

    elif "DEnKF" == upd_a:
        # Uses "Deterministic EnKF" (sakov'08)
        C = Y.T @ Y + R.full * N1
        YC = mrdiv(Y, C)
        KG = A.T @ YC
        HK = Y.T @ YC
        E = E + KG @ dy - 0.5 * (KG @ Y.T).T

    else:
        raise KeyError("No analysis update method found: '" + upd_a + "'.")

    # Diagnostic: relative influence of observations
    if stats is not None:
        if "trHK" in locals():
            stats.trHK.a[ko] = trHK / hnoise.M  # type: ignore[reportPossiblyUnbound]
        elif "HK" in locals():
            stats.trHK.a[ko] = HK.trace() / hnoise.M  # type: ignore[reportPossiblyUnbound]

    # -------------------------------
    if log_transform:
        E = np.exp(np.clip(E, -50, 50))
    # --------------------------------

    return E


def post_process(E, infl: float, rot):
    """Inflate, Rotate.

    To avoid recomputing/recombining anomalies,
    this should have been inside `EnKF_analysis`

    But it is kept as a separate function

    - for readability;
    - to avoid inflating/rotationg smoothed states (for the `EnKS`).
    """
    do_infl = infl != 1.0 and infl != "-N"

    if do_infl or rot:
        A, mu = center(E)
        N, Nx = E.shape
        T = eye(N)

        if do_infl:
            T = infl * T

        if rot:
            T = genOG_1(N, rot) @ T

        E = mu + T @ A
    return E


def add_noise(E, dt, noise, method):
    """Treatment of additive noise for ensembles.

    Refs: [raanes2014][]
    """
    if noise.C == 0:
        return E

    N, Nx = E.shape
    A, mu = center(E)
    Q12 = noise.C.Left
    Q = noise.C.full

    def sqrt_core():
        T = np.nan  # cause error if used
        Qa12 = np.nan  # cause error if used
        A2 = A.copy()  # Instead of using (the implicitly nonlocal) A,
        # which changes A outside as well. NB: This is a bug in Datum!
        if N <= Nx:
            Ainv = tinv(A2.T)
            Qa12 = Ainv @ Q12
            T = funm_psd(eye(N) + dt * (N - 1) * (Qa12 @ Qa12.T), sqrt)
            A2 = T @ A2
        else:  # "Left-multiplying" form
            P = A2.T @ A2 / (N - 1)
            L = funm_psd(eye(Nx) + dt * mrdiv(Q, P), sqrt)
            A2 = A2 @ L.T
        E = mu + A2
        return E, T, Qa12

    if method == "Stoch":
        # In-place addition works (also) for empty [] noise sample.
        E += sqrt(dt) * noise.sample(N)

    elif method == "none":
        pass

    elif method == "Mult-1":
        varE = np.var(E, axis=0, ddof=1).sum()
        ratio = (varE + dt * diag(Q).sum()) / varE
        E = mu + sqrt(ratio) * A
        E = svdi(*tsvd(E, 0.999))  # Explained in Datum

    elif method == "Mult-M":
        varE = np.var(E, axis=0)
        ratios = sqrt((varE + dt * diag(Q)) / varE)
        E = mu + A * ratios
        E = svdi(*tsvd(E, 0.999))  # Explained in Datum

    elif method == "Sqrt-Core":
        E = sqrt_core()[0]

    elif method == "Sqrt-Mult-1":
        varE0 = np.var(E, axis=0, ddof=1).sum()
        varE2 = varE0 + dt * diag(Q).sum()
        E, _, Qa12 = sqrt_core()
        if N <= Nx:
            A, mu = center(E)
            varE1 = np.var(E, axis=0, ddof=1).sum()
            ratio = varE2 / varE1
            E = mu + sqrt(ratio) * A
            E = svdi(*tsvd(E, 0.999))  # Explained in Datum

    elif method == "Sqrt-Add-Z":
        E, _, Qa12 = sqrt_core()
        if N <= Nx:
            Z = Q12 - A.T @ Qa12
            E += sqrt(dt) * (Z @ rng.standard_normal((Z.shape[1], N))).T

    elif method == "Sqrt-Dep":
        E, T, Qa12 = sqrt_core()
        if N <= Nx:
            # Q_hat12: reuse svd for both inversion and projection.
            Q_hat12 = A.T @ Qa12
            U, s, VT = tsvd(Q_hat12, 0.99)
            Q_hat12_inv = (VT.T * s ** (-1.0)) @ U.T
            Q_hat12_proj = VT.T @ VT
            rQ = Q12.shape[1]
            # Calc D_til
            Z = Q12 - Q_hat12
            D_hat = A.T @ (T - eye(N))
            Xi_hat = Q_hat12_inv @ D_hat
            Xi_til = (eye(rQ) - Q_hat12_proj) @ rng.standard_normal((rQ, N))
            D_til = Z @ (Xi_hat + sqrt(dt) * Xi_til)
            E += D_til.T

    else:
        raise KeyError("No such method")

    return E


