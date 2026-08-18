"""The EnKF and other ensemble-based methods."""

from turtle import clear
import numpy as np
import numpy.random as rnd
import scipy.linalg as sla
from numpy import diag, eye, sqrt, zeros
from scipy.linalg.basic import inv
from scipy.stats import ortho_group
import matplotlib.pyplot as plt
import statistics
import seaborn as sns; sns.set()
import pandas as pd
import h5py
import random
from math import pi, exp, log

import dapper.tools.multiproc as mp
from dapper.stats import center, mean0
from dapper.tools.linalg import mldiv, mrdiv, pad0, svd0, svdi, tinv, tsvd
from dapper.tools.matrices import funm_psd, genOG_1,genOG
from dapper.tools.progressbar import progbar

from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, MaxAbsScaler
from skimage import measure

from . import da_method


@da_method
class ens_method:
    """Declare default ensemble arguments."""

    infl: float        = 1.0
    rot: bool          = False
    fnoise_treatm: str = 'Stoch'


@ens_method
class EnKF:
    """The ensemble Kalman filter.

    Refs: `bib.evensen2009ensemble`.
    """

    upd_a: str
    N: int

    def assimilate(self, HMM, xx, yy):
        Dyn, Obs, chrono, X0, stats = \
            HMM.Dyn, HMM.Obs, HMM.tseq, HMM.X0, self.stats

        # Init
        E = X0.sample(self.N)
        stats.assess(0, E=E)
        cycle=0
	  
        # Loop
        for k, kObs, t, dt in progbar(chrono.ticker):
            E = Dyn(E, t-dt, dt)
            E = add_noise(E, dt, Dyn.noise, self.fnoise_treatm)
            
            # Analysis update
            if kObs is not None:
                cycle+=1
                stats.assess(k, kObs, 'f', E=E)
                #print(E)
                E = EnKF_analysis(E, Obs(E, t), Obs.noise,
                                  yy[kObs], self.upd_a, stats, kObs)
                E = post_process(E, self.infl, self.rot)

            stats.assess(k, kObs, E=E)
            #if cycle in [6,18,30,46,60,100] :
                #print(self.stats.err.rms.a[cycle-1])


def EnKF_analysis(E, Eo, hnoise, y, upd_a, stats, kObs):
    """Perform the EnKF analysis update.

    This implementation includes several flavours and forms,
        specified by `upd_a`.

    Main references: `bib.sakov2008deterministic`,
    `bib.sakov2008implications`, `bib.hoteit2015mitigating`
    """
    R     = hnoise.C     # Obs noise cov
    N, Nx = E.shape      # Dimensionality
    N1    = N-1          # Ens size - 1

    mu = np.mean(E, 0)   # Ens mean
    A  = E - mu          # Ens anomalies

    xo = np.mean(Eo, 0)  # Obs ens mean
    Y  = Eo-xo           # Obs ens anomalies
    dy = y - xo          # Mean "innovation"

    if 'PertObs' in upd_a:
        # Uses classic, perturbed observations (Burgers'98)
        #print(mu)
        #print(np.cov(E))
        C  = Y.T @ Y + R.full*N1
        D  = mean0(hnoise.sample(N))
        YC = mrdiv(Y, C)
        KG = A.T @ YC
        HK = Y.T @ YC
        dE = (KG @ (y - D - Eo).T).T
        E  = E + dE

    elif 'Sqrt' in upd_a:
        # Uses a symmetric square root (ETKF)
        # to deterministically transform the ensemble.

        # The various versions below differ only numerically.
        # EVD is default, but for large N use SVD version.
        if upd_a == 'Sqrt' and N > Nx:
            upd_a = 'Sqrt svd'

        if 'explicit' in upd_a:
            # Not recommended due to numerical costs and instability.
            # Implementation using inv (in ens space)
            Pw = sla.inv(Y @ R.inv @ Y.T + N1*eye(N))
            T  = sla.sqrtm(Pw) * sqrt(N1)
            HK = R.inv @ Y.T @ Pw @ Y
            # KG = R.inv @ Y.T @ Pw @ A
        elif 'svd' in upd_a:
            # Implementation using svd of Y R^{-1/2}.
            V, s, _ = svd0(Y @ R.sym_sqrt_inv.T)
            d       = pad0(s**2, N) + N1
            Pw      = (V * d**(-1.0)) @ V.T
            T       = (V * d**(-0.5)) @ V.T * sqrt(N1)
            # docs/snippets/trHK.jpg
            trHK    = np.sum((s**2+N1)**(-1.0) * s**2)
        elif 'sS' in upd_a:
            # Same as 'svd', but with slightly different notation
            # (sometimes used by Sakov) using the normalization sqrt(N1).
            S       = Y @ R.sym_sqrt_inv.T / sqrt(N1)
            V, s, _ = svd0(S)
            d       = pad0(s**2, N) + 1
            Pw      = (V * d**(-1.0))@V.T / N1  # = G/(N1)
            T       = (V * d**(-0.5))@V.T
            # docs/snippets/trHK.jpg
            trHK    = np.sum((s**2 + 1)**(-1.0)*s**2)
        else:  # 'eig' in upd_a:
            # Implementation using eig. val. decomp.
            d, V   = sla.eigh(Y @ R.inv @ Y.T + N1*eye(N))
            T      = V@diag(d**(-0.5))@V.T * sqrt(N1)
            Pw     = V@diag(d**(-1.0))@V.T
            HK     = R.inv @ Y.T @ (V @ diag(d**(-1)) @ V.T) @ Y
        w = dy @ R.inv @ Y.T @ Pw
        E = mu + w@A + T@A

    elif 'Serial' in upd_a:
        # Observations assimilated one-at-a-time:
        inds = serial_inds(upd_a, y, R, A)
        #  Requires de-correlation:
        dy   = dy @ R.sym_sqrt_inv.T
        Y    = Y  @ R.sym_sqrt_inv.T
        # Enhancement in the nonlinear case:
        # re-compute Y each scalar obs assim.
        # But: little benefit, model costly (?),
        # updates cannot be accumulated on S and T.

        if any(x in upd_a for x in ['Stoch', 'ESOPS', 'Var1']):
            # More details: Misc/Serial_ESOPS.py.
            for i, j in enumerate(inds):

                # Perturbation creation
                if 'ESOPS' in upd_a:
                    # "2nd-O exact perturbation sampling"
                    if i == 0:
                        # Init -- increase nullspace by 1
                        V, s, UT = svd0(A)
                        s[N-2:] = 0
                        A = svdi(V, s, UT)
                        v = V[:, N-2]
                    else:
                        # Orthogonalize v wrt. the new A
                        #
                        # v = Zj - Yj (from paper) requires Y==HX.
                        # Instead: mult` should be c*ones(Nx) so we can
                        # project v into ker(A) such that v@A is null.
                        mult  = (v@A) / (Yj@A) # noqa
                        v     = v - mult[0]*Yj # noqa
                        v    /= sqrt(v@v)
                    Zj  = v*sqrt(N1)  # Standardized perturbation along v
                    Zj *= np.sign(rnd.rand()-0.5)  # Random sign
                else:
                    # The usual stochastic perturbations.
                    Zj = mean0(rnd.randn(N))  # Un-coloured noise
                    if 'Var1' in upd_a:
                        Zj *= sqrt(N/(Zj@Zj))

                # Select j-th obs
                Yj  = Y[:, j]       # [j] obs anomalies
                dyj = dy[j]         # [j] innov mean
                DYj = Zj - Yj       # [j] innov anomalies
                DYj = DYj[:, None]  # Make 2d vertical

                # Kalman gain computation
                C     = Yj@Yj + N1  # Total obs cov
                KGx   = Yj @ A / C  # KG to update state
                KGy   = Yj @ Y / C  # KG to update obs

                # Updates
                A    += DYj * KGx
                mu   += dyj * KGx
                Y    += DYj * KGy
                dy   -= dyj * KGy
            E = mu + A
        else:
            # "Potter scheme", "EnSRF"
            # - EAKF's two-stage "update-regress" form yields
            #   the same *ensemble* as this.
            # - The form below may be derived as "serial ETKF",
            #   but does not yield the same
            #   ensemble as 'Sqrt' (which processes obs as a batch)
            #   -- only the same mean/cov.
            T = eye(N)
            for j in inds:
                Yj = Y[:, j]
                C  = Yj@Yj + N1
                Tj = np.outer(Yj, Yj / (C + sqrt(N1*C)))
                T -= Tj @ T
                Y -= Tj @ Y
            w = dy@Y.T@T/N1
            E = mu + w@A + T@A

    elif 'DEnKF' == upd_a:
        # Uses "Deterministic EnKF" (sakov'08)
        C  = Y.T @ Y + R.full*N1
        YC = mrdiv(Y, C)
        KG = A.T @ YC
        HK = Y.T @ YC
        E  = E + KG@dy - 0.5*(KG@Y.T).T
    elif 'ETKF Eho' == upd_a:
        R_tilde = hnoise.C.sym_sqrt_inv
        Np = Y.shape[1]
        H_bar = Y@R_tilde
        d_bar = R_tilde@dy
        alpha_H =np.linalg.solve((N1*np.eye(Np)+np.transpose(H_bar)@H_bar),d_bar)
        xa=mu+A.T@H_bar@alpha_H
	
        V, s, _ = svd0(Y @ R.sym_sqrt_inv.T)
        d       = pad0(s**2, N) + N1
        T       = (V * d**(-0.5)) @ V.T 
        E = xa+sqrt(N1)*T@A
    elif 'ETKF Qe' == upd_a:
        R_tilde = hnoise.C.sym_sqrt_inv
        Np = Y.shape[1]
        H_bar = Y@R_tilde
        d_bar = R_tilde@dy
        alpha_H =np.linalg.solve((N1*np.eye(Np)+np.transpose(H_bar)@H_bar),d_bar)
        xa=mu+A.T@H_bar@alpha_H
	
        eps=1
        teta = sqrt(N)/(sqrt(N)-eps)
        Q_eps = -teta/N*np.ones([N,N])
        Q_eps[0,:] = eps/sqrt(N)
        Q_eps[:,0] = eps/sqrt(N)
        for j in range(1,N):
            Q_eps[j,j] = 1-teta/N
	 
        W = np.zeros([N,N])
	
        V, s, _ = svd0(Y @ R.sym_sqrt_inv.T)
        V2=V.copy()
        rgV=min(N1,len(s))
        
        V=V[:,:rgV]
	
        d       = pad0(s**2, rgV) + rgV
       
        W[N-rgV:,:]=np.transpose(d**(-0.5)*V) 
        #print("******")
        #print(Q_eps@W@A)
        #print(V2@W@A)
        E = xa+sqrt(N1)*Q_eps@W@A
    else:
        raise KeyError("No analysis update method found: '" + upd_a + "'.")

    # Diagnostic: relative influence of observations
    if 'trHK' in locals():
        stats.trHK[kObs] = trHK      / hnoise.M
    elif 'HK' in locals():
        stats.trHK[kObs] = HK.trace()/hnoise.M

    return E


def post_process(E, infl, rot):
    """Inflate, Rotate.

    To avoid recomputing/recombining anomalies,
    this should have been inside :func:`EnKF_analysis`

    But it is kept as a separate function

    - for readability;
    - to avoid inflating/rotationg smoothed states (for the :func:`EnKS`).
    """
    do_infl = infl != 1.0 and infl != '-N'

    if do_infl or rot:
        A, mu  = center(E)
        N, Nx  = E.shape
        T      = eye(N)

        if do_infl:
            T = infl * T

        if rot:
            T = genOG_1(N, rot) @ T

        E = mu + T@A
    return E


def add_noise(E, dt, noise, method):
    """Treatment of additive noise for ensembles.

    Refs: `bib.raanes2014ext`
    """
    if noise.C == 0:
        return E

    N, Nx = E.shape
    A, mu = center(E)
    Q12   = noise.C.Left
    Q     = noise.C.full

    def sqrt_core():
        T    = np.nan    # cause error if used
        Qa12 = np.nan    # cause error if used
        A2   = A.copy()  # Instead of using (the implicitly nonlocal) A,
        # which changes A outside as well. NB: This is a bug in Datum!
        if N <= Nx:
            Ainv = tinv(A2.T)
            Qa12 = Ainv@Q12
            T    = funm_psd(eye(N) + dt*(N-1)*(Qa12@Qa12.T), sqrt)
            A2   = T@A2
        else:  # "Left-multiplying" form
            P  = A2.T @ A2 / (N-1)
            L  = funm_psd(eye(Nx) + dt*mrdiv(Q, P), sqrt)
            A2 = A2 @ L.T
        E = mu + A2
        return E, T, Qa12

    if method == 'Stoch':
        # In-place addition works (also) for empty [] noise sample.
        E += sqrt(dt)*noise.sample(N)

    elif method == 'none':
        pass

    elif method == 'Mult-1':
        varE   = np.var(E, axis=0, ddof=1).sum()
        ratio  = (varE + dt*diag(Q).sum())/varE
        E      = mu + sqrt(ratio)*A
        E      = svdi(*tsvd(E, 0.999))  # Explained in Datum

    elif method == 'Mult-M':
        varE   = np.var(E, axis=0)
        ratios = sqrt((varE + dt*diag(Q))/varE)
        E      = mu + A*ratios
        E      = svdi(*tsvd(E, 0.999))  # Explained in Datum

    elif method == 'Sqrt-Core':
        E = sqrt_core()[0]

    elif method == 'Sqrt-Mult-1':
        varE0 = np.var(E, axis=0, ddof=1).sum()
        varE2 = (varE0 + dt*diag(Q).sum())
        E, _, Qa12 = sqrt_core()
        if N <= Nx:
            A, mu   = center(E)
            varE1   = np.var(E, axis=0, ddof=1).sum()
            ratio   = varE2/varE1
            E       = mu + sqrt(ratio)*A
            E       = svdi(*tsvd(E, 0.999))  # Explained in Datum

    elif method == 'Sqrt-Add-Z':
        E, _, Qa12 = sqrt_core()
        if N <= Nx:
            Z  = Q12 - A.T@Qa12
            E += sqrt(dt)*(Z@rnd.randn(Z.shape[1], N)).T

    elif method == 'Sqrt-Dep':
        E, T, Qa12 = sqrt_core()
        if N <= Nx:
            # Q_hat12: reuse svd for both inversion and projection.
            Q_hat12      = A.T @ Qa12
            U, s, VT     = tsvd(Q_hat12, 0.99)
            Q_hat12_inv  = (VT.T * s**(-1.0)) @ U.T
            Q_hat12_proj = VT.T@VT
            rQ = Q12.shape[1]
            # Calc D_til
            Z      = Q12 - Q_hat12
            D_hat  = A.T@(T-eye(N))
            Xi_hat = Q_hat12_inv @ D_hat
            Xi_til = (eye(rQ) - Q_hat12_proj)@rnd.randn(rQ, N)
            D_til  = Z@(Xi_hat + sqrt(dt)*Xi_til)
            E     += D_til.T

    else:
        raise KeyError('No such method')

    return E


@ens_method
class EnKS:
    """The ensemble Kalman smoother.

    Refs: `bib.evensen2009ensemble`

    The only difference to the EnKF
    is the management of the lag and the reshapings.
    """

    upd_a: str
    N: int
    Lag: int

    # Reshapings used in smoothers to go to/from
    # 3D arrays, where the 0th axis is the Lag index.
    def reshape_to(self, E):
        K, N, Nx = E.shape
        return E.transpose([1, 0, 2]).reshape((N, K*Nx))

    def reshape_fr(self, E, Nx):
        N, Km = E.shape
        K    = Km//Nx
        return E.reshape((N, K, Nx)).transpose([1, 0, 2])

    def assimilate(self, HMM, xx, yy):
        Dyn, Obs, chrono, X0, stats = \
            HMM.Dyn, HMM.Obs, HMM.tseq, HMM.X0, self.stats

        # Inefficient version, storing full time series ensemble.
        # See iEnKS for a "rolling" version.
        E    = zeros((chrono.K+1, self.N, Dyn.M))
        E[0] = X0.sample(self.N)

        for k, kObs, t, dt in progbar(chrono.ticker):
            E[k] = Dyn(E[k-1], t-dt, dt)
            E[k] = add_noise(E[k], dt, Dyn.noise, self.fnoise_treatm)

            if kObs is not None:
                stats.assess(k, kObs, 'f', E=E[k])

                Eo    = Obs(E[k], t)
                y     = yy[kObs]

                # Inds within Lag
                kk    = range(max(0, k-self.Lag*chrono.dkObs), k+1)

                EE    = E[kk]

                EE    = self.reshape_to(EE)
                EE    = EnKF_analysis(EE, Eo, Obs.noise, y, self.upd_a, stats, kObs)
                E[kk] = self.reshape_fr(EE, Dyn.M)
                E[k]  = post_process(E[k], self.infl, self.rot)
                stats.assess(k, kObs, 'a', E=E[k])

        for k, kObs, _, _ in progbar(chrono.ticker, desc='Assessing'):
            stats.assess(k, kObs, 'u', E=E[k])
            if kObs is not None:
                stats.assess(k, kObs, 's', E=E[k])


@ens_method
class EnRTS:
    """EnRTS (Rauch-Tung-Striebel) smoother.

    Refs: `bib.raanes2016thesis`
    """

    upd_a: str
    N: int
    cntr: float

    def assimilate(self, HMM, xx, yy):
        Dyn, Obs, chrono, X0, stats = \
            HMM.Dyn, HMM.Obs, HMM.tseq, HMM.X0, self.stats

        E    = zeros((chrono.K+1, self.N, Dyn.M))
        Ef   = E.copy()
        E[0] = X0.sample(self.N)

        # Forward pass
        for k, kObs, t, dt in progbar(chrono.ticker):
            E[k]  = Dyn(E[k-1], t-dt, dt)
            E[k]  = add_noise(E[k], dt, Dyn.noise, self.fnoise_treatm)
            Ef[k] = E[k]

            if kObs is not None:
                stats.assess(k, kObs, 'f', E=E[k])
                Eo   = Obs(E[k], t)
                y    = yy[kObs]
                E[k] = EnKF_analysis(E[k], Eo, Obs.noise, y, self.upd_a, stats, kObs)
                E[k] = post_process(E[k], self.infl, self.rot)
                stats.assess(k, kObs, 'a', E=E[k])

        # Backward pass
        for k in progbar(range(chrono.K)[::-1]):
            A  = center(E[k])[0]
            Af = center(Ef[k+1])[0]

            J = tinv(Af) @ A
            J *= self.cntr

            E[k] += (E[k+1] - Ef[k+1]) @ J

        for k, kObs, _, _ in progbar(chrono.ticker, desc='Assessing'):
            stats.assess(k, kObs, 'u', E=E[k])
            if kObs is not None:
                stats.assess(k, kObs, 's', E=E[k])


def serial_inds(upd_a, y, cvR, A):
    """Get the indices used for serial updating.

    - Default: random ordering
    - if "mono" in `upd_a`: `1, 2, ..., len(y)`
    - if "sorted" in `upd_a`: sort by variance
    """
    if 'mono' in upd_a:
        # Not robust?
        inds = np.arange(len(y))
    elif 'sorted' in upd_a:
        N = len(A)
        dC = cvR.diag
        if np.all(dC == dC[0]):
            # Sort y by P
            dC = np.sum(A*A, 0)/(N-1)
        inds = np.argsort(dC)
    else:  # Default: random ordering
        inds = rnd.permutation(len(y))
    return inds


@ens_method
class SL_EAKF:
    """Serial, covariance-localized EAKF.

    Refs: `bib.karspeck2007experimental`.

    Used without localization, this should be equivalent (full ensemble equality)
    to the `EnKF` with `upd_a='Serial'`.
    """

    N: int
    loc_rad: float
    taper: str  = 'GC'
    ordr: str   = 'rand'

    def assimilate(self, HMM, xx, yy):
        Dyn, Obs, chrono, X0, stats = HMM.Dyn, HMM.Obs, HMM.tseq, HMM.X0, self.stats

        N1   = self.N-1
        R    = Obs.noise
        Rm12 = Obs.noise.C.sym_sqrt_inv

        E = X0.sample(self.N)
        stats.assess(0, E=E)

        for k, kObs, t, dt in progbar(chrono.ticker):
            E = Dyn(E, t-dt, dt)
            E = add_noise(E, dt, Dyn.noise, self.fnoise_treatm)

            if kObs is not None:
                stats.assess(k, kObs, 'f', E=E)
                y    = yy[kObs]
                inds = serial_inds(self.ordr, y, R, center(E)[0])

                state_taperer = Obs.localizer(self.loc_rad, 'y2x', t, self.taper)
                for j in inds:
                    # Prep:
                    # ------------------------------------------------------
                    Eo = Obs(E, t)
                    xo = np.mean(Eo, 0)
                    Y  = Eo - xo
                    mu = np.mean(E, 0)
                    A  = E-mu
                    # Update j-th component of observed ensemble:
                    # ------------------------------------------------------
                    Y_j    = Rm12[j, :] @ Y.T
                    dy_j   = Rm12[j, :] @ (y - xo)
                    # Prior var * N1:
                    sig2_j = Y_j@Y_j
                    if sig2_j < 1e-9:
                        continue
                    # Update (below, we drop the locality subscript: _j)
                    sig2_u = 1/(1/sig2_j + 1/N1)      # Postr. var * N1
                    alpha  = (N1/(N1+sig2_j))**(0.5)  # Update contraction factor
                    dy2    = sig2_u * dy_j/N1         # Mean update
                    Y2     = alpha*Y_j                # Anomaly update
                    # Update state (regress update from obs space, using localization)
                    # ------------------------------------------------------
                    ii, tapering = state_taperer(j)
                    # ii, tapering = ..., 1  # cancel localization
                    if len(ii) == 0:
                        continue
                    Xi = A[:, ii]*tapering
                    Regression = Xi.T @ Y_j/np.sum(Y_j**2)
                    mu[ii] += Regression*dy2
                    A[:, ii] += np.outer(Y2 - Y_j, Regression)
                    E = mu + A

                E = post_process(E, self.infl, self.rot)

            stats.assess(k, kObs, E=E)


@ens_method
class LETKF:
    """Same as EnKF (sqrt), but with localization.

    Refs: `bib.hunt2007efficient`.

    NB: Multiproc. yields slow-down for `dapper.mods.Lorenz96`,
    even with `batch_size=(1,)`. But for `dapper.mods.QG`
    (`batch_size=(2,2)` or less) it is quicker.

    NB: If `len(ii)` is small, analysis may be slowed-down with '-N' infl.
    """

    N: int
    loc_rad: float
    taper: str = 'GC'
    xN: float  = 1.0
    g: int     = 0
    mp: bool   = False

    def assimilate(self, HMM, xx, yy):
        Dyn, Obs, chrono, X0, stats, N = \
            HMM.Dyn, HMM.Obs, HMM.tseq, HMM.X0, self.stats, self.N
        R, N1 = HMM.Obs.noise.C, N-1

        _map = mp.map if self.mp else map

        E = X0.sample(N)
        
        stats.assess(0, E=E)
        

        for k, kObs, t, dt in progbar(chrono.ticker):
            # Forecast
            E = Dyn(E, t-dt, dt)
            E = add_noise(E, dt, Dyn.noise, self.fnoise_treatm)
            
            if kObs is not None:
                stats.assess(k, kObs, 'f', E=E)
		
                # Decompose ensmeble
                mu = np.mean(E, 0)
                A  = E - mu
                # Obs space variables
                y     = yy[kObs]
                Y, xo = center(Obs(E, t))
                # Transform obs space
                Y  = Y        @ R.sym_sqrt_inv.T
                dy = (y - xo) @ R.sym_sqrt_inv.T

                # Local analyses
                # Get localization configuration
                state_batches, obs_taperer = \
                    Obs.localizer(self.loc_rad, 'x2y', t, self.taper)
                # Avoid pickling self
                xN, g, infl = self.xN, self.g, self.infl

                def local_analysis(ii):
                    """Do the local analysis.

                    Notation:

                    - ii: inds for the state batch defining the locality
                    - jj: inds for the associated obs
                    """
                    # Locate local obs
                    jj, tapering = obs_taperer(ii)
                    if len(jj) == 0:
                        return E[:, ii], N1  # no update
                    Y_jj   = Y[:, jj]
                    dy_jj  = dy[jj]

                    # Adaptive inflation
                    za = effective_N(Y_jj, dy_jj, xN, g) if infl == '-N' else N1

                    # Taper
                    Y_jj  *= sqrt(tapering)
                    dy_jj *= sqrt(tapering)

                    # Compute ETKF update
                    if len(jj) < N:
                        # SVD version
                        V, sd, _ = svd0(Y_jj)
                        d      = pad0(sd**2, N) + za
                        Pw     = (V * d**(-1.0)) @ V.T
                        T      = (V * d**(-0.5)) @ V.T * sqrt(za)
                    else:
                        # EVD version
                        d, V   = sla.eigh(Y_jj@Y_jj.T + za*eye(N))
                        T     = V@diag(d**(-0.5))@V.T * sqrt(za)
                        Pw    = V@diag(d**(-1.0))@V.T
                    AT  = T @ A[:, ii]
                    dmu = dy_jj @ Y_jj.T @ Pw @ A[:, ii]
                    Eii = mu[ii] + dmu + AT
                    #print(np.shape(Eii))
                    return Eii, za

                # Run local analyses
                EE, za = zip(*_map(local_analysis, state_batches))
                for ii, Eii in zip(state_batches, EE):
                    E[:, ii] = Eii

                # Global post-processing
                E = post_process(E, self.infl, self.rot)

                stats.infl[kObs] = sqrt(N1/np.mean(za))

            stats.assess(k, kObs, E=E)


def effective_N(YR, dyR, xN, g):
    """Effective ensemble size N.

    As measured by the finite-size EnKF-N
    """
    N, Ny = YR.shape
    N1   = N-1

    V, s, UT = svd0(YR)
    du     = UT @ dyR

    eN, cL = hyperprior_coeffs(s, N, xN, g)

    def pad_rk(arr): return pad0(arr, min(N, Ny))
    def dgn_rk(l1): return pad_rk((l1*s)**2) + N1

    # Make dual cost function (in terms of l1)
    def J(l1):
        val = np.sum(du**2/dgn_rk(l1)) \
            + eN/l1**2 \
            + cL*np.log(l1**2)
        return val

    # Derivatives (not required with minimize_scalar):
    def Jp(l1):
        val = -2*l1   * np.sum(pad_rk(s**2) * du**2/dgn_rk(l1)**2) \
            + -2*eN/l1**3 \
            + 2*cL/l1
        return val

    def Jpp(l1):
        val = 8*l1**2 * np.sum(pad_rk(s**4) * du**2/dgn_rk(l1)**3) \
            + 6*eN/l1**4 \
            + -2*cL/l1**2
        return val

    # Find inflation factor (optimize)
    l1 = Newton_m(Jp, Jpp, 1.0)
    # l1 = fmin_bfgs(J, x0=[1], gtol=1e-4, disp=0)
    # l1 = minimize_scalar(J, bracket=(sqrt(prior_mode), 1e2), tol=1e-4).x

    za = N1/l1**2
    return za


# Notes on optimizers for the 'dual' EnKF-N:
# ----------------------------------------
#  Using minimize_scalar:
#  - Doesn't take dJdx. Advantage: only need J
#  - method='bounded' not necessary and slower than 'brent'.
#  - bracket not necessary either...
#  Using multivariate minimization: fmin_cg, fmin_bfgs, fmin_ncg
#  - these also accept dJdx. But only fmin_bfgs approaches
#    the speed of the scalar minimizers.
#  Using scalar root-finders:
#  - brenth(dJ1, LowB, 1e2,     xtol=1e-6) # Same speed as minimization
#  - newton(dJ1,1.0, fprime=dJ2, tol=1e-6) # No improvement
#  - newton(dJ1,1.0, fprime=dJ2, tol=1e-6, fprime2=dJ3) # No improvement
#  - Newton_m(dJ1,dJ2, 1.0) # Significantly faster. Also slightly better CV?
# => Despite inconvienience of defining analytic derivatives,
#    Newton_m seems like the best option.
#  - In extreme (or just non-linear Obs.mod) cases,
#    the EnKF-N cost function may have multiple minima.
#    Then: should use more robust optimizer!
#
# For 'primal'
# ----------------------------------------
# Similarly, Newton_m seems like the best option,
# although alternatives are provided (commented out).
#
def Newton_m(fun, deriv, x0, is_inverted=False,
             conf=1.0, xtol=1e-4, ytol=1e-7, itermax=10**2):
    """Find root of `fun`.

    This is a simple (and pretty fast) implementation of Newton's method.
    """
    itr, dx, Jx = 0, np.inf, fun(x0)
    def norm(x): return sqrt(np.sum(x**2))
    while ytol < norm(Jx) and xtol < norm(dx) and itr < itermax:
        Dx  = deriv(x0)
        if is_inverted:
            dx  = Dx @ Jx
        elif isinstance(Dx, float):
            dx  = Jx/Dx
        else:
            dx  = mldiv(Dx, Jx)
        dx *= conf
        x0 -= dx
        Jx  = fun(x0)
    return x0


def hyperprior_coeffs(s, N, xN=1, g=0):
    r"""Set EnKF-N inflation hyperparams.

    The EnKF-N prior may be specified by the constants:

    - eN: Effect of unknown mean
    - cL: Coeff in front of log term

    These are trivial constants in the original EnKF-N,
    but are further adjusted (corrected and tuned) for the following reasons.

    - Reason 1: mode correction.
      These parameters bridge the Jeffreys (`xN=1`) and Dirac (`xN=Inf`) hyperpriors
      for the prior covariance, B, as discussed in `bib.bocquet2015expanding`.
      Indeed, mode correction becomes necessary when $$ R \rightarrow \infty $$
      because then there should be no ensemble update (and also no inflation!).
      More specifically, the mode of `l1`'s should be adjusted towards 1
      as a function of $$ I - K H $$ ("prior's weight").
      PS: why do we leave the prior mode below 1 at all?
      Because it sets up "tension" (negative feedback) in the inflation cycle:
      the prior pulls downwards, while the likelihood tends to pull upwards.

    - Reason 2: Boosting the inflation prior's certainty from N to xN*N.
      The aim is to take advantage of the fact that the ensemble may not
      have quite as much sampling error as a fully stochastic sample,
      as illustrated in section 2.1 of `bib.raanes2019adaptive`.

    - Its damping effect is similar to work done by J. Anderson.

    The tuning is controlled by:

    - `xN=1`: is fully agnostic, i.e. assumes the ensemble is generated
      from a highly chaotic or stochastic model.
    - `xN>1`: increases the certainty of the hyper-prior,
      which is appropriate for more linear and deterministic systems.
    - `xN<1`: yields a more (than 'fully') agnostic hyper-prior,
      as if N were smaller than it truly is.
    - `xN<=0` is not meaningful.
    """
    N1 = N-1

    eN = (N+1)/N
    cL = (N+g)/N1

    # Mode correction (almost) as in eqn 36 of `bib.bocquet2015expanding`
    prior_mode = eN/cL                        # Mode of l1 (before correction)
    diagonal   = pad0(s**2, N) + N1           # diag of Y@R.inv@Y + N1*I
    #                                           (Hessian of J)
    I_KH       = np.mean(diagonal**(-1))*N1   # ≈ 1/(1 + HBH/R)
    # I_KH      = 1/(1 + (s**2).sum()/N1)     # Scalar alternative: use tr(HBH/R).
    mc         = sqrt(prior_mode**I_KH)       # Correction coeff

    # Apply correction
    eN /= mc
    cL *= mc

    # Boost by xN
    eN *= xN
    cL *= xN

    return eN, cL


def zeta_a(eN, cL, w):
    """EnKF-N inflation estimation via w.

    Returns `zeta_a = (N-1)/pre-inflation^2`.

    Using this inside an iterative minimization as in the
    `dapper.da_methods.variational.iEnKS` effectively blends
    the distinction between the primal and dual EnKF-N.
    """
    N  = len(w)
    N1 = N-1
    za = N1*cL/(eN + w@w)
    return za


@ens_method
class EnKF_N:
    """Finite-size EnKF (EnKF-N).

    Refs: `bib.bocquet2011ensemble`, `bib.bocquet2015expanding`

    This implementation is pedagogical, prioritizing the "dual" form.
    In consequence, the efficiency of the "primal" form suffers a bit.
    The primal form is included for completeness and to demonstrate equivalence.
    In `dapper.da_methods.variational.iEnKS`, however,
    the primal form is preferred because it
    already does optimization for w (as treatment for nonlinear models).

    `infl` should be unnecessary (assuming no model error, or that Q is correct).

    `Hess`: use non-approx Hessian for ensemble transform matrix?

    `g` is the nullity of A (state anomalies's), ie. g=max(1,N-Nx),
    compensating for the redundancy in the space of w.
    But we have made it an input argument instead, with default 0,
    because mode-finding (of p(x) via the dual) completely ignores this redundancy,
    and the mode gets (undesireably) modified by g.

    `xN` allows tuning the hyper-prior for the inflation.
    Usually, I just try setting it to 1 (default), or 2.
    Further description in hyperprior_coeffs().
    """

    N: int
    dual: bool = False
    Hess: bool = False
    xN: float  = 1.0
    g: int     = 0

    def assimilate(self, HMM, xx, yy):
        # Unpack
        Dyn, Obs, chrono, X0, stats = \
            HMM.Dyn, HMM.Obs, HMM.tseq, HMM.X0, self.stats
        R, N, N1 = HMM.Obs.noise.C, self.N, self.N-1

        # Init
        E = X0.sample(N)
        stats.assess(0, E=E)

        # Loop
        for k, kObs, t, dt in progbar(chrono.ticker):
            # Forecast
            E = Dyn(E, t-dt, dt)
            E = add_noise(E, dt, Dyn.noise, self.fnoise_treatm)

            # Analysis
            if kObs is not None:
                stats.assess(k, kObs, 'f', E=E)
                Eo = Obs(E, t)
                y  = yy[kObs]

                mu = np.mean(E, 0)
                A  = E - mu

                xo = np.mean(Eo, 0)
                Y  = Eo-xo
                dy = y - xo

                V, s, UT = svd0(Y @ R.sym_sqrt_inv.T)
                du       = UT @ (dy @ R.sym_sqrt_inv.T)
                def dgn_N(l1): return pad0((l1*s)**2, N) + N1

                # Adjust hyper-prior
                # xN_ = noise_level(self.xN,stats,chrono,N1,kObs,A,
                #                   locals().get('A_old',None))
                eN, cL = hyperprior_coeffs(s, N, self.xN, self.g)

                if self.dual:
                    # Make dual cost function (in terms of l1)
                    def pad_rk(arr): return pad0(arr, min(N, Obs.M))
                    def dgn_rk(l1): return pad_rk((l1*s)**2) + N1

                    def J(l1):
                        val = np.sum(du**2/dgn_rk(l1)) \
                            + eN/l1**2 \
                            + cL*np.log(l1**2)
                        return val

                    # Derivatives (not required with minimize_scalar):
                    def Jp(l1):
                        val = -2*l1 * np.sum(pad_rk(s**2) * du**2/dgn_rk(l1)**2) \
                            + -2*eN/l1**3 + 2*cL/l1
                        return val

                    def Jpp(l1):
                        val = 8*l1**2 * np.sum(pad_rk(s**4) * du**2/dgn_rk(l1)**3) \
                            + 6*eN/l1**4 + -2*cL/l1**2
                        return val
                    # Find inflation factor (optimize)
                    l1 = Newton_m(Jp, Jpp, 1.0)
                    # l1 = fmin_bfgs(J, x0=[1], gtol=1e-4, disp=0)
                    # l1 = minimize_scalar(J, bracket=(sqrt(prior_mode), 1e2),
                    #                      tol=1e-4).x

                else:
                    # Primal form, in a fully linearized version.
                    def za(w): return zeta_a(eN, cL, w)

                    def J(w): return \
                        .5*np.sum(((dy-w@Y)@R.sym_sqrt_inv.T)**2) + \
                        .5*N1*cL*np.log(eN + w@w)
                    # Derivatives (not required with fmin_bfgs):
                    def Jp(w): return -Y@R.inv@(dy-w@Y) + w*za(w)
                    # Jpp   = lambda w:  Y@R.inv@Y.T + \
                    #     za(w)*(eye(N) - 2*np.outer(w,w)/(eN + w@w))
                    # Approx: no radial-angular cross-deriv:
                    # Jpp   = lambda w:  Y@R.inv@Y.T + za(w)*eye(N)

                    def nvrs(w):
                        # inverse of Jpp-approx
                        return (V * (pad0(s**2, N) + za(w)) ** -1.0) @ V.T
                    # Find w (optimize)
                    wa     = Newton_m(Jp, nvrs, zeros(N), is_inverted=True)
                    # wa   = Newton_m(Jp,Jpp ,zeros(N))
                    # wa   = fmin_bfgs(J,zeros(N),Jp,disp=0)
                    l1     = sqrt(N1/za(wa))

                # Uncomment to revert to ETKF
                # l1 = 1.0

                # Explicitly inflate prior
                # => formulae look different from `bib.bocquet2015expanding`.
                A *= l1
                Y *= l1

                # Compute sqrt update
                Pw = (V * dgn_N(l1)**(-1.0)) @ V.T
                w  = dy@R.inv@Y.T@Pw
                # For the anomalies:
                if not self.Hess:
                    # Regular ETKF (i.e. sym sqrt) update (with inflation)
                    T = (V * dgn_N(l1)**(-0.5)) @ V.T * sqrt(N1)
                    # = (Y@R.inv@Y.T/N1 + eye(N))**(-0.5)
                else:
                    # Also include angular-radial co-dependence.
                    # Note: denominator not squared coz
                    # unlike `bib.bocquet2015expanding` we have inflated Y.
                    Hw = Y@R.inv@Y.T/N1 + eye(N) - 2*np.outer(w, w)/(eN + w@w)
                    T  = funm_psd(Hw, lambda x: x**-.5)  # is there a sqrtm Woodbury?

                E = mu + w@A + T@A
                E = post_process(E, self.infl, self.rot)

                stats.infl[kObs] = l1
                stats.trHK[kObs] = (((l1*s)**2 + N1)**(-1.0)*s**2).sum()/HMM.Ny

            stats.assess(k, kObs, E=E)

@ens_method
class ETKF_kernel:
    kernel_type: str
    N: int
    degre_poly : int = 2
    scalerString :  str =''
    

    def assimilate(self, HMM, xx, yy):
        Dyn, Obs, chrono, X0, stats = \
            HMM.Dyn, HMM.Obs, HMM.tseq, HMM.X0, self.stats
        # Init
        E = X0.sample(self.N)
        stats.assess(0, E=E)
        cycle =0
        distances = []
        # Loop
        for k, kObs, t, dt in progbar(chrono.ticker):
            E = Dyn(E, t-dt, dt)
            E = add_noise(E, dt, Dyn.noise, self.fnoise_treatm)

            # Analysis update
            if kObs is not None:
                cycle +=1
                stats.assess(k, kObs, 'f', E=E)
                #E = ETKF_kern_analysis(E,Obs(E,t),t, Obs.noise,Obs.kernel_localizer,
                #print(E)
                E,rg_Pa_X, inv_err_var = ETKF_kern_analysis(E,Obs(E,t),t, Obs.noise,self.kernel_type,
                                    yy[kObs], self.kernel_type, stats, kObs, self.infl, self.degre_poly,self.scalerString)
                E = post_process(E, self.infl, self.rot)
                
                stats.rg_Pa_X[kObs]  = rg_Pa_X
                stats.inv_err_var[kObs]  = inv_err_var
		
            stats.assess(k, kObs, E=E)
            #if cycle in [6,18,30,46,60,100] :
                #print(self.stats.err.rms.a[cycle-1])
        ''' if self.kernel_type == 'gaussian':
            
            print("Ecart-type des distances : ", statistics.stdev(distances))
            print("Variance des distances : ", statistics.variance(distances))
            print("Moyenne des distances : ", statistics.mean(distances))
            print("Médiane des distances : ", statistics.median(distances))
            print("Max des distances : ", np.max(distances))
            print("Min des distances : ", np.min(distances))
            print("5 quantiles des distances : ", statistics.quantiles(distances, n=5))
            plt.figure()
            plt.hist(distances, 100)
            plt.title("Répartition des distances entre les H_i et H_j")
            plt.show() '''
        

def ETKF_kern_analysis(E, Eo,t, hnoise, loc_kernel, y, kernel_type, stats, kObs, infl,  degre_poly,scalerString):
    R     = hnoise.C     # Obs noise cov
    N, Nx = E.shape      # Dimensionality
    N1    = N-1          # Ens size - 1

    mu = np.mean(E, 0)   # Ens mean
    A  = E - mu          # Ens anomalies (X_f)      
	
    xo = np.mean(Eo, 0)  # Obs ens mean
    Y  = Eo-xo           # Obs ens anomalies (HX_f)
    Np = Y.shape[1]
    dy = y - xo          # Mean "innovation" (y-Hx_bar)

    R_full = R.full
    R_tilde = hnoise.C.sym_sqrt_inv
    
    #R_demi = sla.sqrtm(R_full)

    H_bar = Y@R_tilde
    E_aug = np.concatenate((A, H_bar),axis = 1) 
 
    #Scaling of the data
    ####################
    
    #scalerString=''
    if scalerString == 'NormalizerL1':
        scaler = Normalizer(norm='l1')
    elif scalerString == 'NormalizerL2':
        scaler = Normalizer(norm='l2')
    elif scalerString == 'NormalizerMAX':
        scaler = Normalizer(norm='max')
    elif scalerString == 'StandardS':
        scaler = StandardScaler()
    elif scalerString == 'MinMaxS':
        scaler=MinMaxScaler()
    elif scalerString == 'MaxAbsS':
        scaler = MaxAbsScaler()
    else:
        scaler = None

    E_abis=E_aug.copy()
    
    if scaler != None:
        tmp = scaler.fit_transform(np.transpose(E_aug))
        E_aug=np.transpose(tmp)
    
    
    #Computing K
    ##########################
    
    if kernel_type =='lin':
        
        K = np.zeros((Nx+Np,Nx+Np))
        for i in range(Nx+Np):
            for j in range(Nx+Np):
                K[i,j] = lin_kernel(E_aug[:,i],E_aug[:,j])
    elif kernel_type=='poly':
        K = np.zeros((Nx+Np,Nx+Np))
        for i in range(Nx+Np):
            for j in range(Nx+Np):
                K[i,j] = poly_kernel(E_aug[:,i], E_aug[:,j],1, degre_poly)
     
    elif kernel_type=='tanh':
        K = np.zeros((Nx+Np,Nx+Np))
        for i in range(Nx+Np):
            for j in range(Nx+Np):
                K[i,j] = tanh_kernel(E_aug[:,i], E_aug[:,j],1e-5)
		   
    elif kernel_type=='sinc':
        K = np.zeros((Nx+Np,Nx+Np))
        alpha=0.75
        for i in range(Nx+Np):
            for j in range(Nx+Np):
                K[i,j] = alpha*lin_kernel(E_abis[:,i], E_abis[:,j])+(1-alpha)*sincd_kernel(E_aug[:,i], E_aug[:,j],0.1) #sigma=dt
    
    elif kernel_type=='add_min':
        K = np.zeros((Nx+Np,Nx+Np))
        for i in range(Nx+Np):
            for j in range(Nx+Np):
                K[i,j] = lin_kernel(E_abis[:,i], E_abis[:,j])+min_kernel(0.1+E_aug[:,i], 0.1+E_aug[:,j],0.1) #sigma=dt            	
                #print([lin_kernel(E_abis[:,i], E_abis[:,j]),min_kernel(1+E_aug[:,i], 1+E_aug[:,j],1)])
   
    elif kernel_type=='lap':
        K = np.zeros((Nx+Np,Nx+Np))
        for i in range(Nx+Np):
            for j in range(Nx+Np):
                #K[i,j] = lin_kernel(E_abis[:,i], E_abis[:,j])*laplace_kernel(E_aug[:,i], E_aug[:,j],1)
                K[i,j] = lin_kernel(E_abis[:,i], E_abis[:,j])*exp_cos_kernel(E_aug[:,i], E_aug[:,j],1/np.sqrt(N1))

    else:#linear kernel 
        K = np.zeros((Nx+Np,Nx+Np))
        for i in range(Nx+Np):
            for j in range(Nx+Np):
                K[i,j] = lin_kernel(E_aug[:,i],E_aug[:,j])

    #Computing the analysis mean
    ############################
    
    d_bar = R_tilde@dy
    alpha_H =np.linalg.solve((N1*np.eye(Np)+K[Nx:,Nx:]),d_bar)
    wa = K[Nx:,:Nx].T@alpha_H
    xa = mu + wa
    
    #Assembling Pa
    ##############
    
    Sigma_H,U_H=np.linalg.eigh(K[Nx:,Nx:])
    ind_sort=np.flip(np.argsort(Sigma_H))
    Sigma_H=Sigma_H[ind_sort]
    U_H=U_H[:,ind_sort]
    
    for ll in range(Np):
        U_H[:,ll]=U_H[:,ll]/np.sqrt(max(Sigma_H[ll],0)+N1)
    
    Pa_X=np.zeros((Nx,Nx))
    
    Ktmp=K[:Nx,Nx:]@U_H
    Pa_X[:Nx,:Nx]=(K[:Nx,:Nx]-Ktmp@Ktmp.T)/N1
    
    
    #Diagonalizing Pa
    #################
    
    Sigma_H,U_H=np.linalg.eigh(Pa_X)
    
    ind_sort=np.flip(np.argsort(Sigma_H))
    Sigma_H=Sigma_H[ind_sort]
   
    U_H=U_H[:,ind_sort]
    loc_converged=len(Sigma_H) 
    #print(Sigma_H)
    
    trace_trunc=0.0
    
    rg_Pa_X=0
    tol_eig=1.e-12#np.finfo(float).eps
    for ll in range(loc_converged):
        if(Sigma_H[ll]>tol_eig*Sigma_H[0]):
             rg_Pa_X=rg_Pa_X+1
             trace_trunc+=Sigma_H[ll]
        else:
             break
    #if(rg_Pa_X<2):
    #    print(E_abis)
	
    U_H=U_H[:,:rg_Pa_X]
    Sigma_H=Sigma_H[:rg_Pa_X]
    #print(rg_Pa_X)
    
    #Computing the analysis anomalies
    ################################
    
    #Law-rank approximation of the symmetric square root of Pa_X
    rank_A=min(N1,rg_Pa_X)
    if(rg_Pa_X<=N1):
        inv_err_var=0.0	     
    else:
        #scaling of the total variance if N1<rg_Pa_X 
        inv_err_var=np.sum(Sigma_H[N1:])/N1 
    
    #### ATTENTION : trunc
    inv_err_var=0.0    
    
    proj_X_Pa_demi = (U_H[:,:rank_A]@np.diag(np.sqrt(Sigma_H[:rank_A]+inv_err_var*np.ones((rank_A,)))))
    
    #Orthogonal matrix from Farchi and Bocquet: incremental strategy with random rotation    
    for i in range(N-rank_A):
        eps = 1.0
        c = rank_A+i+1
        teta = sqrt(c)/(sqrt(c)-eps)

                         
        Q_eps = -teta/c*np.ones([c,c])
        Q_eps[0,:] = eps/sqrt(c)
        Q_eps[:,0] = eps/sqrt(c)
        for j in range(1,c):
            Q_eps[j,j] = 1-teta/c
            
        W = np.zeros([Nx,c])
        W [:,1:c] = proj_X_Pa_demi
        proj_X_Pa_demi =W@Q_eps@genOG_1(c)

    #relative error on Pa_X
    inv_err_var=sqrt(1+N1**2)*inv_err_var/trace_trunc
    #print(sqrt(1+N1**2)*inv_err_var/trace_trunc)
         
    #print(np.abs(trace_trunc-np.trace(U_H@np.transpose(U_H)))/trace_trunc)
    #print("*****")
    #print(W)z
    #print(U_H)
	
    E=np.zeros((N,Nx))
    for ll in range(N):
        E[ll,:] = xa+sqrt(N1)*np.reshape(proj_X_Pa_demi[:Nx,ll],(Nx,)) 
    
   
	    
    return E,rg_Pa_X, inv_err_var 


def min_kernel(x,y,sigma):
    val=sigma
    for k in range(len(x)):
        val=val*min(x[k],y[k])
    return val
    
def gaussian_kernel(x,y,sigma):
    return np.exp(-np.linalg.norm(x-y)**2/(2*sigma**2))
    
def gaussian_kernel_normA(x,y,U,Sigma):
    h = x-y
    return np.exp(-h.T@(A@h))

def poly_kernel(x,y,m,d):
    n=len(x[:])
    r=m
    #r=1.0*np.sign(y@x.T)
    return ( (r+ (y@x.T)) )**d

def lin_kernel(x,y):
    return y@x.T
    
def loc_gaussien_kernel(d,L):
    return np.exp(-(d/L)**2)
    
def cauchy_kernel(x,y,sigma):
    tmp=1+lin_kernel(x,y)/(sigma**2)
    #tmp=1+(y-x)@(y-x).T/(sigma**2)
    return 1/tmp
    
def myftanh(x,c):
    tmp=c*np.linalg.norm(x)
    return (np.arctanh(tmp)/tmp)*x

def tanh_kernel(x,y,nu):
    return lin_kernel(myftanh(x,sqrt(nu)),myftanh(y,sqrt(nu)))    
    
def polh_kernel(x,y,c,m,d):
    return poly_kernel(myftanh(x,c),myftanh(y,c),m,d)  
 
def hbinom_kernel(x,y,a,c):
#    tmp=lin_kernel(x,y)/((np.linalg.norm(x)+0.1)*(np.linalg.norm(y)+0.1))
    tmp=lin_kernel(x,y)/((np.linalg.norm(myftanh(x,c))+0.1)*(np.linalg.norm(myftanh(y,c))+0.1))
    return 1/((1-tmp)**a) 
    
def hrbf_kernel(x,y,z,c):
    tmp=np.linalg.norm(myftanh(x,c)-myftanh(y,c))
    return np.linalg.norm(myftanh(x,c))*np.linalg.norm(myftanh(y,c))*np.exp(-z*tmp)      

def sinc_kernel(x,y,omega):
    tmp=np.linalg.norm(x-y)
    if np.abs(tmp)<np.finfo(float).eps*max(np.linalg.norm(x),np.linalg.norm(y)):
        kout=1.0
    else:
        kout=np.sin(pi*omega*tmp)/(pi*omega*tmp)
        #kout=np.sin(omega*tmp)/(omega*tmp)
    return kout 
    
def sinc2_kernel(x,y,omega):
    return np.sinc(omega*np.linalg.norm(x-y))
    
def sinch_kernel(x,y,omega,c):
    tmp=np.linalg.norm(myftanh(x,c)-myftanh(y,c))
    if np.abs(tmp)<np.finfo(float).eps*np.linalg.norm(myftanh(x,c)):
        kout=1.0
    else:
        kout=np.linalg.norm(myftanh(x,c))*np.linalg.norm(myftanh(y,c))*np.sin(pi*omega*tmp)/(pi*omega*tmp)
    return kout 

def sinc3_kernel(x,y,omega):
    tmp=np.linalg.norm(x-y)*np.linalg.norm(x+y)
    return np.sinc(omega*tmp**2)

def sincd_kernel(x,y,omega):
    tmp=np.sinc(omega*(x-y))
    kappa=1.0
    for k in range(len(x)):
        kappa=kappa*tmp[k]
    return kappa   

def laplace_kernel(x,y,sigma):
    return np.exp(-sigma*np.linalg.norm(x-y))
    
def exp_cos_kernel(x,y,sigma):
    norm_x=np.linalg.norm(x)
    norm_y=np.linalg.norm(y)
    seuil=np.finfo(float).eps
    condition=(norm_x<seuil)or(norm_y<seuil) 
    if(condition):
        res=0.0
    else:
        res=np.exp(-sigma*(1-lin_kernel(x/norm_x,y/norm_y)**2))
    return res
    
def lh_kernel(x,y,sigma,c):
    #return np.linalg.norm(myftanh(x,c))*np.linalg.norm(myftanh(y,c))*np.exp(-(sigma**2)*np.linalg.norm(myftanh(x,c)-myftanh(y,c))**2)
    return np.exp(-(sigma**2)*np.linalg.norm(myftanh(x,c)-myftanh(y,c))**2)

def sgauss_kernel(x,y,sigma):
    return np.exp(-lin_kernel(x,y)/(2*sigma**2))

def sgh_kernel(x,y,sigma,c):
    return np.linalg.norm(myftanh(x,c))*np.linalg.norm(myftanh(y,c))* np.exp(-tanh_kernel(x,y,c)/(2*sigma**2))
    
def gg_kernel(x,y,sigma):
    return np.arctan2(np.linalg.norm(x),pi)*np.arctan2(np.linalg.norm(y),pi)
#    return np.log(lin_kernel(x,x)/(2*sigma**2))*np.log(lin_kernel(y,y)/(2*sigma**2))

def sigmoid_kernel(x,y,r):
    return np.tanh(lin_kernel(x,y)+r)

def p_norm(x,p):
    pn=0.0
    for i in range(len(x)):
       if np.abs(x[i])>np.finfo(float).eps:
          pn=pn+np.exp(p*np.log(np.abs(x[i])))
    if pn > 0:
       pn=exp(log(pn)/p)
    return pn

def p_norm2p(x,p):
    pn=p_norm(x,p)
    if pn>0:
       return exp(log(pn)*p)
    else:
       return pn

def duality_map(x,p):
    y=np.zeros((len(x),))
    for i in range(len(x)):
       if np.abs(x[i])>np.finfo(float).eps:
          y[i]=np.log((p-1)*np.exp(np.abs(x[i])))*np.sign(x[i])
    return y
    
def p_kernel(x,y,sigma,p):
    q=p/(p-1)
    return p_norm(x,p)*p_norm(y,p)*exp(-p_norm2p(duality_map(x,p)-duality_map(y,p),q)/(exp(log(sigma)*q)))
    #return p_norm(x,p)*p_norm(y,p)*exp(-p_norm2p(x-y,p)/(exp(log(sigma)*p)))
