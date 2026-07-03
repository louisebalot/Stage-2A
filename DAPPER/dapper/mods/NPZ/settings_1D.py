import numpy as np
import dapper as dpr
from functools import partial
import dapper.mods as modelling
from dapper.mods.NPZ import step_1D, step_1D_log, x0_1D, M, dz

Nx = 3 * M

# pour modèle
step_physique = partial(step_1D, M=M, dz=dz)
Dyn_physique = modelling.Operator(M=Nx, model=step_physique, noise=0)

x_stable = np.load('etat_stable_5ans.npy')
X0_physique = modelling.GaussRV(C=0.01, mu=x_stable)

# pour noyaux
step_log = partial(step_1D_log, M=M, dz=dz)
Dyn_log = modelling.Operator(M=Nx, model=step_log, noise=0)

w0_stable = np.log(np.maximum(x_stable, 1e-20))
X0_log = modelling.GaussRV(C=0.01, mu=w0_stable)


jj_satellite = [M]
variance_satellite = 0.05

Obs_physique = modelling.Operator(**modelling.partial_Id_Obs(Nx, jj_satellite), noise=variance_satellite)

def obs_model_log(w):
    x = np.exp(w)
    if w.ndim == 1:
        return x[jj_satellite]
    else:
        return x[:, jj_satellite]

#bruit_obs = modelling.GaussRV(C=0.001, M=len(jj_satellite))
bruit_obs = modelling.GaussRV(C=variance_satellite, M=len(jj_satellite))

Obs_log = modelling.Operator(M=Nx, p=len(jj_satellite), model=obs_model_log, noise=bruit_obs)

tseq = modelling.Chronology(dt=0.1, dko=10, Ko=1000, BurnIn=100)

#HMM = modelling.HiddenMarkovModel(Dyn_physique, Obs_physique, tseq, X0_physique)
HMM_physique = modelling.HiddenMarkovModel(Dyn_physique, Obs_physique, tseq, X0_physique)
HMM_log = modelling.HiddenMarkovModel(Dyn_log, Obs_log, tseq, X0_log)