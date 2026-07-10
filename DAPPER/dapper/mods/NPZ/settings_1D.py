import numpy as np
import dapper as dpr
from functools import partial
import dapper.mods as modelling
from dapper.mods.NPZ import step_1D, M, dz

Nx = 3 * M

step = partial(step_1D, M=M, dz=dz)
Dyn = modelling.Operator(M=Nx, model=step, noise=0)

x_stable = np.load('etat_stable_5ans.npy')
X0 = modelling.GaussRV(C=0.01, mu=x_stable)

jj_satellite = [M]
variance_satellite = 0.05

Obs = modelling.Operator(**modelling.partial_Id_Obs(Nx, jj_satellite), noise=variance_satellite)
tseq = modelling.Chronology(dt=0.1, dko=10, Ko=1825, BurnIn=0)

HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)