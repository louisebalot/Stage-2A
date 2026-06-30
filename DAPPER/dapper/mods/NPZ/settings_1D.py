import numpy as np
from functools import partial
import dapper.mods as modelling
from dapper.mods.NPZ import step_1D, x0_1D, M, D, dz, Kz

Nx = 3 * M

#step_1D_assimil = partial(step_1D, M=M, D=D)
step_1D_assimil = partial(step_1D, M=M, dz=dz, Kz=Kz)

Dyn = modelling.Operator(M=Nx, model=step_1D_assimil, noise=0)

jj_satellite = [M] 
Obs = modelling.Operator(**modelling.partial_Id_Obs(Nx, jj_satellite), noise=0.01)

#tseq = modelling.Chronology(dt=0.1, dko=10, Ko=200, BurnIn=0)
tseq = modelling.Chronology(dt=0.1, dko=10, Ko=1000, BurnIn=100)

X0 = modelling.GaussRV(C=0.01, mu=x0_1D)

HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)