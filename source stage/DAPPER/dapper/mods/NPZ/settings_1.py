import numpy as np

import dapper.mods as modelling
from dapper.mods.NPZ import step, x0

# dt = pas d'intégration de RK4
# dko = observation tous les dko pas
# Ko = nombre observations total
tseq = modelling.Chronology(dt=0.1, dko=10, Ko=1825, BurnIn=0)
tseq = modelling.Chronology(dt=0.1, dko=5, Ko=2920, BurnIn=730)

Nx = len(x0) # Nx = 3 (N, P, Z)

Dyn = modelling.Operator(M=Nx, model=step, noise=0)

# État initial incertain (X0)
# grande incertitude (variance C=0.1) = ensemble très dispersé
X0 = modelling.GaussRV(C=0.01, mu=x0)

# Opérateur d'Observation (satellites)
# On observe uniquement le Phytoplancton
jj = [1]  
# noise=0.01 : l'erreur de mesure du satellite
Obs = modelling.Operator(**modelling.partial_Id_Obs(Nx, jj), noise=0.01)

HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
