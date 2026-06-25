import dapper as dpr
import numpy as np
from dapper.da_methods import EnKF
from dapper.da_methods import KETKF
import dapper.mods as modelling
from dapper.mods.Lorenz96 import step, x0, dstep_dx
from dapper.mods import Chronology
import dapper.tools.progressbar as pb
pb.disable_progbar = True

M = 40 

Dyn = modelling.Operator(M=M, model=step, linear=dstep_dx, noise=0)

jj = np.arange(0, M, 2) 
Obs = modelling.Operator(
    **modelling.partial_Id_Obs(M, jj),
    noise=0.1 
)

tseq = modelling.Chronology(dt=0.05, K=500, dto=0.05)

# 5. État initial probabiliste
X0 = modelling.GaussRV(mu=x0(M), C=0.001)

HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)

# vraie trajectoire (xx) et observations bruitées (yy)
xx, yy = HMM.simulate()

N = 100
xps = dpr.xpList()

xps += EnKF('Sqrt', N=N, infl=1.01, rot=True)
xps += KETKF(N=N, infl=1.01, rot=True, kernel_type='linear')

# tests
xps += KETKF(N=N, infl=1.01, rot=True, kernel_type='sigmoid', c_tanh=0.01 , reg_tikhonov=1e-3)

xps += KETKF(N=N, infl=1.01, rot=True, kernel_type='hyperbolique', c_tanh=1e-3, reg_tikhonov=1e-3)

xps += KETKF(N=N, infl=1.01, rot=True, kernel_type='polynomial', poly_degree=1, reg_tikhonov=1e-2)
xps += KETKF(N=N, infl=1.01, rot=True, kernel_type='polynomial', poly_degree=2, reg_tikhonov=1e-2)

xps += KETKF(N=N, infl=1.01, rot=True, kernel_type='rbf_exp', sigma_rbf=0.25, reg_tikhonov=1e-3)

xps += KETKF(N=N, infl=1.01, rot=True, kernel_type='rbf', sigma_rbf=0.5, reg_tikhonov=1e-3)

xps.launch(HMM, liveplots=False)

print(f"Simulation lancée avec N = {N}")
print(xps.tabulate_avrgs(["rmse.a", "rmv.a"]))