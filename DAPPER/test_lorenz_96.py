import dapper as dpr
import numpy as np
from dapper.da_methods import EnKF
from dapper.da_methods import KETKF
import dapper.mods as modelling
from dapper.mods.Lorenz96 import step, x0, dstep_dx
from dapper.mods import Chronology

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
print("Génération de la vérité et des observations")
xx, yy = HMM.simulate()

# xp = EnKF('Sqrt', N=10, infl=1.02, rot=True)
# Test KETKF
# kernel_type = 'linear', 'polynomial', 'sigmoid', 'rbf', 'rbf_exp' ou 'hyperbolique'
xp = KETKF(N=40, infl=1.01, rot=True, kernel_type ='rbf', poly_degree=2, sigma_rbf= 200, c_tanh=0.001)


if xp.kernel_type == 'polynomial':
    print(f"assimilation avec KETKF (noyau {xp.kernel_type}, degré d = {xp.poly_degree})")

elif xp.kernel_type in ['rbf', 'rbf_exp']:
    print(f"assimilation avec KETKF (noyau {xp.kernel_type}, sigma = {xp.sigma_rbf})")

elif xp.kernel_type in ['sigmoid', 'hyperbolique']:
    print(f"assimilation avec KETKF (noyau {xp.kernel_type}, c = {xp.c_tanh})")

else:
    print(f"assimilation avec KETKF (noyau {xp.kernel_type})")
print(f"inflation = {xp.infl}, N = {xp.N}")

xp.assimilate(HMM, xx, yy, liveplots=True)
xp.stats.average_in_time()

print("résultats :")
print(xp.avrgs.tabulate(['rmse.a', 'rmv.a']))