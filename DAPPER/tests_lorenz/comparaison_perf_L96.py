import dapper as dpr
import numpy as np
from dapper.da_methods import EnKF
from dapper.da_methods import KETKF
import dapper.mods as modelling
from dapper.mods.Lorenz96 import step, x0, dstep_dx
from dapper.mods import Chronology
import dapper.tools.progressbar as pb
pb.disable_progbar = True
import time
import pandas as pd
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

N = 40
n_simulations = 30
infl = 1.02

def creer_filtres():
    return [EnKF('Sqrt', N=N, infl=infl, rot=True),
    KETKF(N=N, infl=infl, rot=True, kernel_type='linear', truncated=False),
    KETKF(N=N, infl=infl, rot=True, kernel_type='sigmoid', c_tanh=0.01, reg_tikhonov=1e-3, truncated=False),
    KETKF(N=N, infl=infl, rot=True, kernel_type='hyperbolique', c_tanh=1e-3, reg_tikhonov=1e-3, truncated=False),
    KETKF(N=N, infl=infl, rot=True, kernel_type='polynomial', poly_degree=1, reg_tikhonov=1e-2, truncated=False),
    KETKF(N=N, infl=infl, rot=True, kernel_type='rbf_exp', sigma_rbf=0.25, reg_tikhonov=1e-3, truncated=False),
    KETKF(N=N, infl=infl, rot=True, kernel_type='rbf', sigma_rbf=0.5, reg_tikhonov=1e-3, truncated=False),
    KETKF(N=N, infl=infl, rot=True, kernel_type='lap', reg_tikhonov=1e-3)]

def nom_filtre(f):
    try:
        return "KETKF-" + f.kernel_type
    except AttributeError:
        return "EnKF-Sqrt"

noms = [nom_filtre(f) for f in creer_filtres()]

rmse  = {nom: [] for nom in noms}
rmv   = {nom: [] for nom in noms}
times = {nom: [] for nom in noms}


for k in range(n_simulations):
    print(f"Simulation {k+1}/{n_simulations}")

    xx, yy = HMM.simulate()

    for f in creer_filtres():

        nom = nom_filtre(f)
        
        xps = dpr.xpList()
        xps += f
        xps[-1].name = nom

        t0 = time.time()
        xps.launch(HMM, liveplots=False, save_as=False)
        t1 = time.time()

        xp = xps[0]

        try:
            val_rmse = xp.avrgs["err"].rms.a.val
            val_rmv  = xp.avrgs["spread"].rms.a.val
        except Exception as e:
            val_rmse = np.nan
            val_rmv  = np.nan

        rmse[nom].append(val_rmse)
        rmv[nom].append(val_rmv)
        times[nom].append(t1 - t0)

rmse_avg = {k: np.nanmean(v) for k, v in rmse.items()}
rmse_std = {k: np.nanstd(v) for k, v in rmse.items()}
rmv_avg  = {k: np.nanmean(v) for k, v in rmv.items()}
time_avg = {k: np.nanmean(v) for k, v in times.items()}

df = pd.DataFrame({
    "RMSE moyen": rmse_avg,
    "RMSE std": rmse_std,
    "RMV moyen": rmv_avg,
    "Temps moyen (s)": time_avg
})

print(f"résultats moyennés sur {n_simulations} simulations, avec N = {N} et inflation = {infl}")
print(df.sort_values(by="RMSE moyen"))