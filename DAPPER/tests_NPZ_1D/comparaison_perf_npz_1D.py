import dapper as dpr
import pandas as pd
import numpy as np
from dapper.mods.NPZ.settings_1D import HMM
from dapper.da_methods import EnKF, KETKF
import time
import os
from contextlib import redirect_stdout
import dapper.tools.progressbar as pb
pb.disable_progbar = True

n_simulations = 10
results_list = []

N = 55
infl = 1.005

def creer_filtres():
    return [EnKF('Sqrt', N=N, infl=infl, rot=True, truncated=True),
    EnKF('Sqrt', N=N, infl=infl, rot=True, log_transform=True),
    #EnKF('PertObs', N=N, infl=infl, rot=False, truncated=True),
    EnKF('DEnKF', N=N, infl=infl, rot=True, truncated=True),
    EnKF('DEnKF', N=N, infl=infl, rot=True, log_transform=True),
    KETKF(N=N, infl=infl, rot=True, kernel_type='linear', truncated=True),
    KETKF(N=N, infl=infl, rot=True, kernel_type='sigmoid', c_tanh=0.01, reg_tikhonov=1e-3, truncated=True),
    KETKF(N=N, infl=infl, rot=True, kernel_type='hyperbolique', c_tanh=1e-3, reg_tikhonov=1e-3, truncated=True),
    KETKF(N=N, infl=infl, rot=True, kernel_type='lap', reg_tikhonov=1e-3, truncated=True),
    KETKF(N=N, infl=infl, rot=True, kernel_type='linear', log_transform=True),
    KETKF(N=N, infl=infl, rot=True, kernel_type='sigmoid', c_tanh=0.01, reg_tikhonov=1e-3, log_transform=True),
    KETKF(N=N, infl=infl, rot=True, kernel_type='hyperbolique', c_tanh=1e-3, reg_tikhonov=1e-3, log_transform=True),
    KETKF(N=N, infl=infl, rot=True, kernel_type='lap', reg_tikhonov=1e-3, log_transform=True)]
    #KETKF(N=N, infl=infl, rot=True, kernel_type='polynomial', poly_degree=1, reg_tikhonov=1e-2, truncated=True),
    #KETKF(N=N, infl=infl, rot=True, kernel_type='rbf_exp', sigma_rbf=0.25, reg_tikhonov=1e-3),
    #KETKF(N=N, infl=infl, rot=True, kernel_type='rbf', sigma_rbf=0.5, reg_tikhonov=1e-3),
    

def nom_filtre(f):
    try:
        return "KETKF-" + f.kernel_type + " -- log_transform = " + str(f.log_transform)
    except AttributeError:
        return "EnKF-" + f.upd_a + " -- log_transform = " + str(f.log_transform)

noms = [nom_filtre(f) for f in creer_filtres()]

rmse  = {nom: [] for nom in noms}
rmv   = {nom: [] for nom in noms}
times = {nom: [] for nom in noms}


for k in range(n_simulations):
    print(f"Simulation {k+1}/{n_simulations}")

    xx, yy = HMM.simulate()

    for f in creer_filtres():

        nom = nom_filtre(f)
        t0 = time.time()
        #xps.launch(HMM, liveplots=False, save_as=False)
        f.assimilate(HMM, xx, yy, liveplots=False)
        t1 = time.time()
        
        val_rmse = np.nanmean(f.stats.err.rms.a)
        val_rmv  = np.nanmean(f.stats.spread.rms.a)
        
        if hasattr(f, 'rang_history'):
            delattr(f, 'rang_history')

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