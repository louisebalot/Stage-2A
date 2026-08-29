import dapper as dpr
import matplotlib.pyplot as plt
import numpy as np
import re
from dapper.mods.NPZ.settings_1D import HMM
from dapper.da_methods import EnKF, KETKF
import dapper.tools.progressbar as pb
pb.disable_progbar = True

xx, yy = HMM.simulate()

def nom_filtre(f):
    try:
        return "KETKF-" + f.kernel_type
    except AttributeError:
        return "EnKF-" + f.upd_a
    

xps = dpr.xpList()

N = 55
infl = 1.005

# références
xps += EnKF('Sqrt', N=N, infl=infl, rot=True, truncated=True)
#xps += EnKF('PertObs', N=N, infl=infl, rot=False, truncated=True),
xps += EnKF('DEnKF', N=N, infl=infl, rot=True, truncated=True),

#xps += KETKF(N=N, infl=infl, rot=True, kernel_type='linear', truncated=True)
xps += KETKF(N=N, infl=infl, rot=True, kernel_type='linear', log_transform=True)

# tests
xps += KETKF(N=N, infl=infl, rot=True, kernel_type='sigmoid', c_tanh=0.01 , reg_tikhonov=1e-3, truncated=True)
xps += KETKF(N=N, infl=infl, rot=True, kernel_type='sigmoid', c_tanh=0.01 , reg_tikhonov=1e-3, log_transform=True)

xps += KETKF(N=N, infl=infl, rot=True, kernel_type='hyperbolique', c_tanh=1e-3, reg_tikhonov=1e-3, truncated=True)
xps += KETKF(N=N, infl=infl, rot=True, kernel_type='hyperbolique', c_tanh=1e-3, reg_tikhonov=1e-3, log_transform=True)

xps += KETKF(N=N, infl=infl, rot=True, kernel_type='lap', reg_tikhonov=1e-3, truncated=True)
xps += KETKF(N=N, infl=infl, rot=True, kernel_type='lap', reg_tikhonov=1e-3, log_transform=True)

xps += KETKF(N=N, infl=infl, rot=True, kernel_type='rbf_exp', sigma_rbf=15.0, reg_tikhonov=0.1, truncated=True)

xps += KETKF(N=N, infl=infl, rot=True, kernel_type='rbf', sigma_rbf=15.0, reg_tikhonov=0.1, truncated=True)

xps += KETKF(N=N, infl=infl, rot=True, kernel_type='polynomial', poly_degree=1, reg_tikhonov=1e-2, truncated=True)

#xps.launch(HMM_log, liveplots=False, save_as=False)
print(f"Simulation lancée avec N = {N}, et inflation = {infl}")
print(f"\n{'Noyau':<25} | {'RMSE (a)':<15} | {'RMV (a)':<15} | {'Paramètres':<42} | {'log_transform'}")
print("-" * 145)

for xp in xps:
    xp.assimilate(HMM, xx, yy, liveplots=False)
    
    nom = nom_filtre(xp)
    
    rmse_moyenne = np.nanmean(xp.stats.err.rms.a)
    rmv_moyenne = np.nanmean(xp.stats.spread.rms.a)

    params = []
    if hasattr(xp, 'reg_tikhonov'):
        params.append(f"reg={xp.reg_tikhonov}")
    if hasattr(xp, 'c_tanh'):
        params.append(f"c_tanh={xp.c_tanh}")
    if hasattr(xp, 'poly_degree'):
        params.append(f"deg={xp.poly_degree}")
    if hasattr(xp, 'sigma_rbf'):
        params.append(f"sigma={xp.sigma_rbf}")

    params_str = ", ".join(params)

    if hasattr(xp, 'rang_history'):
        delattr(xp, 'rang_history')
    
    print(f"{nom:<25} | {rmse_moyenne:<15.4g} | {rmv_moyenne:<15.4g} | {params_str:<42} | {str(xp.log_transform)}")

#print(xps.tabulate_avrgs(['rmse.a', 'rmv.a'])