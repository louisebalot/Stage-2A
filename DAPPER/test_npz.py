import dapper as dpr
import numpy as np
from dapper.mods.NPZ.settings_1 import HMM
from dapper.da_methods import EnKF, KETKF
import dapper.tools.progressbar as pb
pb.disable_progbar = True

n_simulations = 20
results_list = []
N = 40
infl = 1.04

for i in range(n_simulations):
    print(f"Simulation {i+1}/{n_simulations}", end="\r") 

    # vraie trajectoire (xx) et observations bruitées (yy)
    xx, yy = HMM.simulate()

    xps = dpr.xpList()

    # références
    xps += EnKF('Sqrt', N=N, infl=infl, rot=True)
    xps += KETKF(N=N, infl=infl, rot=True, kernel_type='linear')

    # tests
    xps += KETKF(N=N, infl=infl, rot=True, kernel_type='sigmoid', c_tanh=0.01 , reg_tikhonov=1e-3)

    xps += KETKF(N=N, infl=infl, rot=True, kernel_type='hyperbolique', c_tanh=1e-3, reg_tikhonov=1e-3)

    xps += KETKF(N=N, infl=infl, rot=True, kernel_type='polynomial', poly_degree=1, reg_tikhonov=1e-2)

    xps += KETKF(N=N, infl=infl, rot=True, kernel_type='rbf_exp', sigma_rbf=0.25, reg_tikhonov=1e-3)

    xps += KETKF(N=N, infl=infl, rot=True, kernel_type='rbf', sigma_rbf=0.5, reg_tikhonov=1e-3)

    xps.launch(HMM, liveplots=False)
    
    sim_results = []
    for xp in xps:
        if xp.stats is not None and hasattr(xp.stats, 'rmse'):
            val = np.nanmean(xp.stats.rmse.a)
            sim_results.append(val)
        else:
            sim_results.append(np.nan)
    results_list.append(sim_results)

print(f"Simulation lancée avec N = {N}, et inflation = {infl}")

results_array = np.array(results_list)
mean_rmse = np.mean(results_array, axis=0)
std_rmse = np.std(results_array, axis=0)
# print(xps.tabulate_avrgs(['rmse.a', 'rmv.a']))

print("\n" + "="*60)
print(f"résultats moyennés sur {n_simulations} simulations")
print("="*60)
for i, xp in enumerate(xps):
    print(f"{xp.da_method.__class__.__name__:<15} | RMSE Moyen: {mean_rmse[i]:.4f} ± {std_rmse[i]:.4f}")