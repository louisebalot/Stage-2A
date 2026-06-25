import dapper as dpr
import numpy as np
from dapper.mods.NPZ.settings_1D import HMM
from dapper.da_methods import EnKF, KETKF
import dapper.tools.progressbar as pb
pb.disable_progbar = True

xx, yy = HMM.simulate()
yy = np.maximum(yy, 1e-8)

xps = dpr.xpList()

N = 30
infl = 1.04

# références
xps += EnKF('Sqrt', N=N, infl=infl, rot=True)
xps += KETKF(N=N, infl=infl, rot=True, kernel_type='linear')

# tests
xps += KETKF(N=N, infl=infl, rot=True, kernel_type='sigmoid', c_tanh=0.01 , reg_tikhonov=1e-3)

xps += KETKF(N=N, infl=infl, rot=True, kernel_type='hyperbolique', c_tanh=1e-3, reg_tikhonov=1e-3)

xps += KETKF(N=N, infl=infl, rot=True, kernel_type='polynomial', poly_degree=1, reg_tikhonov=1e-2)

xps += KETKF(N=N, infl=infl, rot=True, kernel_type='rbf_exp', sigma_rbf=15.0, reg_tikhonov=0.1)

xps += KETKF(N=N, infl=infl, rot=True, kernel_type='rbf', sigma_rbf=15.0, reg_tikhonov=0.1)

xps += KETKF(N=N, infl=infl, rot=True, kernel_type='lap', reg_tikhonov=1e-3)

xps.launch(HMM, liveplots=False, save_as=False)
print(f"Simulation lancée avec N = {N}, et inflation = {infl}")
print(xps.tabulate_avrgs(['rmse.a', 'rmv.a']))