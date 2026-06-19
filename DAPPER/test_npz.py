import dapper as dpr
from dapper.mods.NPZ.settings_1 import HMM
from dapper.da_methods import EnKF, KETKF
import dapper.tools.progressbar as pb
pb.disable_progbar = True

# vraie trajectoire (xx) et observations bruitées (yy)
xx, yy = HMM.simulate()

xps = dpr.xpList()

N = 5

# références
xps += EnKF('Sqrt', N=N, infl=1.02, rot=True)
xps += KETKF(N=N, infl=1.02, rot=True, kernel_type='linear')

# tests
xps += KETKF(N=N, infl=1.01, rot=True, kernel_type='sigmoid', c_tanh=0.01 , reg_tikhonov=1e-3)

xps += KETKF(N=N, infl=1.01, rot=True, kernel_type='hyperbolique', c_tanh=1e-3, reg_tikhonov=1e-3)

xps += KETKF(N=N, infl=1.01, rot=True, kernel_type='polynomial', poly_degree=1, reg_tikhonov=1e-2)

xps += KETKF(N=N, infl=1.01, rot=True, kernel_type='rbf_exp', sigma_rbf=0.25, reg_tikhonov=1e-3)

xps += KETKF(N=N, infl=1.01, rot=True, kernel_type='rbf', sigma_rbf=0.5, reg_tikhonov=1e-3)

xps.launch(HMM, liveplots=False)
print(f"Simulation lancée avec N = {N}")
print(xps.tabulate_avrgs(['rmse.a', 'rmv.a']))