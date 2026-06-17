import dapper as dpr
from dapper.mods.NPZ.settings_1 import HMM
from dapper.da_methods import EnKF, KETKF

# vraie trajectoire (xx) et observations bruitées (yy)
print("Génération de la vérité et des observations")
xx, yy = HMM.simulate()

xps = dpr.xpList()

# références
xps += EnKF('Sqrt', N=10, infl=1.02, rot=True)
xps += KETKF(N=10, infl=1.02, rot=True, kernel_type='linear')

# tests
xps += KETKF(N=20, infl=1.02, rot=True, kernel_type='polynomial', poly_degree=2)
xps += KETKF(N=20, infl=1.02, rot=True, kernel_type='polynomial', poly_degree=3)

for s in [0.2, 0.4, 0.6, 0.8]:
    xps += KETKF(N=20, infl=1.02, rot=True, kernel_type='rbf', sigma_rbf=s)

xps.launch(HMM, liveplots=False)

print(xps.tabulate_avrgs(['rmse.a', 'rmv.a']))