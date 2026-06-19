import dapper as dpr
import dapper.mods.Lorenz63.sakov2012 as L63
from dapper.da_methods import EnKF, KETKF

# Modèle (Hidden Markov Model - HMM)
# configuration classique 
HMM = L63.HMM

# vraie trajectoire (xx) et observations bruitées (yy)
print("Génération de la vérité et des observations")
xx, yy = HMM.simulate()

# Pour lancer en liste
N = 10
xps = dpr.xpList()

xps += EnKF('Sqrt', N=N, infl=1.02, rot=True)
xps += KETKF(N=N, infl=1.02, rot=True, kernel_type='linear')

# tests
xps += KETKF(N=N, infl=1.01, rot=True, kernel_type='sigmoid', c_tanh=0.01 , reg_tikhonov=1e-3)

xps += KETKF(N=N, infl=1.01, rot=True, kernel_type='hyperbolique', c_tanh=1e-3, reg_tikhonov=1e-3)

xps += KETKF(N=N, infl=1.01, rot=True, kernel_type='polynomial', poly_degree=1, reg_tikhonov=1e-2)
xps += KETKF(N=N, infl=1.01, rot=True, kernel_type='polynomial', poly_degree=2, reg_tikhonov=1e-2)

xps += KETKF(N=N, infl=1.01, rot=True, kernel_type='rbf_exp', sigma_rbf=0.25, reg_tikhonov=1e-3)

xps += KETKF(N=N, infl=1.01, rot=True, kernel_type='rbf', sigma_rbf=0.5, reg_tikhonov=1e-3)

xps.launch(HMM, liveplots=False)

print(xps.tabulate_avrgs(["rmse.a", "rmv.a"]))

"""
# xp = EnKF('Sqrt', N=10, infl=1.02, rot=True)
# Test KETKF
# kernel_type = 'linear', 'polynomial', 'sigmoid', 'rbf', 'rbf_exp' ou 'hyperbolique'
xp = KETKF(N=10, infl=1.02, rot=True, kernel_type ='linear', poly_degree=3, c_tanh=0.005)


if xp.kernel_type == 'polynomial':
    print(f"assimilation avec KETKF (noyau {xp.kernel_type}, degré d = {xp.poly_degree})")

elif xp.kernel_type in ['rbf', 'rbf_exp']:
    print(f"assimilation avec KETKF (noyau {xp.kernel_type}, sigma = {xp.sigma_rbf})")

elif xp.kernel_type in ['sigmoid', 'hyperbolique']:
    print(f"assimilation avec KETKF (noyau {xp.kernel_type}, c = {xp.c_tanh})")

else:
    print(f"assimilation avec KETKF (noyau {xp.kernel_type})")
print(f"inflation = {xp.infl}, N = {xp.N}")

xp.assimilate(HMM, xx, yy, liveplots=False)
xp.stats.average_in_time()

print("résultats :")
print(xp.avrgs.tabulate(['rmse.a', 'rmv.a']))"""