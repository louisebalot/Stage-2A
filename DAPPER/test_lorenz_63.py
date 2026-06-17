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

xps = dpr.xpList()

#xps += EnKF('Sqrt', N=10, infl=1.02, rot=True)
#xps += EnKF('PertObs', N=40, infl=1.06)

xps += KETKF(N=10, infl=1.02, rot=True, kernel_type ='linear')
#xps += KETKF(N=20, infl=1.02, rot=True, kernel_type='rbf', sigma_rbf=10)
#xps += KETKF(N=20, infl=1.02, rot=True, kernel_type='hyperbolique', c_tanh=0.01)
for s in [0.1, 1.0, 5.0, 20.0]:
    xps += KETKF(N=20, infl=1.02, rot=True, kernel_type='rbf', sigma_rbf=s)

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