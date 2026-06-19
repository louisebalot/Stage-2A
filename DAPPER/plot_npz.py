import numpy as np
import matplotlib.pyplot as plt
import dapper as dpr
from dapper.da_methods import EnKF, KETKF

from dapper.mods.NPZ.settings_1 import HMM

print("Génération de la vérité et des observations...")
xx, yy = HMM.simulate()

enkf = EnKF('Sqrt', N=10, infl=1.02, rot=True)
ketkf = KETKF(N=10, kernel_type='linear', infl=1.02, rot=True, reg_tikhonov=1e-10)

enkf.assimilate(HMM, xx, yy, liveplots=False)
ketkf.assimilate(HMM, xx, yy, liveplots=False)

time_truth = HMM.tseq.tt 
truth = xx

time_analysis = HMM.tseq.tt[HMM.tseq.dko :: HMM.tseq.dko]

mu_enkf = enkf.stats.mu.a
mu_ketkf = ketkf.stats.mu.a

t_obs = time_analysis
y_obs = [np.array(y).item() for y in yy]

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
variables = ['Nutriments (N)', 'Phytoplancton (P)', 'Zooplancton (Z)']
colors = ['blue', 'green', 'red']

for i in range(3):
    ax = axes[i]
    
    ax.plot(time_truth, truth[:, i], label='Vérité (Nature)', color='black', linewidth=2, linestyle='--')
    
    ax.plot(time_analysis, mu_enkf[:, i], label='EnKF (Classique)', color='orange', alpha=0.8)
    ax.plot(time_analysis, mu_ketkf[:, i], label='KETKF (Ton filtre)', color=colors[i], alpha=0.8)
    
    if i == 1:
        ax.scatter(t_obs, y_obs, color='black', marker='*', s=50, label='Obs Satellites', zorder=5)

    ax.set_ylabel(variables[i])
    ax.grid(True)

    # ax.set_xlim(0, 200)

    if i == 0:
        ax.legend(loc='upper right')

axes[-1].set_xlabel('Temps')
fig.suptitle('Assimilation de données : Découverte du Zooplancton caché', fontsize=16)
plt.tight_layout()
plt.show()