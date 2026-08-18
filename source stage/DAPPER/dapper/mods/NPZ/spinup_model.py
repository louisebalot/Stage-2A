import numpy as np

import dapper.mods as modelling
from dapper.mods.NPZ import step_1D, M, dz, x0_1D  

# Simulation
dt = 0.1
k_steps = 18250   # 5 ans
xx_1D = np.zeros((k_steps, 3 * M))
# valeurs prises dans l'article
xx_1D[0] = x0_1D

# simulation boucle temporelle 
for k in range(k_steps - 1):
    t = k * dt
    xx_1D[k+1] = step_1D(xx_1D[k], t, dt, M, dz)

final_state = xx_1D[-1]
print(f"État après 5 ans : N(surf)={final_state[0]:.2f}, N(fond)={final_state[M-1]:.2f}")
print(f"Phyto au fond : {final_state[2*M-1]:.2e}")
np.save('etat_stable_5ans.npy', final_state)