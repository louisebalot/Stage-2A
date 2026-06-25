import numpy as np
from matplotlib import pyplot as plt

import dapper.mods as modelling
from dapper.mods.NPZ import step_1D, x0_1D, M, D, depth       

# Simulation
dt = 0.1
k_steps = 2000
xx_1D = np.zeros((k_steps, 3 * M))
xx_1D[0] = x0_1D

# boucle temporelle simu
for k in range(k_steps - 1):
    t = k * dt
    xx_1D[k+1] = step_1D(xx_1D[k], t, dt, M, D)

N_data = xx_1D[:, 0:M].T       # Les M premières colonnes
P_data = xx_1D[:, M:2*M].T     # Les M colonnes du milieu
Z_data = xx_1D[:, 2*M:3*M].T   # Les M dernières colonnes

# même échelle de profondeur
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Nutriments (Bleu)
im_N = axes[0].imshow(N_data, aspect='auto', extent=[0, k_steps*dt, depth, 0], cmap='Blues')
axes[0].set_title("Nutriments (N)")
axes[0].set_xlabel("Temps")
axes[0].set_ylabel("Profondeur (m)")
fig.colorbar(im_N, ax=axes[0])

# Phytoplancton (Vert)
im_P = axes[1].imshow(P_data, aspect='auto', extent=[0, k_steps*dt, depth, 0], cmap='Greens')
axes[1].set_title("Phytoplancton (P)")
axes[1].set_xlabel("Temps")
fig.colorbar(im_P, ax=axes[1])

# Zooplancton (Rouge)
im_Z = axes[2].imshow(Z_data, aspect='auto', extent=[0, k_steps*dt, depth, 0], cmap='Reds')
axes[2].set_title("Zooplancton (Z)")
axes[2].set_xlabel("Temps")
fig.colorbar(im_Z, ax=axes[2])

plt.suptitle("Dynamique 1D de la colonne d'eau (Modèle NPZ)", fontsize=16)
plt.tight_layout()
plt.show()