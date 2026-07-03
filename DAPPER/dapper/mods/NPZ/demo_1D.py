import numpy as np
from matplotlib import pyplot as plt

import dapper.mods as modelling
from dapper.mods.NPZ import step_1D, x0_1D, M, depth, dz  

# Simulation
dt = 0.1
#k_steps = 7300   # 2 ans
k_steps = 18250 
xx_1D = np.zeros((k_steps, 3 * M))
xx_1D[0] = x0_1D

# boucle temporelle simu
for k in range(k_steps - 1):
    t = k * dt
    xx_1D[k+1] = step_1D(xx_1D[k], t, dt, M, dz)

N_data = xx_1D[:, 0:M].T
P_data = xx_1D[:, M:2*M].T
Z_data = xx_1D[:, 2*M:3*M].T

# même échelle de profondeur
fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
t_max = k_steps * dt

# Nutriments (Bleu)
im_N = axes[0].imshow(N_data, aspect='auto', extent=[0, t_max, depth, 0], cmap='YlGnBu', interpolation='nearest')
axes[0].set_title("Nutriments (N)", fontsize=12, fontweight='bold')
axes[0].set_ylabel("Profondeur (m)", fontsize=11)
fig.colorbar(im_N, ax=axes[0], orientation='horizontal', pad=0.1, label="Concentration")

# Phytoplancton (Vert)
im_P = axes[1].imshow(P_data, aspect='auto', extent=[0, t_max, depth, 0], cmap='YlGn', interpolation='nearest')
axes[1].set_title("Phytoplancton (P)", fontsize=12, fontweight='bold')
fig.colorbar(im_P, ax=axes[1], orientation='horizontal', pad=0.1, label="Concentration")

# Zooplancton (Rouge)
im_Z = axes[2].imshow(Z_data, aspect='auto', extent=[0, t_max, depth, 0], cmap='OrRd', interpolation='nearest')
axes[2].set_title("Zooplancton (Z)", fontsize=12, fontweight='bold')
fig.colorbar(im_Z, ax=axes[2], orientation='horizontal', pad=0.1, label="Concentration")

# saison en arrière plan
nb_annees = int(t_max // 365)

for ax in axes:
    ax.set_xlabel("Temps (Jours)", fontsize=11)
    
    for annee in range(nb_annees):
        debut_annee = annee * 365
        
        label_hiver = 'Hiver' if (annee == 0 and ax == axes[0]) else ""
        ax.axvspan(debut_annee, debut_annee + 60, 
                   fill=False, hatch='\\\\\\', edgecolor='dodgerblue', alpha=0.4, 
                   linewidth=0, label=label_hiver)
        
        label_bloom = 'Printemps - Bloom' if (annee == 0 and ax == axes[0]) else ""
        ax.axvspan(debut_annee + 60, debut_annee + 150, 
                   fill=False, hatch='///', edgecolor='limegreen', alpha=0.4, 
                   linewidth=0, label=label_bloom)
        
        ax.axvspan(debut_annee + 330, debut_annee + 365, 
                   fill=False, hatch='\\\\\\', edgecolor='dodgerblue', alpha=0.4, 
                   linewidth=0)

axes[0].legend(loc='upper right', fontsize=9, facecolor='white', framealpha=1.0)

plt.suptitle("Dynamique 1D de la colonne d'eau (Modèle NPZ)", fontsize=16)
plt.tight_layout()
plt.show()