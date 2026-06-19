"""
Démonstration de la dynamique du modèle NPZ (sans assimilation).
"""

import numpy as np
from matplotlib import pyplot as plt

import dapper.mods as modelling
from dapper.mods.NPZ import step, x0

simulator = modelling.with_recursion(step, prog="Simulating")

# Simulation de 1000 pas de temps
dt = 0.1
xx = simulator(x0, k=2000, t0=0, dt=dt)

# Affichage des résultats
plt.figure(figsize=(10, 6))
time = np.arange(len(xx)) * dt

plt.plot(time, xx[:, 0], label="Nutriments (N)", color="blue", lw=2)
plt.plot(time, xx[:, 1], label="Phytoplancton (P)", color="green", lw=2)
plt.plot(time, xx[:, 2], label="Zooplancton (Z)", color="red", lw=2)

plt.title("Dynamique du modèle NPZ (sans assimilation)")
plt.xlabel("Temps")
plt.ylabel("Concentration")
plt.legend()
plt.grid(True)
plt.show()