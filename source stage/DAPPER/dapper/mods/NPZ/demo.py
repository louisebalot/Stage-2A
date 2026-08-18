import numpy as np
from matplotlib import pyplot as plt

import dapper.mods as modelling
from dapper.mods.NPZ import step, x0

simulator = modelling.with_recursion(step, prog="Simulating")

# Simulation de 1000 pas de temps
dt = 0.1
nb_annees_simu = 5
n_steps = int(nb_annees_simu * 365 / dt)
xx = simulator(x0, k=n_steps, t0=0, dt=dt)

# Affichage des résultats
plt.figure(figsize=(12, 6))
time = np.arange(len(xx)) * dt

plt.plot(time, xx[:, 0], label="Nutriments (N)", color="blue", lw=2)
plt.plot(time, xx[:, 1], label="Phytoplancton (P)", color="green", lw=2)
plt.plot(time, xx[:, 2], label="Zooplancton (Z)", color="red", lw=2)

plt.title("Dynamique du modèle NPZ (sans assimilation)")
plt.xlabel("Temps (Jours)")
plt.ylabel("Concentration")
plt.xlim(0, nb_annees_simu*365)
plt.legend()
plt.grid(True)
plt.show()