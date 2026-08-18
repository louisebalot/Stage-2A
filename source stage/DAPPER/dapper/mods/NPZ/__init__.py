import numpy as np
import dapper.mods as modelling

# constantes biologiques

mu_max = 0.9        # taux croissance max phytoplancton
K_N = 0.5           # gradualité de la croissance des nutriments
g_max = 1.0         # taux broutage max zooplancton
K_P = 1.0           # influence prédation pour phytoplancton
m_P = 0.07          # taux mortalité linéaire phytoplancton
m_Z = 0.07          # taux mortalité linéaire zooplancton
beta = 0.5          # efficacité assimilation zooplancton (transformé en biomasse)
gamma_Z = 0.05      # taux excretion, recyclage nutriments par zooplancton

"""
mu_max = 1.0
K_N = 0.5
g_max = 2.0
K_P = 0.5
m_P = 0.1
m_Z = 0.2
beta = 0.6
gamma_Z = 0.1
"""

# État initial [N, P, Z]
x0 = np.array([10.0, 0.1, 0.1])

# 0D
@modelling.ens_compatible
def dxdt(x, t):
    
    # Calcule les dérivées [dN/dt, dP/dt, dZ/dt] selon le modèle NPZ

    x = np.maximum(x, 1e-8)
    N, P, Z = x

    # simulation cycles
    T_saison = 365 
    saison = 0.5 * (1 - np.cos(2 * np.pi * t / T_saison))
    mu_saisonnier = mu_max * (0.2 + 0.8 * saison)

    mu_P = mu_saisonnier * (N / (K_N + N))
    g_Z = g_max * (P / (K_P + P))

    # Équations
    dN = -mu_P * P + gamma_Z * Z + m_P * P + m_Z * Z + (1 - beta) * g_Z * Z
    dP = mu_P * P - g_Z * Z - m_P * P
    dZ = beta * g_Z * Z - m_Z * Z - gamma_Z * Z

    return np.array([dN, dP, dZ])

rk4_step = modelling.with_rk4(dxdt, autonom=False)

def step(x0, t0, dt):
    x_new = rk4_step(x0, t0, dt)
    return np.maximum(x_new, 1e-8)


##### colonne d'eau 1D en Z

## constances physiques
M = 20              # Nombre de couches
depth = 200         # Profondeur totale (m)
dz = depth / M      # épaisseur couche
# Kz = 0.7          # coefficient de diffusion verticale fixe (changé!)
k_ext = 0.02        # Coefficient d'atténuation de la lumière
z_levels = np.linspace(0, depth, M) # Profondeurs de chaque couche

P_seuil = 0.1       # Seuil de broutage (Grazing threshold)
N_deep = 10.0       # Réserve infinie de nutriments au fond

# paramètres de diffusion
Kzb = 0.1           # Diffusion au fond (stratifié, calme)
Kz0 = 10.0          # Diffusion en surface (brassé par le vent)
c_therm = 5.0 / depth   # Finesse de la thermocline

# Fonction MLD(t) : Profondeur de la couche de mélange
# mixed layer depth
def MLD(t):
    MLD_hiver = 0.8 * depth
    MLD_ete = 0.2 * depth
    moyenne = (MLD_hiver + MLD_ete) / 2.0
    amplitude = (MLD_hiver - MLD_ete) / 2.0
    return moyenne - amplitude * np.cos(2 * np.pi * t / 365)

def calcul_Kz(z, t):
    Mt = MLD(t)
    num = np.arctan(c_therm * (Mt - z)) - np.arctan(c_therm * (Mt - depth))
    den = np.arctan(c_therm * Mt) - np.arctan(c_therm * (Mt - depth))
    return Kzb + (Kz0 - Kzb) * (num / den)

# état initial
N0 = np.full(M, 10.0)
P0 = np.full(M, 0.1)
Z0 = np.full(M, 0.1) 
x0_1D = np.concatenate([N0, P0, Z0])

# version adaptée fortran
@modelling.ens_compatible
def dxdt_1D(x_flat, t):
    original_shape = x_flat.shape 
    x = x_flat.reshape(3, M, -1)
    x = np.maximum(x, 1e-8)
    N_conc, P_conc, Z_conc = x[0], x[1], x[2] 
    
    # Cycle saisonnier
    T_saison = 365 
    saison = 0.5 * (1 - np.cos(2 * np.pi * t / T_saison))
    mu_saisonnier = mu_max * (0.2 + 0.8 * saison)

    # Auto-ombrage - Self-shading : calcul de la biomasse cumulée au-dessus
    # k_ext = atténuation de l'eau, k_self = coefficient d'auto-ombrage
    k_self = 0.05 
    P_int = dz * np.cumsum(P_conc, axis=0)

    lumiere = np.exp(-k_ext * z_levels.reshape(M,1)-k_self * P_int)

    mu_P = mu_saisonnier * lumiere * (N_conc / (K_N + N_conc))
    # Le Zooplancton ne mange plus si P < P_seuil
    P_eff = np.maximum(P_conc - P_seuil, 0.0)
    g_Z = g_max * (P_eff / (K_P + P_eff))
    
    dN = -mu_P * P_conc + gamma_Z * Z_conc + m_P * P_conc + m_Z * Z_conc + (1 - beta) * g_Z * Z_conc
    dP = mu_P * P_conc - g_Z * Z_conc - m_P * P_conc
    dZ = beta * g_Z * Z_conc - m_Z * Z_conc - gamma_Z * Z_conc
    
    return np.array([dN, dP, dZ]).reshape(original_shape)


rk4_step_1D = modelling.with_rk4(dxdt_1D, autonom=False)

def step_1D(etat_precedent, t, dt, M, dz):
    """
    Calcule un pas de temps complet sans matrice globale de diffusion,
    en appliquant directement les différences finies explicites (comme Fortran).
    """
    x_bio_finale = rk4_step_1D(etat_precedent, t, dt) 
    Kz_array = calcul_Kz(z_levels, t)
    coeff = Kz_array / (dz**2)

    # modèle vérité
    if etat_precedent.ndim == 1:
        x_bio_finale = x_bio_finale.reshape(3, M)
        etat_initial = etat_precedent.reshape(3, M) 
        diffusion = np.zeros((3, M))
        
        for i in range(3):
            x = etat_initial[i]
            # éléments finis
            diffusion[i, 1:-1] = coeff[1:-1] * (x[2:] - 2 * x[1:-1] + x[:-2])
            diffusion[i, 0]  = coeff[0] * (x[1] - x[0])
            
            x_bio_finale[i] += dt * diffusion[i]
            
        # Conditions de Dirichlet au fond (réserve infinie)
        x_bio_finale[0, -1] = N_deep
        x_bio_finale[1, -1] = 0.0
        x_bio_finale[2, -1] = 0.0
            
        return np.maximum(x_bio_finale, 1e-8).flatten()
        
    # assimilation
    else:
        nb_membres = etat_precedent.shape[0]
        x_bio_finale = x_bio_finale.reshape(nb_membres, 3, M)
        etat_initial = etat_precedent.reshape(nb_membres, 3, M) 
        diffusion = np.zeros((nb_membres, 3, M))
        
        coeff_b = coeff.reshape(1, M)

        for i in range(3):
            x = etat_initial[:, i, :] 
            diffusion[:, i, 1:-1] = coeff_b[:, 1:-1] * (x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2])
            diffusion[:, i, 0]  = coeff_b[:, 0] * (x[:, 1] - x[:, 0])
            
            x_bio_finale[:, i, :] += dt * diffusion[:, i, :]
            
        x_bio_finale[:, 0, -1] = N_deep
        x_bio_finale[:, 1, -1] = 0.0
        x_bio_finale[:, 2, -1] = 0.0
            
        return np.maximum(x_bio_finale, 1e-8).reshape(nb_membres, -1)
