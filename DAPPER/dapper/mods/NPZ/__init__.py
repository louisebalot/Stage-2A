import numpy as np
import dapper.mods as modelling

# constantes
mu_max = 1.0
K_N = 0.5
g_max = 2.0
K_P = 0.5
m_P = 0.1
m_Z = 0.2
beta = 0.6
gamma_Z = 0.1

# État initial [N, P, Z]
x0 = np.array([2.0, 0.5, 0.1])

## réglages 1D 
M = 10             # Nombre de couches
depth = 50         # Profondeur totale (m)
dz = depth / M      # épaisseur couche
Kz = 0.7            # coefficient de diffusion verticale
k_ext = 0.03        # Coefficient d'atténuation de la lumière
z_levels = np.linspace(0, depth, M) # Profondeurs de chaque couche

N0 = np.linspace(1.0, 3.0, M)
P0 = np.linspace(0.8, 0.2, M)
Z0 = np.full(M, 0.1)
x0_1D = np.concatenate([N0, P0, Z0])


@modelling.ens_compatible
def dxdt(x, t):

    x = np.maximum(x, 1e-8)
    N, P, Z = x
    
    # simulation cycles
    T_saison = 1000 
    # Onde sinusoïdale qui oscille entre 0 (Hiver) et 1 (Été)
    saison = 0.5 * (1 + np.sin(2 * np.pi * t / T_saison))
    # Le taux de croissance varie de 20% en hiver à 100% en été
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

# mélange turbulent
def get_diffusion_matrix(M, dz, Kz):
    """
    Crée une matrice de diffusion tridiagonale pour M couches.
    dz : épaisseur d'une couche.
    Kz : coefficient de diffusion verticale.
    """
    diag_val = -2.0 * Kz / dz**2
    off_val = 1.0 * Kz / dz**2
    
    # Matrice tridiagonale
    D = np.diag([diag_val] * M) + \
        np.diag([off_val] * (M - 1), k=1) + \
        np.diag([off_val] * (M - 1), k=-1)
    
    # Conditions aux limites Neumann
    D[0, 0] = -1.0 * Kz / dz**2
    D[-1, -1] = -1.0 * Kz / dz**2
    
    return D

@modelling.ens_compatible
def dxdt_1D(x_flat, t):
    original_shape = x_flat.shape 
    
    x = x_flat.reshape(3, M, -1)
    x = np.maximum(x, 1e-8)
    N_conc, P_conc, Z_conc = x[0], x[1], x[2] 
    
    # cyclage
    T_saison = 1000 
    saison = 0.5 * (1 + np.sin(2 * np.pi * t / T_saison))
    mu_saisonnier = mu_max * (0.2 + 0.8 * saison)

    # loi beer-lambert
    lumiere = np.exp(-k_ext * z_levels).reshape(M, 1)

    mu_P = mu_saisonnier * lumiere * (N_conc / (K_N + N_conc))
    g_Z = g_max * (P_conc / (K_P + P_conc))
    
    dN = -mu_P * P_conc + gamma_Z * Z_conc + m_P * P_conc + m_Z * Z_conc + (1 - beta) * g_Z * Z_conc
    dP = mu_P * P_conc - g_Z * Z_conc - m_P * P_conc
    dZ = beta * g_Z * Z_conc - m_Z * Z_conc - gamma_Z * Z_conc
    
    return np.array([dN, dP, dZ]).reshape(original_shape)

rk4_step_1D = modelling.with_rk4(dxdt_1D, autonom=False)
D = get_diffusion_matrix(M, dz, Kz)


def step_1D(etat_precedent, t, dt, M, D):

    x_bio_finale = rk4_step_1D(etat_precedent, t, dt) 
    
    # vérité
    if etat_precedent.ndim == 1:
        x_bio_finale = x_bio_finale.reshape(3, M)
        etat_initial = etat_precedent.reshape(3, M) 
        
        # diffusion verticale = D @ etat_initial
        for i in range(3):
            diffusion = D @ etat_initial[i]
            x_bio_finale[i] += dt * diffusion
            
        # concentrations positives
        return np.maximum(x_bio_finale, 1e-8).flatten()
        
    # assimilation
    else:
        nb_membres = etat_precedent.shape[0]
        
        # Dimensions : (Membres, 3, Couches_Verticales)
        x_bio_finale = x_bio_finale.reshape(nb_membres, 3, M)
        etat_initial = etat_precedent.reshape(nb_membres, 3, M) 
        
        for i in range(3):
            diffusion = (D @ etat_initial[:, i, :].T).T
            x_bio_finale[:, i, :] += dt * diffusion
            
        return np.maximum(x_bio_finale, 1e-8).reshape(nb_membres, -1)