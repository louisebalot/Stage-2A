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

@modelling.ens_compatible
def dxdt(x):
    """
    Calcule les dérivées [dN/dt, dP/dt, dZ/dt] selon le modèle NPZ
    """
    x = np.maximum(x, 1e-8)
    N, P, Z = x
    
    mu_P = mu_max * (N / (K_N + N))
    g_Z = g_max * (P / (K_P + P))
    
    # Équations
    dN = -mu_P * P + gamma_Z * Z + m_P * P + m_Z * Z
    dP = mu_P * P - g_Z * Z - m_P * P
    dZ = beta * g_Z * Z - m_Z * Z - gamma_Z * Z
    
    return np.array([dN, dP, dZ])

step = modelling.with_rk4(dxdt, autonom=True)