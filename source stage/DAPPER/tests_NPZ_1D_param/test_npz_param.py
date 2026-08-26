import dapper as dpr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from dapper.da_methods import EnKF, KETKF

from dapper.mods.NPZ.settings_1D_param import HMM, set_X0_and_simulate
from dapper.mods.NPZ import Np
import dapper.tools.progressbar as pb
pb.disable_progbar = True
    
N = 55  

def creer_filtres():
    return [
        KETKF(N=N, infl=1.001, rot=True, kernel_type='hyperbolique', c_tanh=0.0001, reg_tikhonov=1e-4, log_transform=True, Np=Np),
        KETKF(N=N, infl=1.01, rot=True, kernel_type='linear', reg_tikhonov=1e-3, log_transform=True, Np=Np),
        KETKF(N=N, infl=1.01, rot=True, kernel_type='lap', reg_tikhonov=1e-2, log_transform=True, Np=Np),
        EnKF('PertObs', N=N, infl=1.001, rot=False, truncated=True),
        KETKF(N=N, infl=1.01, rot=True, kernel_type='sigmoid', c_tanh=0.001, reg_tikhonov=1e-3, log_transform=True, Np=Np),
        EnKF('DEnKF', N=N, infl=1.001, rot=True, truncated=True),
        EnKF('Sqrt', N=N, infl=1.001, rot=True, truncated=True),
        KETKF(N=N, infl=1.05, rot=True, kernel_type='rbf', sigma_rbf=0.5, reg_tikhonov=1e-3, log_transform=True, Np=Np),
        KETKF(N=N, infl=1.05, rot=True, kernel_type='rbf_exp', sigma_rbf=0.5, reg_tikhonov=1e-4, log_transform=True, Np=Np),
        KETKF(N=N, infl=1.05, rot=True, kernel_type='polynomial', poly_degree=3, reg_tikhonov=1e-3, log_transform=True, Np=Np)
    ]

def nom_filtre(f):
    nom = f"KETKF-{f.kernel_type}" if hasattr(f, 'kernel_type') else f"EnKF-{getattr(f, 'upd_a', 'Filtre')}"
    params = []
    if getattr(f, 'infl', None) is not None: params.append(f"infl={f.infl}")
    if getattr(f, 'c_tanh', None) is not None: params.append(f"c={f.c_tanh}")
    if getattr(f, 'reg_tikhonov', None) is not None: params.append(f"reg={f.reg_tikhonov}")
    if getattr(f, 'sigma_rbf', None) is not None: params.append(f"sigma={f.sigma_rbf}")
    if getattr(f, 'poly_degree', None) is not None: params.append(f"deg={f.poly_degree}")
    if getattr(f, 'log_transform', False): params.append("log_tr=True")
    elif getattr(f, 'truncated', False): params.append("trunc=True")
    if not getattr(f, 'rot', True): params.append("rot=False")
    return f"{nom} ({', '.join(params)})"

noms = [nom_filtre(f) for f in creer_filtres()]

rmse_state = {nom: [] for nom in noms}
rmse_param = {nom: [] for nom in noms}
rmv = {nom: [] for nom in noms}
times = {nom: [] for nom in noms}
HMM, xx, yy = set_X0_and_simulate(HMM, None)

for f in creer_filtres():
    nom = nom_filtre(f)
    
    t0 = time.time()

    try:
        f.assimilate(HMM, xx, yy, liveplots=False)
        t1 = time.time()
        
        mu_analyse = f.stats.mu.a
        
        erreur_etat = mu_analyse[:, :-1] - xx[:, :-1]
        val_rmse_s = np.sqrt(np.nanmean(erreur_etat**2))
        
        m_P_estime = np.exp(mu_analyse[:, -1])
        m_P_vrai   = np.exp(xx[:, -1])
        val_rmse_p = np.sqrt(np.nanmean((m_P_estime - m_P_vrai)**2))
        
        val_rmv = np.nanmean(f.stats.spread.rms.a)
        
        temps_ecoule = t1 - t0
        
    except Exception as e:
        print(f"Le filtre {nom} a divergé : {type(e).__name__}")
        val_rmse_s, val_rmse_p, val_rmv, temps_ecoule = np.nan, np.nan, np.nan, np.nan

    rmse_state[nom].append(val_rmse_s)
    rmse_param[nom].append(val_rmse_p)
    rmv[nom].append(val_rmv)
    times[nom].append(temps_ecoule)

rmse_state_avg = {k: np.nanmean(v) for k, v in rmse_state.items()}
rmse_param_avg = {k: np.nanmean(v) for k, v in rmse_param.items()}
rmse_state_std = {k: np.nanstd(v) for k, v in rmse_state.items()}
rmse_param_std = {k: np.nanstd(v) for k, v in rmse_param.items()}
rmv_avg  = {k: np.nanmean(v) for k, v in rmv.items()}
time_avg = {k: np.nanmean(v) for k, v in times.items()}

df = pd.DataFrame({
    "RMSE moyen états": rmse_state_avg,
    "RMSE std états": rmse_state_std,
    "RMSE moyen paramètres": rmse_param_avg,
    "RMSE std paramètres": rmse_param_std,
    "RMV moyen": rmv_avg,
    "Temps moyen (s)": time_avg
})

df_trie = df.sort_values(by="RMSE moyen paramètres")

print(f"\nRésultats moyennés avec N = {N}")
print(df_trie.to_string()) 