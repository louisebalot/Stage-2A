import dapper as dpr
import pandas as pd
import numpy as np
from dapper.mods.NPZ.settings_1D import HMM
from dapper.da_methods import EnKF, KETKF
import time
import os
from contextlib import redirect_stdout
import dapper.tools.progressbar as pb
pb.disable_progbar = True

n_simulations = 10
results_list = []

N = 55
infl = 1.01
"""
def creer_filtres():
    return [EnKF('Sqrt', N=N, infl=infl, rot=True, truncated=True),
    EnKF('Sqrt', N=N, infl=infl, rot=True, log_transform=True),
    #EnKF('PertObs', N=N, infl=infl, rot=False, truncated=True),
    EnKF('DEnKF', N=N, infl=infl, rot=True, truncated=True),
    EnKF('DEnKF', N=N, infl=infl, rot=True, log_transform=True),
    KETKF(N=N, infl=infl, rot=True, kernel_type='linear', truncated=True),
    KETKF(N=N, infl=infl, rot=True, kernel_type='sigmoid', c_tanh=0.01, reg_tikhonov=1e-3, truncated=True),
    KETKF(N=N, infl=infl, rot=True, kernel_type='hyperbolique', c_tanh=1e-3, reg_tikhonov=1e-3, truncated=True),
    KETKF(N=N, infl=infl, rot=True, kernel_type='lap', reg_tikhonov=1e-3, truncated=True),
    KETKF(N=N, infl=infl, rot=True, kernel_type='linear', log_transform=True),
    KETKF(N=N, infl=infl, rot=True, kernel_type='sigmoid', c_tanh=0.01, reg_tikhonov=1e-3, log_transform=True),
    KETKF(N=N, infl=infl, rot=True, kernel_type='hyperbolique', c_tanh=1e-3, reg_tikhonov=1e-3, log_transform=True),
    KETKF(N=N, infl=infl, rot=True, kernel_type='lap', reg_tikhonov=1e-3, log_transform=True)]
    #KETKF(N=N, infl=infl, rot=True, kernel_type='polynomial', poly_degree=1, reg_tikhonov=1e-2, truncated=True),
    #KETKF(N=N, infl=infl, rot=True, kernel_type='rbf_exp', sigma_rbf=0.25, reg_tikhonov=1e-3),
    #KETKF(N=N, infl=infl, rot=True, kernel_type='rbf', sigma_rbf=0.5, reg_tikhonov=1e-3),
"""    
inflations = [1.001, 1.01, 1.05]
# Dictionnaire pour appliquer soit la troncature, soit le log_transform
positivites = [{'truncated': True}, {'log_transform': True}]

def creer_filtres():
    filtres = []
    
    for infl in inflations:
        for pos in positivites:
            
            # ==========================================
            # 1. FILTRES CLASSIQUES (EnKF, DEnKF, PertObs)
            # ==========================================
            filtres.append(EnKF('Sqrt', N=N, infl=infl, rot=True, **pos))
            filtres.append(EnKF('DEnKF', N=N, infl=infl, rot=True, **pos))
            filtres.append(EnKF('PertObs', N=N, infl=infl, rot=False, **pos))

            # ==========================================
            # 2. KETKF - NOYAU LINÉAIRE
            # ==========================================
            for reg in [1e-4, 1e-3, 1e-2]:
                filtres.append(KETKF(N=N, infl=infl, rot=True, kernel_type='linear', reg_tikhonov=reg, **pos))

            # ==========================================
            # 3. KETKF - NOYAU SIGMOÏDE
            # ==========================================
            for c_tanh in [1e-3, 0.01, 0.1]:
                for reg in [1e-4, 1e-3]:
                    filtres.append(KETKF(N=N, infl=infl, rot=True, kernel_type='sigmoid', c_tanh=c_tanh, reg_tikhonov=reg, **pos))

            # ==========================================
            # 4. KETKF - NOYAU HYPERBOLIQUE
            # ==========================================
            for c_tanh in [1e-4, 1e-3, 1e-2]:
                for reg in [1e-4, 1e-3]:
                    filtres.append(KETKF(N=N, infl=infl, rot=True, kernel_type='hyperbolique', c_tanh=c_tanh, reg_tikhonov=reg, **pos))

            # ==========================================
            # 5. KETKF - NOYAU LAPLACIEN
            # ==========================================
            for reg in [1e-4, 1e-3, 1e-2]:
                filtres.append(KETKF(N=N, infl=infl, rot=True, kernel_type='lap', reg_tikhonov=reg, **pos))

            # ==========================================
            # 6. KETKF - NOYAUX RBF & RBF EXP
            # ==========================================
            for sigma in [0.1, 0.25, 0.5]:
                for reg in [1e-4, 1e-3]:
                    filtres.append(KETKF(N=N, infl=infl, rot=True, kernel_type='rbf', sigma_rbf=sigma, reg_tikhonov=reg, **pos))
                    filtres.append(KETKF(N=N, infl=infl, rot=True, kernel_type='rbf_exp', sigma_rbf=sigma, reg_tikhonov=reg, **pos))

            # ==========================================
            # 7. KETKF - NOYAU POLYNOMIAL
            # ==========================================
            for degree in [1, 2, 3]:
                for reg in [1e-3, 1e-2]:
                    filtres.append(KETKF(N=N, infl=infl, rot=True, kernel_type='polynomial', poly_degree=degree, reg_tikhonov=reg, **pos))
                    
    return filtres
"""
def nom_filtre(f):
    try:
        return "KETKF-" + f.kernel_type + " -- log_transform = " + str(f.log_transform)
    except AttributeError:
        return "EnKF-" + f.upd_a + " -- log_transform = " + str(f.log_transform)
"""    
def nom_filtre(f):
    if hasattr(f, 'kernel_type'):
        nom = f"KETKF-{f.kernel_type}"
    else:
        nom = f"EnKF-{getattr(f, 'upd_a', 'Filtre')}"
    
    params = []
    
    if getattr(f, 'infl', None) is not None:
        params.append(f"infl={f.infl}")
    if getattr(f, 'c_tanh', None) is not None:
        params.append(f"c={f.c_tanh}")
    if getattr(f, 'reg_tikhonov', None) is not None:
        params.append(f"reg={f.reg_tikhonov}")
    if getattr(f, 'sigma_rbf', None) is not None:
        params.append(f"sigma={f.sigma_rbf}")
    if getattr(f, 'poly_degree', None) is not None:
        params.append(f"deg={f.poly_degree}")
        
    if getattr(f, 'log_transform', False):
        params.append("log_tr=True")
    elif getattr(f, 'truncated', False):
        params.append("trunc=True")
        
    if not getattr(f, 'rot', True):
        params.append("rot=False")

    return f"{nom} ({', '.join(params)})"

noms = [nom_filtre(f) for f in creer_filtres()]

rmse  = {nom: [] for nom in noms}
rmv   = {nom: [] for nom in noms}
times = {nom: [] for nom in noms}


for k in range(n_simulations):
    print(f"Simulation {k+1}/{n_simulations}")

    xx, yy = HMM.simulate()

    for f in creer_filtres():
        nom = nom_filtre(f)
        t0 = time.time()
        
        try:
            # On tente l'assimilation
            f.assimilate(HMM, xx, yy, liveplots=False)
            t1 = time.time()
            
            # Si ça passe, on récupère les stats
            val_rmse = np.nanmean(f.stats.err.rms.a)
            val_rmv  = np.nanmean(f.stats.spread.rms.a)
            temps_ecoule = t1 - t0
            
        except Exception as e:
            # Si ça plante (LinAlgError, overflow, etc.)
            print(f"  [!] Le filtre {nom} a divergé/planté : {type(e).__name__}")
            val_rmse = np.nan
            val_rmv  = np.nan
            temps_ecoule = np.nan
            
        if hasattr(f, 'rang_history'):
            delattr(f, 'rang_history')

        rmse[nom].append(val_rmse)
        rmv[nom].append(val_rmv)
        times[nom].append(temps_ecoule)


rmse_avg = {k: np.nanmean(v) for k, v in rmse.items()}
rmse_std = {k: np.nanstd(v) for k, v in rmse.items()}
rmv_avg  = {k: np.nanmean(v) for k, v in rmv.items()}
time_avg = {k: np.nanmean(v) for k, v in times.items()}

df = pd.DataFrame({
    "RMSE moyen": rmse_avg,
    "RMSE std": rmse_std,
    "RMV moyen": rmv_avg,
    "Temps moyen (s)": time_avg
})

df_trie = df.sort_values(by="RMSE moyen")

print(f"\nRésultats moyennés sur {n_simulations} simulations, avec N = {N}")
print(df_trie)

# ==========================================
# SAUVEGARDE DANS UN FICHIER TXT
# ==========================================
nom_fichier = "resultats_filtres.txt"
with open(nom_fichier, "w", encoding="utf-8") as f:
    f.write(f"Résultats moyennés sur {n_simulations} simulations, avec N = {N}\n")
    f.write("="*80 + "\n")
    f.write(df_trie.to_string())

print(f"\nLes résultats ont été sauvegardés avec succès dans le fichier : {nom_fichier}")