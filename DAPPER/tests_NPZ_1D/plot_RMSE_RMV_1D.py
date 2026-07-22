import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from dapper.mods.NPZ.settings_1D import HMM, M
from dapper.da_methods import EnKF, KETKF
import dapper.tools.progressbar as pb

pb.disable_progbar = True

xx, yy = HMM.simulate()

t_obs = HMM.tseq.tt[HMM.tseq.dko :: HMM.tseq.dko]

N = 50
infl = 1.04

def creer_filtres():
    return [
        EnKF('Sqrt', N=N, infl=1.001, rot=True, truncated=True),
        KETKF(N=N, infl=1.001, rot=True, kernel_type='linear', log_transform=True),
        KETKF(N=N, infl=infl, rot=True, kernel_type='sigmoid', c_tanh=0.1, reg_tikhonov=1e-3, log_transform=True),
        KETKF(N=N, infl=infl, rot=True, kernel_type='hyperbolique', c_tanh=1e-3, reg_tikhonov=1e-2, log_transform=True),
        #KETKF(N=N, infl=infl, rot=True, kernel_type='polynomial', poly_degree=1, reg_tikhonov=1e-2),
        #KETKF(N=N, infl=infl, rot=True, kernel_type='rbf_exp', sigma_rbf=0.25, reg_tikhonov=1e-3),
        #KETKF(N=N, infl=infl, rot=True, kernel_type='rbf', sigma_rbf=0.5, reg_tikhonov=1e-3),
        KETKF(N=N, infl=infl, rot=True, kernel_type='lap', reg_tikhonov=1e-3, log_transform=True)
    ]

def nom_filtre(f):
    try:
        return "KETKF-" + f.kernel_type
    except AttributeError:
        return "EnKF-Sqrt"


fig = make_subplots(
    rows=5, cols=1, 
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=(
        "RMSE des Nutriments au cours du temps", 
        "RMSE du Phytoplancton au cours du temps", 
        "RMSE du Zooplancton au cours du temps", 
        "RMV au cours du temps",
        "Rang de la Matrice de Gram K"
    )
)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf']

for i, xp in enumerate(creer_filtres()):
    
    nom_algo = nom_filtre(xp)
    print(f"Assimilation en cours pour {nom_algo}")
    
    xp.assimilate(HMM, xx, yy, liveplots=False)

    erreur = xp.stats.mu.a - xx[HMM.tseq.dko :: HMM.tseq.dko]

    rmse_t_N = rmse_N_t = np.sqrt(np.mean(erreur[:, :M]**2, axis=1))
    rmse_t_P = rmse_N_t = np.sqrt(np.mean(erreur[:, M:2*M]**2, axis=1))
    rmse_t_Z = rmse_N_t = np.sqrt(np.mean(erreur[:, 2*M:]**2, axis=1))

    rmv_t = xp.stats.spread.rms.a

    if hasattr(xp, 'rang_history'):
        jours_sautes = len(t_obs) - len(xp.rang_history)
        
        rang_t = [np.nan] * jours_sautes + xp.rang_history
    else:
        # L'EnKF n'a pas de matrice de Gram K
        rang_t = [np.nan] * len(t_obs)
    
    color = colors[i % len(colors)]
    
    fig.add_trace(go.Scatter(
        x=t_obs, y=rmse_t_N, 
        mode='lines', name=f"{nom_algo}",
        line=dict(color=color, width=2), legendgroup=nom_algo
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t_obs, y=rmse_t_P, 
        mode='lines', name=f"{nom_algo}",
        line=dict(color=color, width=2), legendgroup=nom_algo,
        showlegend=False
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=t_obs, y=rmse_t_Z, 
        mode='lines', name=f"{nom_algo}",
        line=dict(color=color, width=2), legendgroup=nom_algo,
        showlegend=False
    ), row=3, col=1)
    
    fig.add_trace(go.Scatter(
        x=t_obs, y=rmv_t, 
        mode='lines', name=f"{nom_algo}",
        line=dict(color=color, width=2, dash='dash'), legendgroup=nom_algo,
        showlegend=False
    ), row=4, col=1)

    fig.add_trace(go.Scatter(
        x=t_obs, y=rang_t, 
        mode='lines', name=f"{nom_algo}",
        line=dict(color=color, width=2, dash='dot'), legendgroup=nom_algo,
        showlegend=False
    ), row=5, col=1)

fig.update_layout(
    title=f"Analyse de l'effondrement : impact des noyaux sur le rang et l'erreur, N = {N} et inflation = {infl}",
    height=1500,
    hovermode="x unified",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

fig.add_hline(y=N, row=5, col=1, line_dash="dash", line_color="gray", 
              annotation_text=f"N={N}", annotation_position="top left")

fig.update_xaxes(title_text="Temps (Jours)", row=5, col=1)
fig.update_xaxes(title_text="Temps (Jours)", row=3, col=1, showticklabels=True, ticks="outside")
fig.update_yaxes(title_text="RMSE Nutriments", row=1, col=1)
fig.update_yaxes(title_text="RMSE Phytoplancton", row=2, col=1)
fig.update_yaxes(title_text="RMSE Zooplancton", row=3, col=1)
fig.update_yaxes(title_text="RMV", row=4, col=1)
fig.update_yaxes(title_text="Rang (K)", row=5, col=1)

jours_max = int(max(t_obs))
nb_annees = (jours_max // 365)

for annee in range(nb_annees):
    debut_annee = annee * 365
    
    # Zone Printemps / Bloom
    fig.add_vrect(
        x0=debut_annee + 75, x1=debut_annee + 150,
        fillcolor="limegreen", opacity=0.15, 
        layer="below", line_width=0,
        annotation_text="Printemps (Bloom)" if annee == 0 else "", 
        annotation_position="top left",
        row="all", col=1
    )
    
    # Zone Hiver début
    fig.add_vrect(
        x0=debut_annee, x1=debut_annee + 60,
        fillcolor="dodgerblue", opacity=0.15, 
        layer="below", line_width=0,
        annotation_text="Hiver (Mélange)" if annee == 0 else "", 
        annotation_position="top left",
        row="all", col=1
    )
    
    # Zone Hiver
    fig.add_vrect(
        x0=debut_annee + 330, x1=debut_annee + 365,
        fillcolor="dodgerblue", opacity=0.15, 
        layer="below", line_width=0,
        row="all", col=1
    )

fig.show()