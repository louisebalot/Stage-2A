import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from dapper.mods.NPZ.settings_1D_param import HMM
from dapper.mods.NPZ import Np, M
from dapper.da_methods import EnKF, KETKF
import dapper.tools.progressbar as pb

pb.disable_progbar = True

xx, yy = HMM.simulate()

t_obs = HMM.tseq.tt[HMM.tseq.dko :: HMM.tseq.dko]

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
    ]

def nom_filtre(f):
    if hasattr(f, 'kernel_type'):
        return "KETKF-" + f.kernel_type
    else:
        return f"EnKF-{getattr(f, 'upd_a', 'Sqrt')}"


fig = make_subplots(
    rows=6, cols=1, 
    shared_xaxes=True,
    vertical_spacing=0.03,
    subplot_titles=(
        "RMSE des Nutriments", 
        "RMSE du Phytoplancton", 
        "RMSE du Zooplancton", 
        "RMSE des Paramètres (physique)",
        "RMV au cours du temps",
        "Rang de la Matrice de Gram K"
    )
)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf']

dko = HMM.tseq.dko
xx_obs = xx[dko::dko]

for i, xp in enumerate(creer_filtres()):
    
    nom_algo = nom_filtre(xp)
    print(f"Assimilation en cours pour {nom_algo}")
    
    xp.assimilate(HMM, xx, yy, liveplots=False)

    mu_analyse = xp.stats.mu.a

    erreur_etat = mu_analyse[:, :-Np] - xx_obs[:, :-Np]
    
    rmse_t_N = np.sqrt(np.mean(erreur_etat[:, :M]**2, axis=1))
    rmse_t_P = np.sqrt(np.mean(erreur_etat[:, M:2*M]**2, axis=1))
    rmse_t_Z = np.sqrt(np.mean(erreur_etat[:, 2*M:]**2, axis=1))

    params_estimes = np.exp(mu_analyse[:, -Np:])
    params_vrais   = np.exp(xx_obs[:, -Np:])
    rmse_t_params = np.sqrt(np.mean((params_estimes - params_vrais)**2, axis=1))

    rmv_t = xp.stats.spread.rms.a

    if hasattr(xp, 'rang_history'):
        jours_sautes = len(t_obs) - len(xp.rang_history)
        rang_t = [np.nan] * jours_sautes + xp.rang_history
    else:
        rang_t = [np.nan] * len(t_obs)
    
    color = colors[i % len(colors)]
    
    fig.add_trace(go.Scatter(x=t_obs, y=rmse_t_N, mode='lines', name=nom_algo,
        line=dict(color=color, width=2), legendgroup=nom_algo), row=1, col=1)

    fig.add_trace(go.Scatter(x=t_obs, y=rmse_t_P, mode='lines', name=nom_algo,
        line=dict(color=color, width=2), legendgroup=nom_algo, showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(x=t_obs, y=rmse_t_Z, mode='lines', name=nom_algo,
        line=dict(color=color, width=2), legendgroup=nom_algo, showlegend=False), row=3, col=1)

    fig.add_trace(go.Scatter(x=t_obs, y=rmse_t_params, mode='lines', name=nom_algo,
        line=dict(color=color, width=2, dash='dashdot'), legendgroup=nom_algo, showlegend=False), row=4, col=1)
    
    fig.add_trace(go.Scatter(x=t_obs, y=rmv_t, mode='lines', name=nom_algo,
        line=dict(color=color, width=2, dash='dash'), legendgroup=nom_algo, showlegend=False), row=5, col=1)

    fig.add_trace(go.Scatter(x=t_obs, y=rang_t, mode='lines', name=nom_algo,
        line=dict(color=color, width=2, dash='dot'), legendgroup=nom_algo, showlegend=False), row=6, col=1)

fig.update_layout(
    title=f"Simulation avec estimation de paramètres (Np = {Np}), N = {N}",
    height=1700,
    hovermode="x unified",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

fig.add_hline(y=N, row=6, col=1, line_dash="dash", line_color="gray", 
              annotation_text=f"N={N}", annotation_position="top left")

fig.update_xaxes(title_text="Temps (Jours)", row=6, col=1)
fig.update_yaxes(title_text="RMSE Nutriments", row=1, col=1)
fig.update_yaxes(title_text="RMSE Phytoplancton", row=2, col=1)
fig.update_yaxes(title_text="RMSE Zooplancton", row=3, col=1)
fig.update_yaxes(title_text="RMSE Paramètres", row=4, col=1)
fig.update_yaxes(title_text="RMV", row=5, col=1)
fig.update_yaxes(title_text="Rang (K)", row=6, col=1)

jours_max = int(max(t_obs))
nb_annees = (jours_max // 365)

for annee in range(nb_annees):
    debut_annee = annee * 365
    
    fig.add_vrect(x0=debut_annee + 75, x1=debut_annee + 150,
        fillcolor="limegreen", opacity=0.15, layer="below", line_width=0,
        annotation_text="Printemps" if annee == 0 else "", annotation_position="top left", row="all", col=1)
    
    fig.add_vrect(x0=debut_annee, x1=debut_annee + 60,
        fillcolor="dodgerblue", opacity=0.15, layer="below", line_width=0,
        annotation_text="Hiver" if annee == 0 else "", annotation_position="top left", row="all", col=1)
    
    fig.add_vrect(x0=debut_annee + 330, x1=debut_annee + 365,
        fillcolor="dodgerblue", opacity=0.15, layer="below", line_width=0, row="all", col=1)

fig.show()