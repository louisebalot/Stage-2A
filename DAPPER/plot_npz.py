import numpy as np
import dapper as dpr
from dapper.da_methods import EnKF, KETKF

from dapper.mods.NPZ.settings_1 import HMM
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("Génération de la vérité et des observations...")
xx, yy = HMM.simulate()

enkf = EnKF('Sqrt', N=10, infl=1.02, rot=True)
ketkf = KETKF(N=10, kernel_type='linear', infl=1.02, rot=True, reg_tikhonov=1e-10)

enkf.assimilate(HMM, xx, yy, liveplots=False)
ketkf.assimilate(HMM, xx, yy, liveplots=False)

time_truth = HMM.tseq.tt 
truth = xx

time_analysis = HMM.tseq.tt[HMM.tseq.dko :: HMM.tseq.dko]

mu_enkf = enkf.stats.mu.a
mu_ketkf = ketkf.stats.mu.a

t_obs = time_analysis
y_plot = yy.flatten()

fig = make_subplots(
    rows=3, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.05,
    subplot_titles=('Nutriments (N)', 'Phytoplancton (P)', 'Zooplancton (Z)')
)

colors_ketkf = ['blue', 'green', 'red']
variables = ['N', 'P', 'Z']

for i in range(3):
    show_leg = True if i == 0 else False
    
    fig.add_trace(go.Scatter(
        x=time_truth, y=truth[:, i], 
        mode='lines', 
        name='Vérité (Nature)',
        line=dict(color='black', width=2, dash='dash'),
        legendgroup='verite',
        showlegend=show_leg
    ), row=i+1, col=1)
    
    fig.add_trace(go.Scatter(
        x=time_analysis, y=mu_enkf[:, i], 
        mode='lines', 
        name='EnKF (Classique)',
        line=dict(color='orange', width=2),
        legendgroup='enkf',
        showlegend=show_leg
    ), row=i+1, col=1)
    
    fig.add_trace(go.Scatter(
        x=time_analysis, y=mu_ketkf[:, i], 
        mode='lines', 
        name='KETKF (Ton Filtre)',
        line=dict(color=colors_ketkf[i], width=2),
        legendgroup='ketkf',
        showlegend=show_leg
    ), row=i+1, col=1)
    
    if i == 1:
        fig.add_trace(go.Scatter(
            x=t_obs, y=y_plot, 
            mode='markers', 
            name='Obs Satellites',
            marker=dict(symbol='star', size=8, color='black'),
            showlegend=True
        ), row=2, col=1)

fig.update_layout(
    title="Assimilation de données : Découverte du Zooplancton caché",
    height=900,
    hovermode="x unified",
    template="plotly_white",
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02, 
        xanchor="right", x=1
    )
)
fig.update_xaxes(title_text="Temps", row=3, col=1)

fig.show()