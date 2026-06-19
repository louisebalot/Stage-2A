import numpy as np
import dapper as dpr
from dapper.da_methods import EnKF, KETKF

from dapper.mods.NPZ.settings_1 import HMM
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("Génération de la vérité et des observations...")
xx, yy = HMM.simulate()

enkf = EnKF('Sqrt', N=30, infl=1.01, rot=True)
ketkf_lin = KETKF(N=30, kernel_type='linear', infl=1.02, rot=True, reg_tikhonov=1e-10)
ketkf_hyp  = KETKF(N=30, infl=1.01, rot=True, kernel_type='hyperbolique', c_tanh=1e-3, reg_tikhonov=1e-3)
ketkf_sig = KETKF(N=30, infl=1.01, rot=True, kernel_type='sigmoid', c_tanh=0.01 , reg_tikhonov=1e-3)

enkf.assimilate(HMM, xx, yy, liveplots=False)
ketkf_lin.assimilate(HMM, xx, yy, liveplots=False)
ketkf_hyp.assimilate(HMM, xx, yy, liveplots=False)
ketkf_sig.assimilate(HMM, xx, yy, liveplots=False)

time_truth = HMM.tseq.tt 
truth = xx
# liste[début : fin : pas]
time_analysis = HMM.tseq.tt[HMM.tseq.dko :: HMM.tseq.dko]

mu_enkf = enkf.stats.mu.a
mu_ketkf_lin = ketkf_lin.stats.mu.a
mu_ketkf_hyp = ketkf_hyp.stats.mu.a
mu_ketkf_sig = ketkf_sig.stats.mu.a

t_obs = time_analysis
y_plot = [np.array(y).item() for y in yy]

fig = make_subplots(
    rows=3, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.05,
    subplot_titles=('Nutriments (N)', 'Phytoplancton (P)', 'Zooplancton (Z)')
)

variables = ['N', 'P', 'Z']

for i in range(3):
    show_leg = True if i == 0 else False
    
    fig.add_trace(go.Scatter(
        x=time_truth, y=truth[:, i], 
        mode='lines', 
        name='Vérité',
        line=dict(color='black', width=2, dash='dash'),
        legendgroup='verite',
        showlegend=show_leg
    ), row=i+1, col=1)
    
    fig.add_trace(go.Scatter(
        x=time_analysis, y=mu_enkf[:, i], 
        mode='lines', 
        name='EnKF',
        line=dict(color='orange', width=2),
        legendgroup='enkf',
        showlegend=show_leg
    ), row=i+1, col=1)
    
    fig.add_trace(go.Scatter(
        x=time_analysis, y=mu_ketkf_lin[:, i], 
        mode='lines', 
        name='KETKF : '+ketkf_lin.kernel_type,
        line=dict(color='green', width=2),
        legendgroup='ketkf_lin',
        showlegend=show_leg
    ), row=i+1, col=1)

    fig.add_trace(go.Scatter(
        x=time_analysis, y=mu_ketkf_hyp[:, i], 
        mode='lines', 
        name='KETKF : '+ketkf_hyp.kernel_type,
        line=dict(color='blue', width=2),
        legendgroup='ketkf_hyp',
        showlegend=show_leg
    ), row=i+1, col=1)

    fig.add_trace(go.Scatter(
        x=time_analysis, y=mu_ketkf_sig[:, i], 
        mode='lines', 
        name='KETKF : '+ketkf_sig.kernel_type,
        line=dict(color='red', width=2),
        legendgroup='ketkf_sig',
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
    title="Modèle NPZ",
    height=900,
    hovermode="x unified",
    template="plotly_white",
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02, 
        xanchor="right", x=1
    )
)
fig.update_xaxes(range=[0, 2000])
fig.update_xaxes(title_text="Temps", row=3, col=1)

fig.show()