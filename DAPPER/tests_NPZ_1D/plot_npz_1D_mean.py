import numpy as np
import dapper as dpr
from dapper.da_methods import EnKF, KETKF

from dapper.mods.NPZ.settings_1D import HMM_log, HMM_physique, M
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dapper.tools.progressbar as pb
pb.disable_progbar = True

#xx, yy = HMM.simulate()
xx_phys, yy_phys = HMM_physique.simulate()

N = 50
infl = 1.04
enkf = EnKF('Sqrt', N=N, infl=1.01, rot=True)
ketkf_lin = KETKF(N=N, kernel_type='linear', infl=infl, rot=True, reg_tikhonov=1e-10)
ketkf_hyp  = KETKF(N=N, infl=infl, rot=True, kernel_type='hyperbolique', c_tanh=1e-3, reg_tikhonov=1e-3)
ketkf_sig = KETKF(N=N, infl=infl, rot=True, kernel_type='sigmoid', c_tanh=0.01 , reg_tikhonov=1e-3)
ketkf_lap = KETKF(N=N, infl=infl, rot=True, kernel_type='lap', reg_tikhonov=1e-3)

xx_pour_stats = np.log(np.maximum(xx_phys, 1e-20))

enkf.assimilate(HMM_log, xx_pour_stats, yy_phys, liveplots=False)
ketkf_lin.assimilate(HMM_log, xx_pour_stats, yy_phys, liveplots=False)
ketkf_hyp.assimilate(HMM_log, xx_pour_stats, yy_phys, liveplots=False)
ketkf_sig.assimilate(HMM_log, xx_pour_stats, yy_phys, liveplots=False)
ketkf_lap.assimilate(HMM_log, xx_pour_stats, yy_phys, liveplots=False)

time_truth = HMM_physique.tseq.tt 
truth = xx_phys
# liste[début : fin : pas]
time_analysis = HMM_physique.tseq.tt[HMM_physique.tseq.dko :: HMM_physique.tseq.dko]

mu_enkf = enkf.stats.mu.a
mu_ketkf_lin = ketkf_lin.stats.mu.a
mu_ketkf_hyp = ketkf_hyp.stats.mu.a
mu_ketkf_sig = ketkf_sig.stats.mu.a
mu_ketkf_lap = ketkf_lap.stats.mu.a

mu_enkf = np.exp(mu_enkf)
mu_ketkf_lin = np.exp(mu_ketkf_lin)
mu_ketkf_hyp = np.exp(mu_ketkf_hyp)
mu_ketkf_sig = np.exp(mu_ketkf_sig)
mu_ketkf_lap = np.exp(mu_ketkf_lap)

t_obs = time_analysis
y_plot = [np.array(y).item() for y in yy_phys]

fig = make_subplots(
    rows=3, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.05,
    subplot_titles=('Nutriments (N) - Moyenne verticale', 'Phytoplancton (P) - Moyenne verticale', 'Zooplancton (Z) - Moyenne verticale')
)

for i in range(3):
    idx_start = i * M
    idx_end = (i + 1) * M
    
    show_leg = True if i == 0 else False
    
    fig.add_trace(go.Scatter(
        x=time_truth, y=np.mean(truth[:, idx_start:idx_end], axis=1), 
        mode='lines', 
        name='Vérité',
        line=dict(color='black', width=2, dash='dash'),
        legendgroup='verite',
        showlegend=show_leg
    ), row=i+1, col=1)
    
    fig.add_trace(go.Scatter(
        x=time_analysis, y=np.mean(mu_enkf[:, idx_start:idx_end], axis=1), 
        mode='lines', 
        name='EnKF',
        line=dict(color='orange', width=2),
        legendgroup='enkf',
        showlegend=show_leg
    ), row=i+1, col=1)
    
    fig.add_trace(go.Scatter(
        x=time_analysis, y=np.mean(mu_ketkf_lin[:, idx_start:idx_end], axis=1), 
        mode='lines', 
        name='KETKF: linear',
        line=dict(color='green', width=2),
        legendgroup='ketkf_lin',
        showlegend=show_leg
    ), row=i+1, col=1)

    fig.add_trace(go.Scatter(
        x=time_analysis, y=np.mean(mu_ketkf_hyp[:, idx_start:idx_end], axis=1), 
        mode='lines', 
        name='KETKF: hyperbolique',
        line=dict(color='blue', width=2),
        legendgroup='ketkf_hyp',
        showlegend=show_leg
    ), row=i+1, col=1)

    fig.add_trace(go.Scatter(
        x=time_analysis, y=np.mean(mu_ketkf_sig[:, idx_start:idx_end], axis=1), 
        mode='lines', 
        name='KETKF: sigmoid',
        line=dict(color='red', width=2),
        legendgroup='ketkf_sig',
        showlegend=show_leg
    ), row=i+1, col=1)

    fig.add_trace(go.Scatter(
        x=time_analysis, y=np.mean(mu_ketkf_lap[:, idx_start:idx_end], axis=1), 
        mode='lines', 
        name='KETKF: lap',
        line=dict(color='purple', width=2),
        legendgroup='ketkf_lap',
        showlegend=show_leg
    ), row=i+1, col=1)
    
    if i == 1:
        fig.add_trace(go.Scatter(
            x=t_obs, y=y_plot, 
            mode='markers', 
            name='Obs Satellites (Surface)',
            marker=dict(symbol='star', size=8, color='black'),
            showlegend=True
        ), row=2, col=1)

fig.update_layout(
    title="Comparaison des filtres sur la moyenne sur Z (Observations à la surface)",
    height=900,
    hovermode="x unified",
    template="plotly_white",
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02, 
        xanchor="right", x=1
    )
)
fig.update_xaxes(range=[0, 1000], title_text="Temps en jours", row=3, col=1)

fig.show()