import numpy as np
import dapper as dpr
import dapper.mods as modelling
from dapper.mods.NPZ import step_1D_augmented_log, Nx, Np
TRUTH_m_P = 0.07
GUESS_m_P = 0.10

TRUTH_log_m_P = np.log(TRUTH_m_P)
GUESS_log_m_P = np.log(GUESS_m_P)

chemin = '/home/louise/Bureau/Stage/Stage-2A/source stage/DAPPER/etat_stable_5ans.npy'
x_stable = np.load(chemin)

def X0(log_param_mean, log_param_var):
    mu_aug = np.concatenate([x_stable, [log_param_mean]])
    C_aug = np.concatenate([np.full(Nx, 0.05**2), [log_param_var]]) 
    return modelling.GaussRV(mu=mu_aug, C=C_aug)

def set_X0_and_simulate(hmm, xp):
    dpr.set_seed(3000)
    hmm.X0 = X0(TRUTH_log_m_P, 0.0)
    xx, yy = hmm.simulate()
    hmm.X0 = X0(GUESS_log_m_P, 0.2**2)
    return hmm, xx, yy

Dyn = {
    'M': Nx + Np, 
    'model': step_1D_augmented_log, 
    'noise': 0
}

jj = np.arange(Nx)
obs_dict = modelling.partial_Id_Obs(Nx + Np, jj)
obs_dict["noise"] = 0.01
Obs = modelling.Operator(**obs_dict)

tseq = modelling.Chronology(dt=0.1, dko=10, Ko=1825, BurnIn=365)
parts = dict(state=np.arange(Nx), param=np.arange(Np) + Nx)

HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0=X0(GUESS_log_m_P, 0.2**2), sectors=parts)