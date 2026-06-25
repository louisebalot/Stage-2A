from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
from scipy.spatial.distance import cdist
from numpy import diag, eye, sqrt, zeros, ones, mean, std

import dapper as dpr
from dapper.tools.progressbar import progbar
from dapper.stats import inflate_ens

from . import da_method

@dataclass
class KETKF(da_method):
    """Kernel Ensemble Transform Kalman Filter"""

    N: int          # Taille de l'ensemble
    infl: float     # Facteur d'inflation
    rot: bool       # Rotation aléatoire

    # pour le kernel
    kernel_type: str         # 'linear', 'polynomial', 'sigmoid', 'rbf', 'rbf_exp' ou 'hyperbolique'
    poly_degree: int = 2     # Degré du noyau polynomial
    sigma_rbf: float = 1.0   # Pour les noyaux basés sur la distance
    c_tanh: float = 1.0      # Pour le tanh
    reg_tikhonov: float = 1e-15

    def phi_poincare(self, X):
        coef = sqrt(self.c_tanh) * np.linalg.norm(X, axis=1, keepdims=True)

        # sécurité
        coef[coef == 0] = 1e-16
        coef_boule = np.clip(coef, -0.9999, 0.9999)

        return X * (np.arctanh(coef_boule) / coef) 


    def compute_kernel(self, X, Y):
        """
        Calcule la matrice de Gram selon le kernel_type choisi
        """        
        sigma_rbf = self.sigma_rbf

        # sigma_rbf adaptatif
        """pairwise_dists = cdist(X, Y)
        sigma_rbf = np.sqrt(0.5 * np.median(pairwise_dists**2))
        sigma_rbf = np.clip(sigma_rbf, 0.1, 10.0)"""
        
        # (le noyau linéaire revient à faire un EnKF classique)
        if self.kernel_type == 'linear':
            return X @ Y.T
        
        elif self.kernel_type == 'polynomial':
            prod = X @ Y.T
            return (1.0 + prod) ** self.poly_degree
        
        elif self.kernel_type == 'sigmoid':
            prod = X @ Y.T
            return np.tanh(self.c_tanh * prod)
        
        elif self.kernel_type == 'rbf':
            dist2 = cdist(X, Y, 'sqeuclidean')
            dist2 = np.clip(dist2, 0, 1e6)
            K = np.exp(-dist2 / (2.0 * sigma_rbf**2))
            return K + 1e-4 * eye(K.shape[0])
        
        elif self.kernel_type == 'rbf_exp':
            dist = cdist(X, Y, metric='euclidean')
            return np.exp(-dist / sigma_rbf)
        
        elif self.kernel_type == 'hyperbolique':
            return self.phi_poincare(X) @ self.phi_poincare(Y).T
        
        else :
            raise ValueError("Le noyau doit être kernel_type = 'linear', 'polynomial', 'sigmoid', 'rbf', 'rbf_exp' ou 'hyperbolique'")
    
    
    def rotation_farchi_bocquet(self, R_X, r_Sigma):
        """
        Cas r_Sigma < N
        """
        RX = R_X.copy()
        n = RX.shape[0]
        eps = 1.0

        def matrice_pattern(q, theta):
            mat = np.full((q, q), -theta / q)
            for i in range (q):
                mat[i][i]= 1-theta/q
                mat[0][i]= eps/(sqrt(q))
                mat[i][0]= eps/(sqrt(q))
            return mat

        for i in range (self.N - r_Sigma):
            q = r_Sigma + i+1
            theta = sqrt(q)/(sqrt(q)-eps)
            #Q_eps = - theta/q * matrice_pattern(q, theta)
            Q_eps = matrice_pattern(q, theta)
            zeros_col = zeros((n, 1))
            W = np.hstack([zeros_col, RX])
            RX = W @ Q_eps
        
        return RX
    

    def reechantillonnage_evensen(self, U_tilde, Sigma_tilde, r_Sigma, rsamp=2):
        """
        Cas r_Sigma > N
        rsamp: facteur de rééchantillonnage (choisi à 2 par défaut ici)
        """
        Nsamp = rsamp * r_Sigma
        n = U_tilde.shape[0]
        
        # matrice orthogonale de taille Nsamp
        O_mat, _ = sla.qr(np.random.randn(Nsamp, Nsamp), mode='economic')
        
        # tirage indépendant selon N(0,1)
        Z = np.random.randn(r_Sigma, Nsamp)
        A_tilde = U_tilde @ sqrt(Sigma_tilde) @ Z
        
        # Matrice de centrage M de taille Nsamp x Nsamp
        M = eye(Nsamp) - (1.0 / Nsamp) * ones((Nsamp, Nsamp))
        A_tilde = A_tilde @ M
        
        if rsamp > 1:
            # SVD de A_tilde
            U_samp, S_samp, V_samp_T = sla.svd(A_tilde, full_matrices=False)
            
            O_tilde = O_mat[:self.N, :r_Sigma].T
            
            U_samp_scaled = U_samp[:, :r_Sigma] * (sqrt(1.0 / rsamp) * S_samp[:r_Sigma])
            
            # Matrice de centrage M_tilde de taille N x N
            M_tilde = eye(self.N) - (1.0 / self.N) * ones((self.N, self.N))
            
            RX = U_samp_scaled @ O_tilde @ M_tilde
        else:
            RX = A_tilde[:, :self.N]
            
        return RX
    

    def assimilate(self, HMM, xx, yy):
        self.stats = dpr.stats.Stats(self, HMM, xx, yy)
        
        # Tirage de l'ensemble initial a priori
        E = HMM.X0.sample(self.N)
        
        # Dimension modèle
        n = E.shape[1]

        self.stats.assess(0, E=E)
        
        for k, kObs, t, dt in progbar(HMM.tseq.ticker):
            # prévision
            E = HMM.Dyn(E, t - dt, dt)
            self.stats.assess(k, kObs, 'f', E=E)
            
            if kObs is None:
                continue
                
            # observation
            y = yy[kObs]
            p = len(y)
            Obs_op = HMM.Obs(t)
            # projection de l'ensemble de prévision dans l'espace observé
            E_obs = Obs_op(E)
            
            mu_f = mean(E, axis=0)
            mu_f_obs = mean(E_obs, axis=0)

            X_f = (E - mu_f).T          # taille (n, N)
            Y_f = (E_obs - mu_f_obs).T  # taille (p, N)

            # matrice de covariance de l'erreur d'observation R
            R = Obs_op.noise.C.full

            # inflation adaptative ---
            # innovation
            d = y - mu_f_obs
            
            d_var = np.sum(d**2)           # Erreur totale constatée dans la vraie vie
            R_var = np.trace(R)            # Erreur théorique attendue du satellite
            
            # Variance de l'ensemble dans l'espace des observations
            P_obs_var = np.trace((Y_f @ Y_f.T) / (self.N - 1)) 
            
            ratio = (d_var - R_var) / (P_obs_var + 1e-8)
            
            if ratio > 1.0:
                lambda_infl = sqrt(ratio)  # racine car on compare des variances, et infl est ecart-type
            else:
                lambda_infl = 1.0
                
            lambda_infl = np.clip(lambda_infl, 1.0, self.infl)
            # ----

            # Décomposition en valeurs propres de R
            val_p, vec_p = sla.eigh(R)
            valeurs_propres_inv_racine = 1.0 / np.sqrt(val_p)

            # R^{-1/2} 
            R_inv_half = vec_p @ diag(valeurs_propres_inv_racine) @ vec_p.T

            Y_tilde = R_inv_half @ Y_f
            e_d_tilde = R_inv_half @ (y - mu_f_obs)
            
            # Construction de l'État Augmenté (n+p variables, N membres)
            Z = np.vstack((X_f, Y_tilde))
            
            Z = Z - Z.mean(axis=1, keepdims=True)
            K = self.compute_kernel(Z, Z)
            
            K_X  = K[:n, :n]      # Bloc purement modèle (n x n)
            K_H  = K[n:, n:]      # Bloc purement observé (p x p)
            K_XH = K[:n, n:]      # Bloc croisé (n x p)

            # alpha*H
            M_inv = (self.N - 1) * eye(p) + K_H
            M_reg = M_inv + self.reg_tikhonov * eye(p)
            alpha_H = sla.solve(M_reg, e_d_tilde)
            #alpha_H = sla.solve(M_inv, e_d_tilde)
            
            # moyenne modèle (taille n)
            mu_a = mu_f + K_XH @ alpha_H
            
            # Décomposition en valeurs propres de K_H (symétrique)
            valP_K_H, U_H = sla.eigh(K_H)
            
            # on force les valeurs propres à être >= 0 (gram definie >=0)
            valP_K_H = np.clip(valP_K_H, 0, None)

            diag_inv = diag(1.0 / ((self.N - 1) + valP_K_H + self.reg_tikhonov))
            terme_central = U_H @ diag_inv @ U_H.T
            
            # Covariance analysée physique
            Pa_X = (1.0 / (self.N - 1)) * (K_X - K_XH @ terme_central @ K_XH.T)

            # sécurité 
            Pa_X = Pa_X + self.reg_tikhonov * eye(n)
            
            # Décomposition en valeurs propres de Pa_X
            valP_Pa_X, U_Pa = sla.eigh(Pa_X)
            valP_Pa_X = np.clip(valP_Pa_X, 0, None) #sécurité
            
            # troncature au rang r_Sigma (valeurs propres > 0)
            tol = 1e-10
            idx_pos = valP_Pa_X > tol
            r_Sigma = np.sum(idx_pos)
            
            Sigma_tilde = diag(valP_Pa_X[idx_pos])
            U_tilde = U_Pa[:, idx_pos]
            
            # matrice racine de base R_X (n,r_Sigma)
            R_X = U_tilde @ sqrt(Sigma_tilde)

            # distinction de cas en fonction du rang r_Sigma
            if r_Sigma == self.N:
                E = mu_a + sqrt(self.N - 1) * R_X.T

            elif r_Sigma < self.N:
                R_X_aug = self.rotation_farchi_bocquet(R_X, r_Sigma)
                E = mu_a + sqrt(self.N - 1) * R_X_aug.T
                
            else:
                R_X_resampled = self.reechantillonnage_evensen(U_tilde, Sigma_tilde, r_Sigma)
                E = mu_a + R_X_resampled.T

            # inflation pour éviter les ensemble collapse
            #E = inflate_ens(E, self.infl)
            E = inflate_ens(E, lambda_infl)
                
            self.stats.assess(k, kObs, 'a', E=E)