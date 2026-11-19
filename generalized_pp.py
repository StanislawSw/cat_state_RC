"""Generalized positive-P simulation of coherent-state superpositions and mixtures in a
linear coupled-waveguide photonic circuit.

A state is represented as a weighted sum of coherent-state pairs

    rho = sum_k w_k |alpha_k><beta_k| / <beta_k|alpha_k>,

where each pair (alpha_k, beta_k) is propagated as one phase-space trajectory and every
normally ordered observable is the w_k-weighted sum over pairs. The circuit is linear, so the
trajectories are deterministic and a single trajectory per pair gives the exact result, with
no Fock-space cutoff.

Conventions used throughout
---------------------------
* Propagation coordinate z (mm); the solver integrates d/dz = (1/(hbar*vg)) d/dt.
* The trajectory pair is stored as (alpha, alpha_c) with alpha^+ = conj(alpha_c), so for the
  initial pair alpha_c = beta (the bra amplitude).
* Equations of motion correspond to the Hamiltonian
      H = -hbar*delta*n - J*(a_i^dag a_j + h.c.)
  with amplitude loss gamma (energy units, i.e. hbar*gamma).
"""
from itertools import combinations, product

import numba as nb
import numpy as np


# --------------------------------------------------------------------------------------
# States -> weighted positive-P pairs
# --------------------------------------------------------------------------------------

def BlochToPP(r, theta, phi, alpha):
    """State on the Bloch sphere spanned by |alpha>, |-alpha>.

    r = 1 is the pure cat cos(theta/2)|alpha> + exp(i phi) sin(theta/2)|-alpha>,
    r = 0 the classical mixture (|alpha><alpha| + |-alpha><-alpha|)/2.
    Returns the 2x2 coefficient matrix dm, the pairs PParr (4x2) and the weights (4).
    """
    vec = r*np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)]) # Bloch vector
    eye = np.eye(2)
    pauli = [np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.array([[1,0],[0,-1]])]

    dm = (1/2)*(eye + vec[0]*pauli[0] + vec[1]*pauli[1] + vec[2]*pauli[2])

    PParr = np.zeros([4,2],dtype='complex')

    overlap = np.exp(-0.5*(abs(alpha)**2 + abs(-alpha)**2 + 2*np.conj(alpha)*alpha))  # <-alpha|alpha>
    weights_0 = np.array([dm[0,0],overlap*dm[0,1],overlap*dm[1,0],dm[1,1]])
    weights = weights_0.copy()/np.sum(weights_0)

    PParr[0,0] = alpha
    PParr[0,1] = alpha

    PParr[1,0] = alpha
    PParr[1,1] = -alpha

    PParr[2,0] = -alpha
    PParr[2,1] = alpha

    PParr[3,0] = -alpha
    PParr[3,1] = -alpha

    return dm, PParr, weights


def overlap(a,b):
    """<a|b> for coherent states."""
    return np.exp(-0.5*(abs(a)**2 + abs(b)**2 - 2*np.conj(a)*b))


def normalisation_factor(coeffs, alphas):
    coeffs = np.array(coeffs, dtype=complex)
    alphas = np.array(alphas, dtype=complex)
    S = 0.0 + 0.0j
    for i in range(len(coeffs)):
        for j in range(len(coeffs)):
            S += coeffs[i] * np.conj(coeffs[j]) * overlap(alphas[j], alphas[i])
    return 1.0 / S


def coh_to_PP(states,coefficients):
    """Pure superposition sum_i c_i |alpha_i>  ->  n^2 pairs and weights."""
    coef = coefficients
    norm = normalisation_factor(coef, states)
    coef_conj = np.conj(coef)
    combos = list(product(range(len(states)),repeat = 2))
    P = np.zeros([len(combos),2],dtype='complex')
    weights = np.zeros([len(combos)],dtype='complex')
    for i in range(len(combos)):
        idx_a = combos[i][0]
        idx_ac = combos[i][1]
        state1 = states[idx_a]
        state2 = states[idx_ac]
        coef1 = coef[idx_a]
        coef2 = coef_conj[idx_ac]
        weights[i] = norm*coef1*coef2*overlap(state2,state1)
        P[i,0] = state1
        P[i,1] = state2
    return P,weights


# --------------------------------------------------------------------------------------
# Wigner functions
# --------------------------------------------------------------------------------------

def PPToW(PParr,weights,q,p,Ns):
    """Wigner function of sum_k w_k |alpha_k><beta_k|/<beta_k|alpha_k> on a grid of
    gamma = Q + iP (same convention as qutip.wigner(..., g=2))."""
    qs = np.linspace(-q,q,Ns)
    ps = np.linspace(-p,p,Ns)
    Q,P = np.meshgrid(qs,ps)
    Wigs = Q + 1j*P
    one = np.ones([Ns,Ns])
    W = 0
    for i in range(len(weights)):
        W += np.real(weights[i]*np.exp(-2*(np.conj(Wigs)-one*np.conj(PParr[i,1]))*(Wigs-one*PParr[i,0])))
    W = np.real((2/(np.pi))*W)
    return Q,P,W


def qutip_tester(stat,coefficients,p,q,Ns,Nc=30):
    """Reference Wigner function of the pure superposition, from QuTiP in a truncated Fock space."""
    from qutip import coherent, wigner
    cat = coefficients[0]*coherent(Nc,stat[0])
    for i in range(1,len(stat)):
        cat += coefficients[i]*coherent(Nc,stat[i])
    cat = cat.unit()
    qs = np.linspace(-q,q,Ns)
    ps = np.linspace(-p,p,Ns)
    Q,P = np.meshgrid(qs,ps)
    W = wigner(cat,qs,ps,g=2)
    return Q,P,W


# --------------------------------------------------------------------------------------
# Phase-space solver (linear circuit)
# --------------------------------------------------------------------------------------

@nb.jit(nopython=True)
def PP_solver_many_mode_Midpoint(hbar,vg,delta,gamma,tmax,dt,alpha0,alpha_c0,modes,J): #Exponential midpoint method
  """Exponential-midpoint integration of one trajectory pair through the linear circuit.
  J[k,l,i] is the coupling between modes k,l at grid point i (held constant over the step)."""
  Nt = int(tmax/dt)
  halfdt = 0.5*dt
  sol_alpha =  np.zeros((Nt,modes),dtype='complex')
  sol_alpha_c = np.zeros((Nt,modes),dtype='complex')
  alpha = alpha0.copy()
  alpha_c = alpha_c0.copy()
  delta = delta.copy()
  gamma = float(gamma)

  # on-site part (detuning and loss), the same for alpha and alpha_c
  alpha_change_E = np.zeros((modes),dtype='complex')
  for w in range(modes):
    alpha_change_E[w] = (1/(hbar*vg))*(1j*hbar*delta[w] - 0.5*gamma)

  for i in range(Nt):
    sol_alpha[i,:] = alpha
    sol_alpha_c[i,:] = alpha_c

    coupling_a = np.zeros((modes),dtype='complex')
    coupling_ac = np.zeros((modes),dtype='complex')
    for k in range(modes):
      vals_a = 0
      vals_ac = 0
      for l in range(modes):
        vals_a = vals_a + J[k,l,i]*alpha[l]
        vals_ac = vals_ac+ J[k,l,i]*alpha_c[l]
      coupling_a[k] = vals_a
      coupling_ac[k] = vals_ac

    mid_a = np.zeros((modes),dtype='complex')
    mid_a_c = np.zeros((modes),dtype='complex')
    alpha_change_R = np.zeros((modes),dtype='complex')
    alpha_change_R_c = np.zeros((modes),dtype='complex')
    for w in range(modes):
      alpha_change_R[w] =  (1/(hbar*vg))*(1j*coupling_a[w])
      alpha_change_R_c[w] = (1/(hbar*vg))*(1j*coupling_ac[w])
    for w in range(modes):
      if np.abs(alpha_change_E[w]) == 0:
          mid_a[w] = alpha[w] + alpha_change_R[w]*halfdt
          mid_a_c[w] = alpha_c[w] + alpha_change_R_c[w]*halfdt
      else:
          mid_a[w] = (alpha[w] + (alpha_change_R[w]/alpha_change_E[w]))*np.exp(alpha_change_E[w]*halfdt) - (alpha_change_R[w]/alpha_change_E[w])
          mid_a_c[w] = (alpha_c[w] + (alpha_change_R_c[w]/alpha_change_E[w]))*np.exp(alpha_change_E[w]*halfdt) - (alpha_change_R_c[w]/alpha_change_E[w])

    coupling_a = np.zeros((modes),dtype='complex')
    coupling_ac = np.zeros((modes),dtype='complex')
    for k in range(modes):
      vals_a = 0
      vals_ac = 0
      for l in range(modes):
        vals_a = vals_a + J[k,l,i]*mid_a[l]
        vals_ac = vals_ac+ J[k,l,i]*mid_a_c[l]
      coupling_a[k] = vals_a
      coupling_ac[k] = vals_ac

    for w in range(modes):
      alpha_change_R[w] =  (1/(hbar*vg))*(1j*coupling_a[w])
      alpha_change_R_c[w] = (1/(hbar*vg))*(1j*coupling_ac[w])
    for w in range(modes):
      if np.abs(alpha_change_E[w]) == 0:
          alpha[w] = alpha[w] + alpha_change_R[w]*dt
          alpha_c[w] = alpha_c[w] + alpha_change_R_c[w]*dt
      else:
          alpha[w] = (alpha[w] + (alpha_change_R[w]/alpha_change_E[w]))*np.exp(alpha_change_E[w]*dt) - (alpha_change_R[w]/alpha_change_E[w])
          alpha_c[w] = (alpha_c[w] + (alpha_change_R_c[w]/alpha_change_E[w]))*np.exp(alpha_change_E[w]*dt) - (alpha_change_R_c[w]/alpha_change_E[w])
  return sol_alpha,sol_alpha_c


def Positive_P_CAT(Physical,Numerical,modes,states,coeff,viewW = False,mixed=False,mixed_par = None,detector_noise=False,bgr=None,wigner_grid=(4,4,200)):
   """Propagate a cat / kitten / mixture (in the modes where W != 0) together with coherent
   states in the modes phase_ref, and return
       times, <n_i>(z), g2_ij(z)
   (plus Q, P, output Wigner functions and all trajectories if viewW=True).

   Physical  = [hbar, vg, delta, gamma, J, W, phase_ref, coh_phase]
   Numerical = [dt, tmax]
   mixed=True : state from BlochToPP(*mixed_par), mixed_par = [r, theta, phi, alpha]
   mixed=False: pure superposition sum_i coeff[i] |states[i]>
   detector_noise=True adds an uncorrelated background of mean bgr counts to every detector.
   """
   [hbar,vg,delta,gamma,J,W,phase_ref,coh_phase] = Physical
   [dt,tmax] = Numerical
   if mixed ==True:
       [r,theta,phi,alpha] = mixed_par
       dm,PP,weights = BlochToPP(r,theta,phi,alpha)
   else:
       PP, weights = coh_to_PP(states,coeff)
   # the cat occupies the (normalised) mode sum_m W_m a_m
   W = np.array(W, dtype='complex')
   W = W/np.sqrt(np.sum(np.abs(W)**2))

   tmax = tmax - dt
   combos = list(combinations(np.arange(modes),2))
   Nt = int(tmax/dt)
   times = np.arange(Nt)*dt
   N_a = np.zeros([modes,Nt],dtype='complex')
   G_top = np.zeros([len(combos),Nt],dtype='complex')
   N_bot1 = np.zeros([len(combos),Nt],dtype='complex')
   N_bot2 = np.zeros([len(combos),Nt],dtype='complex')
   if viewW == True:
       PParr_big = np.zeros([Nt,modes,len(weights),2],dtype='complex')

   for k in range(len(weights)):
       scal = weights[k]
       a0 = PP[k,0]*W
       a_c0 = PP[k,1]*W
       for h in range(len(coh_phase)):
           a0[phase_ref[h]] = coh_phase[h]
           a_c0[phase_ref[h]] = coh_phase[h]
       a,a_c = PP_solver_many_mode_Midpoint(hbar,vg,delta,gamma,tmax,dt,a0,a_c0,modes,J)
       N_a += (scal*a*np.conj(a_c)).T
       if viewW == True:
           PParr_big[:,:,k,0] = a
           PParr_big[:,:,k,1] = a_c
       for c in range(len(combos)):
           i, j = combos[c]
           G_top[c,:] += scal*np.conj(a_c[:,i])*np.conj(a_c[:,j])*a[:,j]*a[:,i]
           N_bot1[c,:] += scal*np.conj(a_c[:,i])*a[:,i]
           N_bot2[c,:] += scal*np.conj(a_c[:,j])*a[:,j]

   N_av_a = np.real(N_a)
   N_bot1_av = np.real(N_bot1)
   N_bot2_av = np.real(N_bot2)
   if detector_noise == True:
       G_av = (np.real(G_top)+bgr*N_bot1_av + bgr*N_bot2_av + bgr**2) / ((N_bot1_av+bgr) * (N_bot2_av+bgr))
       N_av_a = N_av_a + bgr          # measured mean counts include the background
   else:
       G_av = np.real(G_top) / (N_bot1_av * N_bot2_av)
   if viewW == True:
       # Wigner function of every output mode at the end of the chip: the reduced state of
       # mode m is given by the pairs restricted to that mode, with the same weights
       q, p, Ns = wigner_grid
       wigs = []
       for m in range(modes):
           Q,P,Wig = PPToW(PParr_big[-1,m,:,:],weights,q,p,Ns)
           wigs.append([Wig])
       return times,N_av_a,G_av,Q,P,wigs,PParr_big
   else:
       return times,N_av_a,G_av


def run_CAT(states,coef,Physical,Numerical,modes,mixed,mixed_par,detector_noise=False,bgr=None):
    """Output <n_i> and g2_ij at the end of the chip."""
    # Physical = [hbar,vg,delta,gamma,J_mat,W,phase_ref,coh_phase]
    # Numerical = [dt,tmax]  (dt must match the spatial grid J_mat was built on)
    time,N_av_a,G_av = Positive_P_CAT(Physical,Numerical,modes,states,coef,mixed=mixed,mixed_par=mixed_par,detector_noise=detector_noise,bgr=bgr)
    return N_av_a[:,-1],G_av[:,-1]


# --------------------------------------------------------------------------------------
# Coupling geometry
# --------------------------------------------------------------------------------------

def add_SuperGauss(J,mode,x,x_av,sigma,d,J_amp):
    """Add a super-Gaussian coupling pulse J_amp*exp(-((x-x_av)/sigma)^d) between waveguides
    mode and mode+1."""
    if d%2 != 0:
        raise ValueError('"d" should be an even number')
    J[mode,:] = J[mode,:] + J_amp*np.exp(-((x-x_av*np.ones([len(x)],dtype='complex'))/sigma)**(d))
    return J


def J_matrix(J_values):
    """[J_12(z), J_23(z), ...]  ->  nearest-neighbour coupling matrix J[k,l,z]."""
    modes = J_values.shape[0] + 1
    J_mat = np.zeros([modes,modes,J_values.shape[1]],dtype='complex')
    for j in range(modes-1):
        J_mat[j,j+1,:] = J_mat[j+1,j,:] = J_values[j,:]
    return J_mat


# --------------------------------------------------------------------------------------
# Exact reference: QuTiP in a truncated Fock space
# --------------------------------------------------------------------------------------

def master(modes,Ncut,hbar,vg,gamma,delta,J_values,x,state_param,phase_ref,coh_phase,cat_mode):
    """Exact Hilbert-space simulation of the same chip, in the same conventions as the
    positive-P solver (H = -hbar*delta*n - J*hop, z in mm, rates divided by hbar*vg).
    Input: coherent states coh_phase in modes phase_ref and the BlochToPP state
    [r, theta, phi, alpha] in mode cat_mode.

    Returns z, <n_i>(z), g2_ij(z) (pairs ordered as itertools.combinations).
    For gamma = 0 each eigen-component of the (at most rank-2) input state is propagated as a
    ket, which is much cheaper than a density-matrix simulation.
    """
    from qutip import (destroy, qeye, tensor, coherent, ket2dm, mesolve, sesolve, coefficient)
    r, theta, phi, alpha = state_param
    a_list = [tensor([destroy(Ncut) if m == i else qeye(Ncut) for m in range(modes)]) for i in range(modes)]
    combos = list(combinations(range(modes),2))
    e_ops = [a.dag()*a for a in a_list]
    for (i, j) in combos:
        e_ops.append(a_list[i].dag()*a_list[j].dag()*a_list[j]*a_list[i])

    H0 = 0
    for i in range(modes):
        H0 += -(np.real(delta[i])/vg)*a_list[i].dag()*a_list[i]
    H_td = [H0]
    for j in range(modes-1):
        hop = a_list[j].dag()*a_list[j+1] + a_list[j+1].dag()*a_list[j]
        H_td.append([-hop, coefficient(np.real(J_values[j,:])/(hbar*vg), tlist=x)])
    c_ops = [np.sqrt(gamma/(hbar*vg))*a for a in a_list] if gamma > 0 else []

    # cat-mode density matrix in the same parametrisation as BlochToPP
    dm, _, _ = BlochToPP(r, theta, phi, alpha)
    basis = [coherent(Ncut, alpha), coherent(Ncut, -alpha)]
    rho_cat = sum(dm[i, j]*basis[i]*basis[j].dag() for i in range(2) for j in range(2))
    rho_cat = rho_cat/rho_cat.tr()
    def full_state(cat_ket):
        kets = []
        for m in range(modes):
            if m == cat_mode:
                kets.append(cat_ket)
            else:
                kets.append(coherent(Ncut, coh_phase[list(phase_ref).index(m)]))
        return tensor(kets)

    # max_step: the couplings are zero at z=0, so without it the adaptive integrator can
    # step straight over the pulses
    opts = {'store_states': False, 'atol': 1e-10, 'rtol': 1e-8, 'nsteps': 100000,
            'max_step': 10*(x[1]-x[0])}
    if gamma == 0:
        vals, vecs = rho_cat.eigenstates()
        expect = 0
        for p_k, v in zip(vals, vecs):
            if p_k > 1e-12:
                res = sesolve(H_td, full_state(v), x, e_ops=e_ops, options=opts)
                expect = expect + p_k*np.array(res.expect)
    else:
        rho0 = tensor([rho_cat if m == cat_mode else ket2dm(coherent(Ncut, coh_phase[list(phase_ref).index(m)]))
                       for m in range(modes)])
        res = mesolve(H_td, rho0, x, c_ops, e_ops=e_ops, options=opts)
        expect = np.array(res.expect)
    expect = np.real(expect)
    n = expect[:modes]
    g2 = np.array([expect[modes+c]/(n[i]*n[j]) for c, (i, j) in enumerate(combos)])
    return x, n, g2


# --------------------------------------------------------------------------------------
# Data set
# --------------------------------------------------------------------------------------

def Spirals(Nsample,seeds):
    """Two interleaved noisy spirals in the unit square; returns points X (2N x 2) and labels y."""
    [seed1,seed2,seed3,seed4] = seeds
    N = Nsample
    np.random.seed(seed1)
    theta = np.sqrt(np.random.rand(N))*2.2*np.pi
    np.random.seed(seed2)
    theta_2 = np.sqrt(np.random.rand(N))*2.2*np.pi

    r_a = 2*theta + np.pi
    data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
    np.random.seed(seed3)
    x_a = data_a + np.random.randn(N,2)
    x_a = (x_a+20)/40

    r_b = -2*theta_2 - np.pi
    data_b = np.array([np.cos(theta_2)*r_b, np.sin(theta_2)*r_b]).T
    np.random.seed(seed4)
    x_b = data_b + np.random.randn(N,2)
    x_b = (x_b+20)/40
    np.random.seed()
    res_a = np.append(x_a, np.zeros((N,1)), axis=1)
    res_b = np.append(x_b, np.ones((N,1)), axis=1)

    res = np.array(np.append(res_a, res_b, axis=0))
    np.random.shuffle(res)

    X = np.array([res[:,0], res[:,1]]).T
    y = np.array(res[:,2])
    return  X, y
