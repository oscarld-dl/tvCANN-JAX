# Oscar Ludeña Navarro, oscar.ludenanavarro@dlr.de
# DLR Institute of Lightweight Systems, October 2026

import jax
jax.config.update("jax_enable_x64", True) 
import jax.numpy as jnp

#---------------------------------------------------------------------------------------------------------
# Kinematics
#---------------------------------------------------------------------------------------------------------

def right_cauchy_green(F):
    return F.T @ F

def isochoric_C(C):
    J = jnp.sqrt(jnp.linalg.det(C)) 
    return J ** (-2.0/3.0) * C, J
    
def invariants(C_bar):
    I1_bar = jnp.trace(C_bar)
    I2_bar = jnp.trace(jnp.linalg.inv(C_bar)) 
    return I1_bar, I2_bar 

#---------------------------------------------------------------------------------------------------------
# Projection
#---------------------------------------------------------------------------------------------------------

def project(A, C, J):
    C_inv = jnp.linalg.inv(C)
    return J ** (-2.0/3.0) * (A - jnp.tensordot(C, A, axes=2) / 3.0 * C_inv)

#--------------------------------------------------------------------------------------------------------
# Parameters of the raw network
#--------------------------------------------------------------------------------------------------------

def init_psi_params(key, hidden=(8,8)): 
    keys = jax.random.split(key, len(hidden) + 2) 
    params = {
        "W_defo": 0.1 * jax.random.normal(keys[0], (hidden[0], 2)), 
         "w_T": 0.1 * jax.random.normal(keys[1], (hidden[0],)),     
        "b0": jnp.zeros(hidden[0]), 
        "layers": [],
    }
    for i in range(1, len(hidden)):
        params["layers"].append({
            "W": 0.1 * jax.random.normal(keys[i+1], (hidden[i], hidden[i-1])), 
            "b": jnp.zeros(hidden[i]), 
        })
    params["w_out"] = 0.1 * jax.random.normal(keys[-1], (hidden[-1], )) 
    return params

#---------------------------------------------------------------------------------------------------------
# Forward pass of the raw network
#---------------------------------------------------------------------------------------------------------

positive = jax.nn.softplus
act_fn = jax.nn.softplus

def psi_raw(params, I1_bar, I2_bar, T_hat):
    defo = jnp.array([I1_bar-3.0, I2_bar-3.0]) 
    z = act_fn(positive(params["W_defo"]) @ defo + params["w_T"]*T_hat + params["b0"]) 

    for layer in params["layers"]:
        z = act_fn(positive(layer["W"]) @ z + layer["b"])

    return positive(params["w_out"]) @ z

#---------------------------------------------------------------------------------------------------------
# Normalized generating potential
#---------------------------------------------------------------------------------------------------------

EPS = 1e-3 
I_REF = 3.0

def psi_corrected(params, C_bar, T_hat, eps=EPS):
    I1_bar, I2_bar = invariants(C_bar)
    A_k, B_k = jax.grad(psi_raw, argnums=(1,2))(params, I_REF, I_REF, T_hat)
    delta_k = A_k - B_k
    alpha_k = eps * act_fn(-delta_k / eps)
    beta_k = eps * act_fn( delta_k / eps)
    return (psi_raw(params, I1_bar, I2_bar, T_hat)
            -psi_raw(params, I_REF, I_REF, T_hat)
            + alpha_k * (I1_bar - 3.0) + beta_k * (I2_bar - 3.0))

def S_bar_k(params, C_bar, T_hat):
    return 2.0 * jax.grad(psi_corrected, argnums=1)(params, C_bar, T_hat)


#---------------------------------------------------------------------------------------------------------
# Relaxation time network
#---------------------------------------------------------------------------------------------------------

N_MB = 5                                # number of maxwell branches
T_K = jnp.logspace(-2, 2, N_MB)         # branch time constant [s], one per decade

# The tau networks store the same kind of unconstrained pytree; constraints live in the forward pass. 
init_tau_params = init_psi_params

#---------------------------------------------------------------------------------------------------------
# Forward pass of the relaxation time network
#---------------------------------------------------------------------------------------------------------

def tau_raw(params, I1_bar, I2_bar, T_hat):
    defo = jnp.array([I1_bar - 3.0, I2_bar - 3.0]) 
    z = act_fn(params["W_defo"] @ defo + params["w_T"] * T_hat + params["b0"]) 
    for layer in params["layers"]:
        z = act_fn(layer["W"] @ z + layer["b"]) 
    return params["w_out"] @ z 

def tau_k(params, T_k, C_bar, T_hat):
    I1_bar, I2_bar = invariants(C_bar)
    return T_k * positive(tau_raw(params, I1_bar, I2_bar, T_hat))



