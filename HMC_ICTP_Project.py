import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def data_generate(d, k, alpha, Delta, sig, v):
    c_k, c_d = 1/np.sqrt(k), 1/np.sqrt(d)  
    #n = int(alpha*d**2/np.sqrt(d))
    n = int(alpha*d)
    W0 = tf.random.normal((k, d), dtype=tf.float32)
    X = tf.random.normal((d, n), dtype=tf.float32)
    Z = tf.random.normal((n,), dtype=tf.float32)
    M = W0@X*c_d
    M = sig(M)        
    Y = tf.tensordot(v, M, axes=1)*c_k + np.sqrt(Delta)*Z
    return W0, X, Y, v

def H(W, X, Y, v, Delta, sig):
    k, d = W.shape
    c_k, c_d = 1/np.sqrt(k), 1/np.sqrt(d)
    M = W@X*c_d
    M = sig(M)
    D = tf.tensordot(v, M, axes=1)*c_k - Y
    H0 = -tf.reduce_sum(W**2)/2
    H1 = -tf.reduce_sum(D**2)/(2*Delta)
    return H0+H1

def H_vec(W_vec, X, Y, v, Delta, sig, k, d):
    W = tf.reshape(W_vec, (k, d))
    return H(W, X, Y, v, Delta, sig) 

def test_error(Ws, W0, v, sig, ntest):
    k, d = W0.shape
    c_k, c_d = 1/np.sqrt(k), 1/np.sqrt(d)
    Xtest = tf.random.normal((d, ntest), dtype=tf.float32)
    M = W0@Xtest*c_d
    M = sig(M)
    Ytest = tf.tensordot(v, M, axes=1)*c_k
    errors=[]
    for i in range(Ws.shape[0]):
        M = Ws[i]@Xtest*c_d
        M = sig(M)
        Yhat = tf.einsum('a,ab->b', v, M)*c_k
        error = tf.reduce_mean((Yhat - Ytest)**2)/2
        error = error.numpy()
        errors.append(error)
    return errors

def q2(Ws, W0, v):
    k, d = W0.shape
    S0 = tf.einsum('k,ki,kj->ij', v, W0, W0)
    Ss = tf.einsum('k,nki,nkj->nij', v, Ws, Ws)
    q2s = 1 - tf.reduce_sum((Ss - S0)**2/(2*k*d**2), axis=[1, 2])
    return q2s.numpy()

def overlap(Ws, W0, v, order):
    k, d = W0.shape
    q00 = tf.squeeze(v[None, :]@tf.pow(W0@tf.transpose(W0), order)@v[:, None]).numpy()/(k*d**order)
    ovs = []
    for W in Ws:
        q01 = tf.squeeze(v[None, :]@tf.pow(W0@tf.transpose(W), order)@v[:, None]).numpy()/(k*d**order)
        q11 = tf.squeeze(v[None, :]@tf.pow(W@tf.transpose(W), order)@v[:, None]).numpy()/(k*d**order)
        ov = 1-(q00+q11-2*q01)/2
        ovs.append(ov)
    return ovs

def hmc(hmc_params, W_init, X, Y, v, k, alpha, Delta, sig, show_acceptance_rate, show_adaptation_steps):
    d, _ = X.shape
    step_size = hmc_params['step_size']
    num_leapfrog_steps = hmc_params['num_leapfrog_steps']
    num_adaptation_steps = hmc_params['num_adaptation_steps']
    num_post_adapt_steps = hmc_params['num_post_adapt_steps']
    
    W_init = tf.reshape(W_init, [-1])
    
    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=lambda W_vec: H_vec(W_vec, X, Y, v, Delta, sig, k, d),
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps)
    
    adaptive_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=hmc_kernel,
        num_adaptation_steps= num_adaptation_steps)
    
    @tf.function
    def run_chain():
        samples, kernel_results = tfp.mcmc.sample_chain(
            num_results=num_adaptation_steps + num_post_adapt_steps,
            current_state=W_init,
            num_burnin_steps=0,
            kernel=adaptive_kernel,
            trace_fn=lambda current_state, kernel_results:  kernel_results
        )
        return samples, kernel_results
        
    Ws, kernel_results = run_chain()
    if show_adaptation_steps:
        Ws = tf.concat([[W_init], Ws], axis=0)
    else:
        Ws = Ws[num_adaptation_steps:]
    Ws = tf.reshape(Ws, (tf.shape(Ws)[0], k, d))
    if show_acceptance_rate:
        acceptance_rate = tf.reduce_mean(tf.cast(kernel_results.inner_results.is_accepted, dtype=tf.float32))
        print("Acceptance rate:", acceptance_rate.numpy())
    return Ws
