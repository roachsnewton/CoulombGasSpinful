import jax
from jax.config import config
config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(42)
import jax.numpy as jnp

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.orbitals import sp_orbitals
import haiku as hk
from src.sampler_spin import make_autoregressive_sampler_spin

def transformer(M):
    from src.autoregressive import Transformer

    num_layers, model_size, num_heads = 2, 32, 4
    hidden_size = 48

    def forward_fn(x):
        model = Transformer(M, num_layers, model_size, num_heads, hidden_size)
        return model(x)

    van = hk.transform(forward_fn)
    return van

def test_shapes():
    n, num_states = 10, 40
    nup, ndown = n//2, n//2
    sp_indices = jnp.array( sp_orbitals(2)[0] )[:num_states]

    van = transformer(num_states)
    dummy_state_idx = sp_indices[:n].astype(jnp.float64)
    params = van.init(key, dummy_state_idx)

    sampler, log_prob_novmap = make_autoregressive_sampler_spin(van, sp_indices, nup, ndown, num_states)
    log_prob = jax.vmap(log_prob_novmap, (None, 0), 0)
    batch = 200
    state_indices = sampler(params, key, batch)
    print("state_indices:", state_indices[0], "\nstate_indices.shape:", state_indices.shape)

    assert state_indices.shape == (batch, n)
    assert jnp.alltrue(state_indices < num_states)

    # The first nup indices are from small to large, and the last ndown indices are from small to large.
    assert jnp.alltrue(state_indices[:, :nup-1] < state_indices[:, 1:nup])
    assert jnp.alltrue(state_indices[:, nup:n-1] < state_indices[:, nup+1:n])

    logp = log_prob(params, state_indices)
    #print("logp:", logp, "\nlogp.shape:", logp.shape)
    assert logp.shape == (batch,)

def test_normalization():
    """
        Check probability normalization of the autoregressive model. Note this is a
    VERY STRONG CHECK of autoregressive property of the probability distribution.
    """
    import itertools

    n, num_states = 4, 10
    nup, ndown = n//2, n//2
    sp_indices = jnp.array( sp_orbitals(2)[0] )[:num_states]

    van = transformer(num_states)
    dummy_state_idx = sp_indices[:n].astype(jnp.float64)
    params = van.init(key, dummy_state_idx)

    # generage all possible state indices
    state_indices_up = jnp.array( list(itertools.combinations(range(num_states), nup)) )
    state_indices_down = jnp.array( list(itertools.combinations(range(num_states), ndown)) )
    
    state_indices = jnp.array( list(itertools.product(state_indices_up, state_indices_down)) )
    state_indices = jnp.reshape(state_indices, (state_indices.shape[0], -1))
    # print(state_indices.shape)
    # for i in range(state_indices.shape[0]):
    #      print(state_indices[i])
    # The first nup indices are from small to large, and the last ndown indices are from small to large.
    assert jnp.alltrue(state_indices[:, :nup-1] < state_indices[:, 1:nup])
    assert jnp.alltrue(state_indices[:, nup:n-1] < state_indices[:, nup+1:n])    

    _, log_prob_novmap = make_autoregressive_sampler_spin(van, sp_indices, nup, ndown, num_states)
    log_prob = jax.vmap(log_prob_novmap, (None, 0), 0)

    logp = log_prob(params, state_indices)
    norm = jnp.exp(logp).sum()
    print("logp:", logp, "\nnorm:", norm)
    assert jnp.allclose(norm, 1.)
