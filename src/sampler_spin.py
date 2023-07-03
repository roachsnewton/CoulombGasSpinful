import jax
import jax.numpy as jnp

def make_autoregressive_sampler_spin(network, sp_indices, nup, ndown, num_states, mask_fn=False):

    n = nup + ndown
    
    def _mask(state_idx):
        state_idx_up  = state_idx[:nup]
        state_idx_down= state_idx[nup:]
        
        # mask: spin up
        maskup = jnp.tril(jnp.ones((nup, num_states)), k=num_states-nup)
        idx_lb = jnp.concatenate( (jnp.array([-1]), state_idx_up[:-1]) )
        maskup = jnp.where(jnp.arange(num_states) > idx_lb[:, None], maskup, 0.)

        # mask: spin down
        maskdown = jnp.tril(jnp.ones((ndown, num_states)), k=num_states-ndown)
        idx_lb   = jnp.concatenate( (jnp.array([-1]), state_idx_down[:-1]) )
        maskdown = jnp.where(jnp.arange(num_states) > idx_lb[:, None], maskdown, 0.)

        mask = jnp.concatenate((maskup, maskdown), axis = 0)
        return mask

    def _logits(params, state_idx):
        """
        INPUT: state_idx: (n,), with elements being integers in [0, num_states).
        OUTPUT: masked_logits: (n, num_states)
        """
        logits = network.apply(params, None, sp_indices[state_idx])
        mask = _mask(state_idx)
        masked_logits = jnp.where(mask, logits, -1e50)
        return masked_logits

    def sampler(params, key, batch):
        state_indices = jnp.zeros((batch, n), dtype=jnp.int64)
        for i in range(n):
            key, subkey = jax.random.split(key)
            # logits.shape: (batch, n, num_states)
            logits = jax.vmap(_logits, (None, 0), 0)(params, state_indices)
            state_indices = state_indices.at[:, i].set(
                            jax.random.categorical(subkey, logits[:, i, :], axis=-1))
        return state_indices

    def log_prob(params, state_idx):
        logits = _logits(params, state_idx)
        logp = jax.nn.log_softmax(logits, axis=-1)
        logp = logp[jnp.arange(n), state_idx].sum()
        return logp

    if mask_fn:
        # Return the function `_mask` only for test and illustration purpose.
        return _mask, sampler, log_prob
    else:
        return sampler, log_prob


make_classical_score = lambda log_prob: jax.vmap(jax.grad(log_prob), (None, 0), 0)
