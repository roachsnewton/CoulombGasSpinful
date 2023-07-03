import jax
import jax.numpy as jnp
import haiku as hk

class FermiNet_spin(hk.Module):
    def __init__(self, depth, spsize, tpsize, L, init_stddev=0.01):
        super().__init__()
        self.depth = depth
        self.L = L
        self.init_stddev = init_stddev
        self.splayers = [hk.Linear(spsize, w_init=hk.initializers.RandomNormal(stddev=self.init_stddev))
                            for _ in range(depth)]
        self.tplayers = [hk.Linear(tpsize, w_init=hk.initializers.RandomNormal(stddev=self.init_stddev))
                            for _ in range(depth-1)]

    # f1 (N, dim)
    def _spstream0(self, x):
        """ Initial spstream, with shape (n, spsize0). """
        return jnp.zeros_like(x)

    # f2 (N, N, dim * 2 + 2)
    # cos_rij (3), sin_rij (3), dij_sin (1), dij_cos (1)
    def _tpstream0(self, x):
        """ Initial tpstream, with shape (n, n, tpsize0). """
        rij = x[:, None, :] - x
        cos_rij, sin_rij = jnp.cos(2*jnp.pi/self.L * rij), jnp.sin(2*jnp.pi/self.L * rij)
        n, _ = x.shape
        dij_sin = jnp.linalg.norm(sin_rij + jnp.eye(n)[..., None], axis=-1) * (1 - jnp.eye(n)) ##
        dij_cos = jnp.linalg.norm(cos_rij + jnp.eye(n)[..., None], axis=-1) * (1 - jnp.eye(n)) ##
        return jnp.concatenate((cos_rij, sin_rij, dij_sin[..., None], dij_cos[..., None]), axis=-1)

    # g = [f1, mean(f1_up, axis=0), mean(f1_down, axis=0), mean(f2, axis=1), mean(f2, axis=0)]
    def _f(self, spstream, tpstream):
        """
            The feature `f` as input to the sptream network.
            `f` has shape (n, fsize), where fsize = 2*spsize0 + tpsize0.
        """
        n, _ = spstream.shape
        nup = n // 2
        f = jnp.concatenate((spstream,
                             spstream[:nup,:].mean(axis=0, keepdims=True).repeat(n, axis=0),
                             spstream[nup:,:].mean(axis=0, keepdims=True).repeat(n, axis=0),
                             tpstream.mean(axis=1),
                             tpstream.mean(axis=0)
                             ), axis=-1)
        return f

    def __call__(self, x):
        spstream, tpstream = self._spstream0(x), self._tpstream0(x)

        for i in range(self.depth-1):
            f = self._f(spstream, tpstream)
            if i==0:
                spstream = jax.nn.softplus( self.splayers[i](f) )
                tpstream = jax.nn.softplus( self.tplayers[i](tpstream) )
            else:
                spstream += jax.nn.softplus( self.splayers[i](f) )
                tpstream += jax.nn.softplus( self.tplayers[i](tpstream) )

        f = self._f(spstream, tpstream)
        spstream += jax.nn.softplus( self.splayers[-1](f) )
        _, dim = x.shape
        final = hk.Linear(dim, w_init=hk.initializers.RandomNormal(stddev=self.init_stddev))
        return x + final(spstream)
