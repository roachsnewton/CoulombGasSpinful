import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import haiku as hk
import numpy as np

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def test_flow_spin():
    """
        Generate a flow model `model` and test its properties.
    """
    from src.flow_spin import FermiNet_spin
    depth = 3
    spsize, tpsize = 16, 16
    L = 1.234
    n, dim = 7, 3

    def flow_fn(x):
        model = FermiNet_spin(depth, spsize, tpsize, L)
        return model(x)
    flow = hk.transform(flow_fn)
    model = flow

    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (n, dim))
    params = model.init(key, x)
    z = model.apply(params, None, x)

    """
        Generic test function for various flow models `model`.
    """
    n, dim = 8, 3
    key = jax.random.PRNGKey(42)
    x = jnp.array( np.random.uniform(0., L, (n, dim)) )
    params = model.init(key, x)
    z = model.apply(params, None, x)

    # Test that flow results of two "equivalent" (under lattice translations of PBC)
    # particle configurations are equivalent.
    print("---- Test the flow is well-defined under lattice translations of PBC ----")
    image = np.random.randint(-5, 6, size=(n, dim)) * L
    #print("image:", image / L)
    imagez = model.apply(params, None, x + image)
    assert jnp.allclose(imagez, z + image)

    # Test the translation equivariance.
    print("---- Test translation equivariance ----")
    shift = jnp.array( np.random.randn(dim) )
    #print("shift:", shift)
    shiftz = model.apply(params, None, x + shift)
    assert jnp.allclose(shiftz, z + shift)

    # Test of permutation equivariance.
    print("---- Test permutation equivariance ----")
    """
    Permutations should be applied to the first n//2 particles and the last n//2 particles.
    """
    Pup   = np.random.permutation(n//2)
    Pdown = np.random.permutation(n//2)
    P = jnp.concatenate([Pup, Pdown + n//2])
    Pz = model.apply(params, None, x[P, :])
    assert jnp.allclose(Pz, z[P, :])

