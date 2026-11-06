import jax


# All numerical regression checks use CPU float64, before creating test arrays.
jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

