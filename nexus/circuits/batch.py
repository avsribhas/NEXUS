import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np
import tarjuman

class BatchRunner:
    """Run many trials of one graded circuit in Jaxley, compiled once."""

    def __init__(self, nml_path, neuron_ids, delta_t: float, t_max: float):
        self.neuron_ids = list(neuron_ids)
        self.delta_t = float(delta_t)
        self.t_max = float(t_max)
        self.n_steps = int(round(t_max / delta_t)) + 1
        self.model = tarjuman.from_neuroml(str(nml_path), strict=False)
        module = self.model.module; module.delete_recordings(); module.delete_stimuli()
        self.views = [self.model.view(f"pop_{nid}") for nid in self.neuron_ids]
        for view in self.views:
            view.record("v", verbose=False)
        module.init_states(delta_t=self.delta_t)
        self.module = module

        def one_trial(currents):
            data_stimuli = None
            for i, view in enumerate(self.views):
                data_stimuli = view.data_stimulate(currents[i], data_stimuli=data_stimuli, verbose=False)
            v = jx.integrate(self.module, delta_t=self.delta_t, t_max=self.t_max, data_stimuli=data_stimuli)
            return v[:, : self.n_steps]
        self._run = jax.jit(jax.vmap(one_trial))

    def run(self, currents) -> np.ndarray:
        """Run a batch of trials; currents in nA with shape (trials, neurons, n_steps); returns voltages in mV, same shape."""
        currents = np.asarray(currents, dtype=np.float32)
        if currents.ndim != 3 or currents.shape[1] != len(self.neuron_ids) or currents.shape[2] != self.n_steps:
            raise ValueError(f"currents must have shape (trials, {len(self.neuron_ids)}, {self.n_steps}), got {currents.shape}")
        out = self._run(jnp.asarray(currents))
        return np.asarray(out)
