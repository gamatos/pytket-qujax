from typing import Sequence
import pytest
from jax import numpy as jnp, jit, grad, random

from pytket.circuit import Circuit

from pytket.extensions.qujax.experimental import tk_to_qujax_args

def test_measure_and_reset() -> None:

    n_qubits = 2
    n_bits = 1
    circuit = Circuit(n_qubits, n_bits)
    circuit.H(0)
    circuit.CX(0, 1)
    circuit.Measure(0, 0)
    circuit.Reset(1)

    op_seq, op_metaparams_seq, param_inds_seq, n_qb, n_b = tk_to_qujax_args(circuit)

    assert op_seq == ["H", "CX", "Measure", "Reset"]
    assert op_metaparams_seq == [(0,), (0, 1), (0, 0), (1,)]
    assert param_inds_seq == [(), (), {"prng_keys" : 0}, {"prng_keys" : 1}]
    assert n_qubits == n_qb
    assert n_bits == n_b
