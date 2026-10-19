# IBM QPU code

`run_ibm_qpu.py` compares fixed non-private and private Pima QNN checkpoints on
the same eight balanced public-test cases. Training is not performed on IBM
hardware. The free default command performs exact statevector validation:

```bash
python Code/IBM_QPU/run_ibm_qpu.py
```

It writes `Results/IBM_QPU/ibm_local_validation.json`. `--hardware` submits one
real Runtime Estimator job, writes the job ID immediately, then writes the
completed result. This can consume QPU allocation and requires a configured IBM
account. `--backend NAME` chooses a device; otherwise the least-busy eligible
backend is selected. `--output-dir PATH` overrides the canonical result folder.

Fixed defaults are 4 qubits, 2 RY layers, 1,024 shots per binding, 16 bindings,
transpiler seed 2026, and a 540-second execution cap. See
`Documentation/PARAMETERS.md` for all constants and their interpretation.
