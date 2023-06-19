from qiskit import QuantumCircuit

import numpy as np
from qiskit.algorithms.eigensolvers.eigensolver import EigensolverResult
from qiskit.algorithms.list_or_dict import ListOrDict

from qiskit.primitives import BaseEstimator, BaseSampler
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp, DensityMatrix

from qiskit.algorithms import AlgorithmError
from qiskit.algorithms import VariationalAlgorithm, VariationalResult
from qiskit.algorithms.optimizers import ADAM, OptimizerResult
from qiskit.algorithms.gradients import BaseEstimatorGradient
from qiskit.algorithms.utils import validate_bounds, validate_initial_point
from qiskit.algorithms.utils.set_batching import _set_default_batchsize
from qiskit.algorithms.eigensolvers import Eigensolver, EigensolverResult

from collections.abc import Callable, Sequence
from typing import Any
from time import time

import logging

from qiskit.quantum_info.operators.base_operator import BaseOperator
logger = logging.getLogger(__name__)

class VQSEResult(VariationalResult, EigensolverResult):
    """Variational quantum state eigensolver result."""

    def __init__(self) -> None:
        super().__init__()
        self._cost_function_evals: int | None = None

    @property
    def cost_function_evals(self) -> int | None:
        """The number of cost optimizer evaluations."""
        return self._cost_function_evals

    @cost_function_evals.setter
    def cost_function_evals(self, value: int) -> None:
        self._cost_function_evals = value
        
        

class VQSE(VariationalAlgorithm, Eigensolver):
    def __init__(
        self, 
        estimator: BaseEstimator,
        sampler: BaseSampler,
        ansatz: QuantumCircuit,
        optimizer: ADAM,
        *,
        adaptive: bool = True,
        gradient: BaseEstimatorGradient | None = None,
        initial_point: Sequence[float] | None = None,
        callback: Callable[[int, np.ndarray, float, dict[str, Any]], None] | None = None,
    ):
        super().__init__()

        self.estimator = estimator
        self.sampler = sampler
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.gradient = gradient
        # this has to go via getters and setters due to the VariationalAlgorithm interface
        self.initial_point = initial_point
        self.callback = callback

        self.adaptive = adaptive 
        
        self._err = []

    @property
    def initial_point(self) -> Sequence[float] | None:
        return self._initial_point

    @initial_point.setter
    def initial_point(self, value: Sequence[float] | None) -> None:
        self._initial_point = value
        
    def compute_eigenvalues(
        self, 
        operator: BaseOperator | PauliSumOp, 
        aux_operators: ListOrDict[BaseOperator | PauliSumOp] | None = None
    ) -> EigensolverResult:
        
        return EigensolverResult()
    
    def compute_eigensystem(
        self,
        target: QuantumCircuit,
        m: int,
        f: Callable[[float], float] = lambda x: x,
        q: Sequence[float] | None = None,
        r: Sequence[float] | None = None,
        delta: float = 0.01
    ) -> VQSEResult:
        self._check_target(target)
        
        if q is None:
            self.q = [1./(i+2) for i in range(m)]
        else:
            if len(q) != m:
                raise AlgorithmError("q much have length equal to m.")
            self.q = q
        
        if r is None:
            if m == 1:
                self.r = [1]*self.ansatz.num_qubits
            else:
                self.r = [1 + (i - 1)*delta for i in range(self.ansatz.num_qubits)]
        else:
            if len(r) != self.ansatz.num_qubits:
                raise AlgorithmError("r must have length equal to the number of qubits.")
            self.r = r


        # Validate target against ansatz  
        
        initial_point = validate_initial_point(self.initial_point, self.ansatz)

        bounds = validate_bounds(self.ansatz)
         
        start_time = time()

        evaluate_energy = self._get_evaluate_energy(target, f, m)

        if self.gradient is not None:
            evaluate_gradient = self._get_evaluate_gradient(target, f, m)
        else:
            evaluate_gradient = None

        # perform optimization
        if callable(self.optimizer):
            optimizer_result = self.optimizer(
                fun=evaluate_energy, x0=initial_point, jac=evaluate_gradient, bounds=bounds
            )
        else:
            # we always want to submit as many estimations per job as possible for minimal
            # overhead on the hardware
            was_updated = _set_default_batchsize(self.optimizer)

            optimizer_result = self.optimizer.minimize(
                fun=evaluate_energy, x0=initial_point, jac=evaluate_gradient, bounds=bounds
            )

            # reset to original value
            if was_updated:
                self.optimizer.set_max_evals_grouped(None)

        optimizer_time = time() - start_time

        logger.info(
            "Optimization complete in %s seconds.\nFound optimal point %s",
            optimizer_time,
            optimizer_result.x,
        )

        return self._build_vqse_result(
            target, m, optimizer_result, optimizer_time
        )
        
    def _get_evaluate_gradient(
        self,
        target: QuantumCircuit,
        f: Callable[[float], float],
        m: int,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Get a function handle to evaluate the gradient at given parameters for the ansatz.

        Args:
            ansatz: The ansatz preparing the quantum state.
            operator: The operator whose energy to evaluate.

        Returns:
            A function handle to evaluate the gradient at given parameters for the ansatz.

        Raises:
            AlgorithmError: If the primitive job to evaluate the gradient fails.
        """
        circuit = target.compose(self.ansatz)

        def evaluate_gradient(parameters: np.ndarray) -> np.ndarray:
            # broadcasting not required for the estimator gradients
            try:
                job = self.gradient.run([circuit], [self._hamiltonian(f, target, parameters, m)], [parameters])
                gradients = job.result().gradients
            except Exception as exc:
                raise AlgorithmError("The primitive job to evaluate the gradient failed!") from exc

            return gradients[0]

        return evaluate_gradient
    
    def _get_evaluate_energy(
        self,
        target: QuantumCircuit,
        f: Callable[[float], float],
        m: int,
    ) -> Callable[[np.ndarray], np.ndarray | float]:
        """Returns a function handle to evaluate the energy at given parameters for the ansatz.
        This is the objective function to be passed to the optimizer that is used for evaluation.

        Args:
            ansatz: The ansatz preparing the quantum state.
            operator: The operator whose energy to evaluate.

        Returns:
            A callable that computes and returns the energy of the hamiltonian of each parameter.

        Raises:
            AlgorithmError: If the primitive job to evaluate the energy fails.
        """
        num_parameters = self.ansatz.num_parameters

        # avoid creating an instance variable to remain stateless regarding results
        eval_count = 0

        circuit = target.compose(self.ansatz)

        def evaluate_energy(parameters: np.ndarray) -> np.ndarray | float:
            nonlocal eval_count

            h = self._hamiltonian(f, target, parameters, m)
            # handle broadcasting: ensure parameters is of shape [array, array, ...]
            parameters = np.reshape(parameters, (-1, num_parameters)).tolist()
            batch_size = len(parameters)

            try:
                job = self.estimator.run(batch_size * [circuit], batch_size * [h], parameters)
                estimator_result = job.result()
            except Exception as exc:
                raise AlgorithmError("The primitive job to evaluate the energy failed!") from exc

            values = estimator_result.values

            if self.callback is not None:
                metadata = estimator_result.metadata
                for params, value, meta in zip(parameters, values, metadata):
                    eval_count += 1
                    self.callback(eval_count, params, value, meta)

            energy = values[0] if len(values) == 1 else values

            # --- logging error --- #
            state = DensityMatrix(target)
            _, est_eigvals = self._estimate_eigenvalues(target, parameters, m)
            true_eigvals = np.linalg.eigvalsh(state)[-m:]

            #print(f'est: {est_eigvals}')
            #print(f'true: {true_eigvals}')
            err = np.sum((est_eigvals - true_eigvals)**2)
            self._err.append(err)
            print(f'err: {err}, energy = {energy}')
            # --------------------- #

            return energy

        return evaluate_energy

    def _global_hamiltonian(
        self,
        target: QuantumCircuit,
        parameters: np.ndarray,
        m: int
    ):
        H = SparsePauliOp.from_list([('I'*self.ansatz.num_qubits, 1)])
        
        if self.adaptive:
            bitstrings, _ = self._estimate_eigenvalues(target, parameters, m)
        else:
            bitstrings = list(range(len(self.q)))
        for q,z in zip(self.q, bitstrings):
            pauli_ops = []
            for j in range(1 << self.ansatz.num_qubits):
                label = f'{j:0{self.ansatz.num_qubits}b}'.replace('0', 'I').replace('1', 'Z')

                amplitude = -q*(-1)**bin(j & z).count('1')/(1 << self.ansatz.num_qubits)
                pauli_ops.append((label, amplitude))

            H += SparsePauliOp.from_list(pauli_ops) 
            H = H.simplify()
            
        return H
    
    def _local_hamiltonian(self):
        pauli_ops = [('I'*self.ansatz.num_qubits, 1)]
        for i in range(self.ansatz.num_qubits):
            label = 'I'*i + 'Z' + 'I'*(self.ansatz.num_qubits - i - 1)
            amplitude = -self.r[i]
            pauli_ops.append((label, amplitude))
        
        H = SparsePauliOp.from_list(pauli_ops)
        return H
       
    def _hamiltonian(
        self, 
        f: Callable[[float], float],
        target: QuantumCircuit,
        parameters: np.ndarray,
        m: int,
    ):
        t = self.optimizer._t/self.optimizer._maxiter
        return (1 - f(t))*self._local_hamiltonian() + f(t)*self._global_hamiltonian(target, parameters, m)

    def _check_target(self, target: QuantumCircuit):
        """Check that the number of qubits of target and ansatz match and that the ansatz is
        parameterized.
        """
        if target.num_qubits != self.ansatz.num_qubits:
            try:
                logger.info(
                    "Trying to resize ansatz to match operator on %s qubits.", target.num_qubits
                )
                self.ansatz.num_qubits = target.num_qubits
            except AttributeError as error:
                raise AlgorithmError(
                    "The number of qubits of the ansatz does not match the "
                    "operator, and the ansatz does not allow setting the "
                    "number of qubits using `num_qubits`."
                ) from error

        if self.ansatz.num_parameters == 0:
            raise AlgorithmError("The ansatz must be parameterized, but has no free parameters.")
    
    def _estimate_eigenvalues(
        self, 
        target: QuantumCircuit,
        parameters: np.ndarray,
        m: int,
    ):
        circuit = target.compose(ansatz)
        circuit.measure_all()

        job = self.sampler.run([circuit], parameters)
        sampler_results = job.result()
        
        probs_dict = sampler_results.quasi_dists[0]
        [*bitstrings], [*probs] = zip(*probs_dict.items())
        bitstrings = np.array(bitstrings)
        probs = np.array(probs)
        
        inds = np.argsort(probs)
        bitstrings, eigenvalues = bitstrings[inds[-m:]], probs[inds[-m:]]
        return bitstrings, eigenvalues
    
    def _build_vqse_result(
        self,
        target: QuantumCircuit,
        m: int,
        optimizer_result: OptimizerResult,
        optimizer_time: float,
    ) -> VQSEResult:
        result = VQSEResult()
        
        result.optimal_circuit = ansatz.copy()
        
        bitstrings, eigenvalues = self._estimate_eigenvalues(target, optimizer_result.x, m)
        result.basis_states = bitstrings
        result.eigenvalues = eigenvalues
        
        result.cost_function_evals = optimizer_result.nfev
        result.optimal_point = optimizer_result.x
        result.optimal_parameters = dict(zip(self.ansatz.parameters, optimizer_result.x))
        result.optimal_value = optimizer_result.fun
        result.optimizer_time = optimizer_time
        result.optimizer_result = optimizer_result
        
        return result


if __name__ == "__main__":
    from qiskit.circuit.library import RealAmplitudes
    from qiskit.quantum_info import Statevector
    from qiskit.primitives import Estimator, Sampler
    from qiskit import ClassicalRegister
    
    num_qubits = 4
    target = RealAmplitudes(num_qubits, reps=2)
    clbit = ClassicalRegister(1)
    target.add_bits(clbit)
    target.measure(0,0)
    assignments = dict(zip(target.parameters, 
                           np.random.rand(len(target.parameters))*np.pi))
    
    target.assign_parameters(assignments, inplace=True)

    ansatz = RealAmplitudes(num_qubits, reps=2)
    estimator = Estimator()
    sampler = Sampler()
    optimizer = ADAM(maxiter=300, amsgrad=True, lr=0.01, beta_1=0.8, beta_2=0.9)    
    
    m = 1
    vqse = VQSE(estimator, sampler, ansatz, optimizer)
    
    result = vqse.compute_eigensystem(target, m, f = lambda x: 1)
    state = DensityMatrix(target)
    print(result.eigenvalues)
    print(np.linalg.eigvalsh(state)[-m:])
    import matplotlib.pyplot as plt
    #plt.plot(vqse._err)
    #plt.show()