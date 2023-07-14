from __future__ import annotations

from qiskit import QuantumCircuit, ClassicalRegister

import numpy as np
from qiskit.algorithms.eigensolvers.eigensolver import EigensolverResult
from qiskit.algorithms.list_or_dict import ListOrDict

from qiskit.primitives import BaseSampler
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import DensityMatrix

from qiskit.algorithms import AlgorithmError
from qiskit.algorithms import VariationalAlgorithm, VariationalResult
from qiskit.algorithms.optimizers import ADAM, OptimizerResult
from qiskit.algorithms.gradients import BaseSamplerGradient
from qiskit.algorithms.utils import validate_bounds, validate_initial_point
from qiskit.algorithms.utils.set_batching import _set_default_batchsize
from qiskit.algorithms.eigensolvers import Eigensolver, EigensolverResult

from collections.abc import Callable, Sequence
from typing import Any, Tuple
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
        sampler: BaseSampler,
        ansatz: QuantumCircuit,
        optimizer: ADAM,
        m: int = 1,
        *,
        q: Sequence[float] | None = None,
        r: Sequence[float] | None = None,
        f: Callable[[float], float] = lambda x: x,
        delta: float | None = 0.01,
        adaptive_update_frequency: int = 30,
        gradient: BaseSamplerGradient | None = None,
        initial_point: Sequence[float] | None = None,
        callback: Callable[[int, np.ndarray, float, dict[str, Any], float], None] | None = None,
        **sampler_options,
    ):
        """
        Args:
            sampler: The sampler primitive.
            ansatz: The parametrized circuit used as the ansatz circuit to prepare the state.
            optimizer: An ADAM optimizer used for the classical minimization of the cost function.
            m: The number of eigenvalues to compute. Computes the lowest m eigenvalues.
            q: The sequence of values used in the global Hamiltonian for the VQSE algorithm. Should
                have length m and be sorted in descending ording.
            r: The sequence of values used in the local Hamiltonian for the VQSE algorithm. Should
                have length equal to the number of qubits in the ansatz.
            f: The function which sets the adiabatic transformation of the Hamiltonian from local to
                global. Should satisfy f(0) = 0 and f(1) = 1.
            delta: If no r is provided, the default values of r are set according to this spacing. 
            adaptive_update_frequency: Sets the frequency with which the VQSE adaptive global hamiltonian
                is updated. Default is every 30 iterations of the ADAM optimizer.
            initial_point: The initial point of the parameters in the provided ansatz.
            callback: A callback that can access the intermediate data
                during the optimization. Four parameter values are passed to the callback as
                follows during each evaluation by the optimizer: the evaluation count,
                the optimizer parameters for the ansatz, the estimated value,
                the estimation metadata, and the current step.
            sampler_options: Remaining kwargs are passed as kwargs to the sampler whenever it is 
                called.
            
        """
        super().__init__()

        self.sampler = sampler
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.gradient = gradient
        # this has to go via getters and setters due to the VariationalAlgorithm interface
        self.initial_point = initial_point
        self.callback = callback

        self._sampler_options = sampler_options

        self.adaptive_update_frequency = adaptive_update_frequency
        self._measured_bitstrings: Sequence[int] = list(range(m))

        self.m = m
        self.f = f
        
        num_qubits = self.ansatz.num_qubits
        
        if delta is None:
            delta = 1/(2.*m)
        
        if r is None:
            if m == 1:
                self._r = [1]*num_qubits
            else:
                self._r = [1 + i*delta for i in range(num_qubits)]
        else:
            if len(r) != num_qubits:
                raise AlgorithmError("r must have length equal to the number of qubits.")
            self._r = r

        if q is None:
            self._q = []
            
            import itertools

            i = 0
            k = 0
            while i < m:
                for comb in itertools.combinations(range(num_qubits), k):
                    bitstring = 0
                    for q in comb:
                        bitstring = (1 << q) | bitstring
                    
                    i += 1
                    energy = 1
                    for j in range(num_qubits):
                        energy += (2*int((bitstring >> j) & 1) - 1)*self._r[j]

                    self._q.append(1 -  energy)

                k += 1
        else:
            if len(q) != m:
                raise AlgorithmError("q much have length equal to m.")
            self._q = q
        



    @property
    def initial_point(self) -> Sequence[float] | None:
        return self._initial_point

    @initial_point.setter
    def initial_point(self, value: Sequence[float] | None) -> None:
        self._initial_point = value
        
    def compute_eigenvalues(
        self, 
        target: QuantumCircuit | DensityMatrix, 
        aux_operators: ListOrDict[BaseOperator | PauliSumOp] | None = None
    ) -> VQSEResult:
        self._check_target(target)
        
        # Validate target against ansatz  
        
        initial_point = validate_initial_point(self.initial_point, self.ansatz)

        bounds = validate_bounds(self.ansatz)
         
        start_time = time()

        evaluate_energy = self._get_evaluate_energy(target)

        if self.gradient is not None:
            evaluate_gradient = self._get_evaluate_gradient(target)
        else:
            evaluate_gradient = None

        # perform optimization
        if callable(self.optimizer):
            optimizer_result = self.optimizer(
                fun=evaluate_energy, x0=initial_point, jac=evaluate_gradient, bounds=bounds
            )
        else:
            # we always want to submit as many samples per job as possible for minimal
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
            target, optimizer_result, optimizer_time
        )
    
    
    def _prepare_circuit(self, target: QuantumCircuit) -> QuantumCircuit:
        circuit = target.compose(self.ansatz)
        if circuit.num_qubits > circuit.num_clbits:
            circuit.add_register(ClassicalRegister(circuit.num_qubits - circuit.num_clbits))
        circuit.measure_all(add_bits=False)
        return circuit
        
    def _get_evaluate_gradient(
        self,
        target: QuantumCircuit | DensityMatrix,
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
        circuit = self._prepare_circuit(target)

        def evaluate_gradient(parameters: np.ndarray) -> np.ndarray:
            # broadcasting not required for the estimator gradients
            try:
                job = self.gradient.run([circuit], [parameters], **self._sampler_options)
                gradients = job.result().gradients
            except Exception as exc:
                raise AlgorithmError("The primitive job to evaluate the gradient failed!") from exc

            return gradients[0]

        return evaluate_gradient
    
    def _get_evaluate_energy(
        self,
        target: QuantumCircuit | DensityMatrix,
    ) -> Callable[[np.ndarray], np.ndarray | float]:
        """Returns a function handle to evaluate the energy at given parameters for the ansatz.
        This is the objective function to be passed to the optimizer that is used for evaluation.

        Args:
            ansatz: The ansatz preparing the quantum state.

        Returns:
            A callable that computes and returns the energy of the hamiltonian of each parameter.

        Raises:
            AlgorithmError: If the primitive job to evaluate the energy fails.
        """
        # avoid creating an instance variable to remain stateless regarding results
        eval_count = 0

        def evaluate_energy(parameters: np.ndarray) -> np.ndarray | float:
            nonlocal eval_count

            t = self.optimizer._t/self.optimizer._maxiter

            # handle broadcasting: ensure parameters is of shape [array, array, ...]
            num_parameters = len(self.ansatz.parameters)
            parameters = np.reshape(parameters, (-1, num_parameters)).tolist()

            try:
                if isinstance(target, QuantumCircuit):
                    outcomes, metadata = self._get_counts_from_quantum_circuit(target, parameters)
                elif isinstance(target, DensityMatrix):
                    outcomes, metadata = self._get_counts_from_density_matrix(target, parameters)
            except Exception as exc:
                raise AlgorithmError("The primitive job to evaluate the energy failed!") from exc

            # This might only work with ADAM optimizer; need to keep track of time some other way.
            t = self.optimizer._t/self.optimizer._maxiter
            
            if isinstance(outcomes, list):
                self._measured_bitstrings = self._compute_bitstrings(outcomes[0])
                if self.callback is not None:
                    for param, outcome, meta in zip(parameters, outcomes, metadata):
                        eval_count += 1
                        self.callback(eval_count, param, outcome, meta, t)
                return [self._compute_expected_energy(dist, t) for dist in outcomes]
            elif isinstance(outcomes, dict):
                self._measured_bitstrings = self._compute_bitstrings(outcomes)
                if self.callback is not None:
                    eval_count += 1
                    self.callback(eval_count, parameters[0], outcomes, metadata, t)
                return self._compute_expected_energy(outcomes, t)

        return evaluate_energy
    
    def _compute_bitstrings(self,
        outcomes: dict[int, float],
    ) -> list[int]:
        """
        Computes the bitstrings corresponding to the m most frequent measurements, which
        are passed as a probabilities dictionary with integer keys.
        
        Args:
            outcomes: A probability dictionary corresponding to measurement outcomes containing
                integers as keys and floats as values.
        
        Returns:
            A list of the m most frequently measured bitstrings in outcomes.
        """
        sorted_outcomes = sorted(list(outcomes.items()), key=lambda x: x[1])
        return [x[0] for x in sorted_outcomes[-self.m:]]
    
    def _compute_expected_energy(
        self,
        outcomes: dict[int, float],
        t: float,
    ) -> float:
        """
        Computes the expected energy of a given set of measurement outcomes at a given time
        with respect to the VQSE adaptive hamiltonian.
        
        Args:
            outcomes: A probability dictionary corresponding to measurement outcomes containing
                integers as keys and floats as values.
            t: The current timestep.
        
        Returns:
            The expectation of the VQSE hamiltonian with respect to the provided outcomes.
        """
        global_energy = 1
        for i,z in enumerate(self._measured_bitstrings):
            if z in outcomes: # If not, it has probability 0 and does not contribute
                global_energy -= self._q[i]*outcomes[z]
            
        local_energy = 1
        for b,p in outcomes.items():
            for j in range(self.ansatz.num_qubits):
                local_energy += (2*int((b >> j) & 1) - 1)*self._r[j]*p
        
        return (1 - self.f(t))*local_energy + self.f(t)*global_energy

    def _check_target(self, target: QuantumCircuit | DensityMatrix):
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
        target: QuantumCircuit | DensityMatrix,
        parameters: np.ndarray,
    ) -> Tuple[Sequence[int], Sequence[float]]:
        """
        Estimates the eigenvalues of target given a set of parameters to be provided to the ansatz. 
        
        Args:
            target: The quantum circuit which produces the desired state or a copy of the DensityMatrix directly.
            parameters: A list of parameters to be provided to the ansatz.
        
        Returns:
            A tuple where the first element is a list of the m most frequently measured outcomes and
            the second element is a list containing the corresponding probabilities.
        """
        if isinstance(target, QuantumCircuit):
            probs_dict, _ = self._get_counts_from_quantum_circuit(target, parameters)
        elif isinstance(target, DensityMatrix):
            probs_dict, _ = self._get_counts_from_density_matrix(target, parameters)
        
        [*bitstrings], [*probs] = zip(*probs_dict.items())
        bitstrings = np.array(bitstrings)
        probs = np.array(probs)
        
        inds = np.argsort(probs)
        bitstrings, eigenvalues = bitstrings[inds[-self.m:]], probs[inds[-self.m:]]
        return bitstrings, eigenvalues
    
    def _get_counts_from_density_matrix(self, 
        target: DensityMatrix, 
        parameters: np.ndarray
    ) -> Tuple[Sequence[dict[int, float]], Sequence[dict[str, Any]]] | Tuple[dict[int, float], dict[str, Any]]:
        """
        Returns measurement counts when the target is a density matrix.
        
            Args:
                target: The target, which is a DensityMatrix.
                parameters: The parameters to be passed to the ansatz circuit. Can be either a 1d array of
                    parameters or a list of 1d parameters.
                
            Returns:
                If one set of parameters is passed, return a pair of dictionaries, where the first element is
                the measurement outcomes and the second element is metadata. 
                
                If multiple sets of parameters are passed, a list of dictionary pairs containing measurement
                outcomes and metadata for each set of parameters is returned.
        """
        if not isinstance(parameters[0], list):
            parameters = [parameters]
        outcomes = []
        metadata = []
        for params in parameters:
            assignments = dict(zip(self.ansatz.parameters, params))
            result = target.evolve(self.ansatz.assign_parameters(assignments)).probabilities()
            outcomes.append({i: result[i] for i in range(len(result))})
            metadata.append({})
        
        if len(outcomes) > 1:
            return outcomes, metadata
        else:
            return outcomes[0], metadata[0]
    
    def _get_counts_from_quantum_circuit(self, 
        target: QuantumCircuit, 
        parameters: np.ndarray
    ) -> Tuple[Sequence[dict[int, float]], Sequence[dict[str, Any]]] | Tuple[dict[int, float], dict[str, Any]]:
        """
        Returns measurement counts when the target is a quantum circuit. 
            Args:
                target: The target, which is a QuantumCircuit.
                parameters: The parameters to be passed to the ansatz circuit. Can be either a 1d array of
                    parameters or a list of 1d parameters.
                
            Returns:
                If one set of parameters is passed, return a pair of dictionaries, where the first element is
                the measurement outcomes and the second element is metadata. 
                
                If multiple sets of parameters are passed, a list of dictionary pairs containing measurement
                outcomes and metadata for each set of parameters is returned.
        """
        if len(np.shape(parameters)) > 1:
            batch_size = len(parameters)
        else:
            batch_size = 1

        circuit = self._prepare_circuit(target)
        job = self.sampler.run(batch_size * [circuit], parameters, **self._sampler_options)
        sampler_result = job.result()
        if len(sampler_result.quasi_dists) > 1:
            return sampler_result.quasi_dists, sampler_result.metadata
        else:
            return sampler_result.quasi_dists[0], sampler_result.metadata[0]

    
    def _build_vqse_result(
        self,
        target: QuantumCircuit | QuantumCircuit,
        optimizer_result: OptimizerResult,
        optimizer_time: float,
    ) -> VQSEResult:
        """
        Builds the VQSEResult to be returned.
        """
        result = VQSEResult()
        
        assignments = dict(zip(self.ansatz.parameters, optimizer_result.x))
        result.optimal_circuit = self.ansatz.assign_parameters(assignments)
        
        bitstrings, eigenvalues = self._estimate_eigenvalues(target, optimizer_result.x)
        result.basis_states = [f'{z:0{self.ansatz.num_qubits}b}' for z in bitstrings]
        result.eigenvalues = eigenvalues
        
        result.cost_function_evals = optimizer_result.nfev

        optimal_parameters = dict(zip(self.ansatz.parameters, optimizer_result.x))
        result.optimal_parameters = optimal_parameters
        result.optimal_circuit = self.ansatz.assign_parameters(optimal_parameters)
        result.optimal_point = optimizer_result.x
        result.optimal_value = optimizer_result.fun
        result.optimizer_time = optimizer_time
        result.optimizer_result = optimizer_result
        
        return result
