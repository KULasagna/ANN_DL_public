"""
Numpy implementation of Hopfield networks
"""

import numpy as np
from dataclasses import dataclass

# Useful references:
# [Hebb] "The Organization of Behavior" by D. O. Hebb (see also lecture slides)
# [LSSM] "Analysis and Synthesis of a Class of Neural Networks: Linear Systems Operating on a Closed Hypercube" by J. Li, A. N. Michel and W. Porod


@dataclass
class NormalizationParameters:
    shape: tuple[int]
    m: float
    M: float


class HopfieldNetwork:    
    def __init__(self, targets, alg='LSSM'):
        """
        Initialize a new Hopfield network for the given target patterns.

        Parameters
        ----------
        targets : array_like
            Matrix of dimension (T, D) with `D` the dimension of each target pattern and `T` the number of target patterns.

        alg : string, default: 'LSSM'
            Train algorithm to use, either "Hebb" for Hebbian learning or "LSSM" for LSSM learning (using a discretized
            Linear Saturating System Model, and used by Matlab).
        """
        self.D = 0
        self.W = 0
        self.b = 0
        self.rng = np.random.default_rng()
        self._set_weights(np.asarray(targets, dtype=float).T, alg)
        self.tf = {
            'Hebb': np.sign,
            'LSSM': lambda x: np.clip(x, -1, 1)  # Saturating linear function
        }.get(alg)
        self.int_inv_tf = {  # Integrated inverse of the transfer function
            'Hebb': np.zeros_like,  # Not defined
            'LSSM': lambda x: x * x / 2
        }.get(alg)

    def _set_weights(self, targets, alg):
        self.D, T = targets.shape
        if alg == 'Hebb':
            # Hebbian learning rule

            # initialize weights using Hebb rule
            rho = 0  # Centering the targets using np.mean(targets) (or only over axis=1)
            W = np.dot(targets - rho, targets.T - rho)  # Sum of outer products of columns of targets

            # Put diagonal elements of W to 0
            np.fill_diagonal(W, 0)
            W /= T * self.D

            self.W = W
            self.b = np.zeros(self.D)
        elif alg == 'LSSM':
            # LSSM learning rule, used by Matlab's newhop method and described in [LSSM]

            # First follow Synthesis Procedure 5.2 of [LSSM] to calculate the Ttau and Itau matrices
            # 2) Subtract last pattern from all others
            Y = targets[:, :-1] - targets[:, [-1]]

            # 3) SVD decomposition of Y
            U, S, _ = np.linalg.svd(Y, full_matrices=True)
            Sigma = np.diag(S)
            k = np.linalg.matrix_rank(Sigma)

            # 4) Sum of outer products of columns of U
            Tp = np.dot(U[:,:k], U[:,:k].T)
            Tm = np.dot(U[:,k:], U[:,k:].T)

            # 5) Calculate Ttau and Itau
            tau = 10  # Positive parameter, should be large enough
            Ttau = Tp - tau * Tm
            Itau = targets[:, -1] - Ttau @ targets[:, -1]

            # Calculate weights and biases using formulas in Remark 5.6 of [LSSM]
            h = 0.15  # Sampling time
            C1 = np.exp(h) - 1
            C2 = -(np.exp(-tau * h) - 1) / tau
            # Diagonal matrix in equation 5.9a
            Dphi = np.concatenate([np.exp(h) * np.ones(k), np.exp(-tau * h) * np.ones(self.D-k)])
            # Diagonal matrix in equation 5.9b
            Dgamma = Dphi - 1
            Dgamma[k:] /= -tau

            # Phi_tau = expm(h * Ttau)
            self.W = U @ np.diag(Dphi) @ U.T
            # Note that the formula in Remark 5.6 omits the multiplication with Itau, but this is wrong
            # Gamma_tau = int(expm(rho * Ttau), rho=0..h) Itau
            self.b = U @ np.diag(Dgamma) @ U.T @ Itau
        else:
            raise ValueError("Unknown learning algorithm, supported algorithms are 'Hebb' and 'LSSM'.")
        # Ensure the bias is a 2D row vector
        self.b = np.expand_dims(self.b, axis=0)

    def simulate(self, data, num_iter=20, sync=True):
        """
        Simulate the state evolution of the Hopfield network for the given input data.

        Parameters
        ----------
        data : array_like
            Initial state(s) of the network. For a single state, this should be a vector of dimension (D);
            for multiple states, this should be a matrix of dimension (P, D).

        num_iter : int
            Number of simulation steps to perform.

        sync : boolean
            Whether the neurons should be updated in synchronous mode (`True`) or asynchronous mode (`False`).

        Returns
        -------
        states : ndarray
            All states the network went through while simulating. This is a matrix of dimension (D, T) for
            a single state input and a tensor of dimension (P, D, T) for multiple state inputs, where T is
            equal to `num_iter + 1`.

        energies : ndarray
            The energies for each encountered state. This is a vector of dimension (T) for a single state
            input and a matrix of dimension (P, T) for multiple state inputs.
        """
        # Copy input data
        data = np.atleast_2d(np.asarray(data, dtype=float))  # Shape (P, D)

        # Collect all encountered states and their energy
        states = np.empty((*data.shape, num_iter+1))  # Shape (P, D, T)
        energies = np.empty((data.shape[0], num_iter+1))  # Shape (P, T)

        s = np.copy(data)  # Shape (P, D)
        states[:, :, 0] = s
        energies[:, 0] = self.energy(s)
        for t in range(num_iter):
            # Compute new state
            s = self._step(s, sync)
            states[:, :, t+1] = s
            # Compute new energy
            energies[:, t+1] = self.energy(s)
        return np.squeeze(states), np.squeeze(energies)

    def _step(self, s, sync):
        if sync:
            s = self.tf(s @ self.W + self.b)  # Vectorized version of W s + b
        else:
            # Randomly iterate over the neurons
            for d in self.rng.permutation(self.D):
                # Update s
                s[:, d] = self.tf(s @ self.W[:, d] + self.b)  # Vectorized version of W[d,:] s + b
        return s

    def energy(self, s):
        """
        Calculate the energy of the given states.
        """
        s = np.atleast_2d(s)  # Shape (P, D)
        e = -0.5 * np.sum((s @ self.W) * s, axis=1, keepdims=True) - s @ self.b.T  # Vectorized version of - 1/2 s^T W s - s^T b
        e += np.sum(self.int_inv_tf(s), axis=1, keepdims=True)  # Vectorized version of sum_{i=1}^D int_{0}^{s_i} g^{-1}(z) dz (see energy equation on page 1407 of [LSSM] or lecture slide 19)
        return np.squeeze(e)

    @staticmethod
    def normalize(data):
        """
        Normalize the given dataset such that it can be used in the creation of a Hopfield network.

        Parameters
        ----------
        data : array_like
            Array of patterns to normalize.

        Returns
        -------
        normalized : ndarray
            Matrix of dimension (P, D) with `P` the number of patterns and `D` the (flattened) dimension of each pattern.

        params : NormalizationParameters
            Normalization parameters that can be used to rescale the normalized data.
        """
        data = np.asarray(data, dtype=float)
        P, *shape = data.shape
        reshaped = data.reshape((P, -1))
        m = np.min(reshaped)
        M = np.max(reshaped)
        normalized = np.sign(2 * (reshaped - m) / (M - m) - 1)
        return normalized, NormalizationParameters(shape, m, M)

    @staticmethod
    def rescale(normalized, params):
        """
        Rescale the given normalized dataset back to its original format.

        Parameters
        ----------
        normalized : array_like
            Matrix of dimension (P, D) with `P` the number of normalized patterns and `D` the normalized dimension of each pattern.

        params : NormalizationParameters
            The normalization parameters to use (as returned by the `normalize` function).

        Returns
        -------
        data : ndarray
            The rescaled patterns.
        """
        normalized = np.asarray(normalized, dtype=float)
        rescaled = (normalized + 1) * (params.M - params.m) / 2 + params.m
        return rescaled.reshape([-1, *params.shape])
