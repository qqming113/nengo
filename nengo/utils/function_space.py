from __future__ import absolute_import

import numpy as np

import matplotlib.pyplot as plt

import nengo

# NOTES:
# range of functions always has dim 1, what about arbitrary dims
# fourier transform
# picking evaluation points for the coefficients of the function space? should these be linear
# combinations of the original basis??

def gaussian_functions_1D(n_basis, sigma, radius):
    centers = np.random.uniform(-radius, radius, (n_basis,))
    functions = []

    for i in range(n_basis):
        def func(point, i=i):
            center = centers[i]
            return np.exp(-(point - center)**2 / (2 *sigma**2))
        functions.append(func)
    return functions

def uniform_cube(domain_dim, radius=1, d=0.001):
    """Returns uniformly spaced points in a hypercube.

    The hypercube is defined by the given radius and dimension.

    Parameters:
    ----------
    domain_dim: int
       the dimension of the domain

    radius: float, optional
       2 * radius is the length of a side of the hypercube

    d: float, optional
       the discretization spacing (a small float)
    """
    if domain_dim == 1:
        domain_points = np.arange(-radius, radius, d)
        domain_points = domain_points.reshape(domain_points.shape[0], 1)
    else:
        axis = np.arange(-radius, radius, d)
        # uniformly spaced points in the hypercube of the domain
        grid = np.meshgrid(*[axis for _ in range(domain_dim)])
        domain_points = np.vstack(map(np.ravel, grid))
    return domain_points


class Function_Space(object):
    """A helper class for using function spaces in nengo.

    Parameters:
    -----------
    fns: list of callables
       The functions that will be used as encoder functions for each neuron.
       There should as many functions as neurons.

    domain_points: ndarray
       Array of points that define the domain of the functions

    d: float
       the discretization factor (used in integration)
    """

    def __init__(self, fns, domain_points, d):
        self.fns = fns
        self.domain = domain_points
        self.dx = d ** domain_points.shape[1] # volume element for integration
        self.n_basis = 20 # default number of basis

        self.values = self._function_values(self.fns, self.domain)
        self.U, self.S, V = np.linalg.svd(self.values)
        self.basis = self.U[:, :self.n_basis] / np.sqrt(self.dx)

    def _function_values(self, functions, domain_points):
        """The values of the function on the domain
           shape (n_points, n_neurons)"""
        values = np.empty((len(domain_points), len(functions)))
        for j, point in enumerate(domain_points):
            for i, function in enumerate(functions):
                values[j, i] = function(point)
        return values

    def select_top_basis(self, n_basis):
        self.n_basis = n_basis
        self.basis = self.U[:, :n_basis] / np.sqrt(self.dx)

    def get_basis(self):
        return self.basis

    def singular_values(self):
        return self.S

    def reconstruct(self, coefficients):
        """Linear combination of the basis functions according to
           the coefficients"""
        return np.dot(self.basis, coefficients)

    def encoder_coeffs(self):
        """Project encoder functions onto basis to get encoder coefficients."""
        return np.dot(self.values.T, self.basis) * self.dx

    def signal_coeffs(self, signal):
        """Project a given signal onto basis to get signal coefficients.
           Size returned is (n_signals, n_basis)"""
        signal_coeff = np.dot(signal.T, self.basis) * self.dx
        if signal_coeff.shape[0] == 1:
            signal_coeff = signal_coeff.reshape((self.n_basis,))
        return signal_coeff
