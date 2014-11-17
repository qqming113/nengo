from __future__ import absolute_import

import numpy as np

import matplotlib.pyplot as plt

import nengo

from nengo.utils.numpy import array


def generate_functions(function, n, *arg_dists):
    """
    Parameters:

    function: callable,
       the function to be used as a basis, ex. gaussian

    n: int,
       number of functions to generate

    arg_dists: instances of nengo distributions
       distributions to sample arguments from, ex. mean of a gaussian function
    """

    # get argument samples to make different functions
    arg_samples = np.array([arg_dist.sample(n) for arg_dist in arg_dists]).T

    functions = []
    for i in range(n):
        def func(points, i=i):
            args = [points]
            args.extend(arg_samples[i])
            return function(*args)
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

    Returns:
    -------
    ndarray of shape (domain_dim, radius/d)

    """

    if domain_dim == 1:
        domain_points = np.arange(-radius, radius, d)
        domain_points = array(domain_points, min_dims=2)
    else:
        axis = np.arange(-radius, radius, d)
        # uniformly spaced points of a hypercube in the domain
        grid = np.meshgrid(*[axis for _ in range(domain_dim)])
        domain_points = np.vstack(map(np.ravel, grid))
    return domain_points


def function_values(functions, points):
    """The values of the function on ``points``.

    Returns:
    --------
    ndarray of shape (n_points * output_dim, n_functions).
    output_dim is the dimension of the range of the functions"""

    range_dim = functions[0](points[0]).shape[0]
    values = np.empty((len(points) * range_dim, len(functions)))
    for i, function in enumerate(functions):
        values[:, i] = function(points).flatten()
    return values


class Function_Space(object):
    """A helper class for using function spaces in nengo.

    Parameters:
    -----------
    fn: callable,
      The function that will be used for tiling the space.
      Must return an ndarray of shape (n_points, range_dim)

    dist_args: list of nengo Distributions
       The distributions to sample functions from.

    domain_dim: int,
      The dimension of the domain of ``fn``.

    n_functions: int, optional
      Number of functions used to tile the space.

    n_basis: int, optional
      Number of orthonormal basis functions to use

    d: float, optional
       the discretization factor (used in spacing the domain points)

    radius: float, optional
       2 * radius is the length of a side of the hypercube
    """

    def __init__(self, fn, domain_dim, dist_args, n_functions=200, n_basis=20,
                 d=0.001, radius=1):

        self.domain = uniform_cube(domain_dim, radius, d)
        self.fns = function_values(generate_functions(fn, n_functions,
                                                      *dist_args),
                                   self.domain)

        self.dx = d ** self.domain.shape[1] # volume element for integration
        self.n_basis = n_basis

    def get_basis(self):
        raise NotImplementedError("Must be implemented by subclasses")

    def reconstruct(self, coefficients):
        raise NotImplementedError("Must be implemented by subclasses")

    def encoder_coeffs(self):
        raise NotImplementedError("Must be implemented by subclasses")

    def signal_coeffs(self, signal):
        raise NotImplementedError("Must be implemented by subclasses")


class Gen_Function_Space(Function_Space):
    """A function space subclass where the basis is derived from the SVD"""

    def __init__(self, fn, domain_dim, dist_args, n_functions=200, n_basis=20,
                 d=0.001, radius=1):

        super(Gen_Function_Space, self).__init__(fn, domain_dim, dist_args,
                                                 n_functions, n_basis,
                                                 d, radius)

        #basis must be orthonormal
        self.U, self.S, V = np.linalg.svd(self.fns)
        self.basis = self.U[:, :self.n_basis] / np.sqrt(self.dx)


    def select_top_basis(self, n_basis):
        self.n_basis = n_basis
        self.basis = self.U[:, :n_basis] / np.sqrt(self.dx)

    def get_basis(self):
        return self.basis

    def singular_values(self):
        return self.S

    def reconstruct(self, coefficients):
        """Linear combination of the basis functions"""
        return np.dot(self.basis, coefficients)

    def encoder_coeffs(self):
        """Project encoder functions onto basis to get encoder coefficients."""
        return self.signal_coeffs(self.fns)

    def signal_coeffs(self, signal):
        """Project a given signal onto basis to get signal coefficients.
           Size returned is (n_signals, n_basis)"""
        signal = array(signal, min_dims=2)
        return  np.dot(signal.T, self.basis) * self.dx
        # if signal_coeff.shape[0] == 1:
            # signal_coeff = signal_coeff.reshape((self.n_basis,))

        # return signal_coeff


class Fourier(Function_Space):

    def __init__(self, fn, domain_dim, dist_args, n_functions=200, n_basis=20,
                 d=0.001, radius=1):

        super(Fourier, self).__init__(fn, domain_dim, dist_args,
                                      n_functions, n_basis,
                                      d, radius)


    def reconstruct(self, coefficients):
        """Linear combination of the basis functions"""
        return np.fft.irfft(coefficients, len(self.domain))

    def encoder_coeffs(self):
        """Project encoder functions onto basis to get encoder coefficients."""
        return self.signal_coeffs(self.fns)

    def signal_coeffs(self, signal):
        """Project a given signal onto basis to get signal coefficients.
           Size returned is (n_signals, n_basis)"""
        signal = array(signal, min_dims=2)
        return np.fft.rfft(signal.T)[:, :self.n_basis]
        # return np.absolute(X) # compute magnitude
