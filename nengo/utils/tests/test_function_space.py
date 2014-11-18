import numpy as np
import pytest

from nengo.utils.function_space import *
from nengo.utils.distributions import Uniform
from nengo.utils.numpy import array


sigma = 0.2


def gaussian(points, center):
    return np.exp(-(points - center)**2 / (2 * sigma ** 2))


def gaussian_2D(points, center1, center2):
    """Returns a gaussian valued function in each dimension (not a multivariate
    guassian"""
    return np.array([gaussian(points, center1),
                     gaussian(points, center2)]).flatten()


@pytest.mark.parametrize("range_dim", [1, 2])
def test_function_repr(Simulator, nl, plt, range_dim):

    # parameters
    n_neurons = 2000 * range_dim  # number of neurons
    domain_dim = 1
    radius = 1

    if range_dim == 1:
        dist_args = [Uniform(-1, 1)]
        base_func = gaussian
    elif range_dim == 2:
        base_func = gaussian_2D
        dist_args = [Uniform(-1, 1), Uniform(-1, 1)]

    FS = Gen_Function_Space(base_func, domain_dim, dist_args,
                            n_functions=n_neurons, n_basis=20)

    # test input is a gaussian bumps function
    gaussians = generate_functions(base_func, 4, *dist_args)
    input_func = np.sum([func(FS.domain) for func in gaussians], axis=0)

    # evaluation points are gaussian bumps functions
    n_eval_points = 400
    funcs = []
    for _ in range(n_eval_points):
        gaussians = generate_functions(base_func, 4, *dist_args)
        funcs.append(np.sum([func(FS.domain)
                             for func in gaussians], axis=0).flatten())
    eval_points = FS.signal_coeffs(np.array(funcs).T)

    # vector space coefficients
    signal_coeffs = FS.signal_coeffs(input_func).flatten()
    encoders = FS.encoder_coeffs()

    f_radius = np.linalg.norm(signal_coeffs)  # radius to use for ensemble

    with nengo.Network() as model:
        # represents the function
        f = nengo.Ensemble(n_neurons=n_neurons, dimensions=FS.n_basis,
                           encoders=encoders, radius=f_radius,
                           eval_points=eval_points, label='f')
        signal = nengo.Node(output=signal_coeffs)
        nengo.Connection(signal, f)

        probe_f = nengo.Probe(f, synapse=0.1)

    sim = Simulator(model)
    sim.run(6)

    reconstruction = FS.reconstruct(sim.data[probe_f][400]).flatten()
    true_f = input_func.flatten()

    plt.saveas = "func_repr_range_dim_%s.pdf" % range_dim

    if range_dim == 1:
        plt.plot(FS.domain, reconstruction, label='model_f')
        plt.plot(FS.domain, true_f, label='true_f')
        plt.legend(loc='best')
    elif range_dim == 2:
        plt.plot(FS.domain, reconstruction[:len(FS.domain)], label='model_f')
        plt.plot(FS.domain, true_f[:len(FS.domain)], label='true_f')
        plt.plot(FS.domain, reconstruction[len(FS.domain):], label='model_f')
        plt.plot(FS.domain, true_f[len(FS.domain):], label='true_f')
        plt.legend(loc='best')

    assert np.allclose(true_f, reconstruction, atol=0.3)


def test_function_gen_eval(plt):
    values = function_values(generate_functions(gaussian, 20, Uniform(-1, 1)),
                             uniform_cube(1, 1, 0.001))
    plt.plot(function_values(generate_functions(gaussian, 20, Uniform(-1, 1)),
                             uniform_cube(1, 1, 0.001)),
             label='function_values')


def test_uniform_cube():
    points = uniform_cube(2, 1, 0.1)
    plt.scatter(points[0, :], points[1, :], label='points')


def test_fourier_basis(plt):
    # parameters
    n_neurons = 100  # number of neurons
    domain_dim = 1
    range_dim = 1
    radius = 1

    FS = Fourier(gaussian, domain_dim, [Uniform(-1, 1)],
                 n_functions=n_neurons)

    true = FS.fns[:, 10]
    model = FS.reconstruct(FS.signal_coeffs(FS.fns[:, 10])).flatten()

    plt.figure('Testing Fourier Basis')
    plt.plot(FS.domain, true, label='Function')
    plt.plot(FS.domain, model, label='reconstruction')
    plt.legend(loc='best')
    plt.savefig('utils.test_function_space.test_fourier_basis.pdf')

    # crop because of Gibbs phenomenon
    assert np.allclose(true[200:-200], model[200:-200], atol=0.2)
