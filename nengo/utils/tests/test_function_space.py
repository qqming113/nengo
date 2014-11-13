import numpy as np
import pytest

from nengo.utils.function_space import *
from nengo.utils.distributions import Uniform
from nengo.utils.numpy import rmse

#NOTES:
# Generalize to functions with range dim more than 1
# split up function value decoding as a linear combination of the basis
# test by picking eval points properly

sigma = 0.2

def gaussian(point, center):
    return np.exp(-(point - center)**2 / (2 * sigma ** 2))


def test_function_repr(Simulator, nl, plt):

    #parameters
    n_neurons = 1000 # number of neurons
    domain_dim = 1
    range_dim = 1
    radius = 1

    FS = Gen_Function_Space(gaussian, domain_dim, [Uniform(-1, 1)],
                            n_functions=n_neurons, n_basis=8)

    # test input is gaussian bumps
    gaussians = generate_functions(gaussian, 4, Uniform(-1, 1))
    input_func = np.sum([func(FS.domain) for func in gaussians], axis=0)

    # evaluation points are gaussian bumps
    n_eval_points = 200
    funcs = []
    for _ in range(n_eval_points):
        gaussians = generate_functions(gaussian, 4, Uniform(-1, 1))
        funcs.append(np.sum([func(FS.domain)
                             for func in gaussians], axis=0).flatten())
    eval_points = FS.signal_coeffs(np.array(funcs).T)

    # vector space coefficients
    signal_coeffs = FS.signal_coeffs(input_func).flatten()
    encoders = FS.encoder_coeffs()

    f_radius = np.linalg.norm(signal_coeffs) # radius to use for ensemble

    with nengo.Network() as model:
        #represents the function
        f = nengo.Ensemble(n_neurons=n_neurons, dimensions=FS.n_basis,
                           encoders=encoders, radius=f_radius,
                           eval_points=eval_points, label='f')
        signal = nengo.Node(output=signal_coeffs)
        nengo.Connection(signal, f)

        probe_f = nengo.Probe(f, synapse=0.1)

    sim = Simulator(model)
    sim.run(6)

    reconstruction = FS.reconstruct(sim.data[probe_f][400])
    true_f = input_func

    reconstruction = reconstruction.flatten()
    true_f = true_f.flatten()

    plt.plot(FS.domain, reconstruction, label='model_f')
    plt.plot(FS.domain, true_f, label='true_f')
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
    plt.scatter(points[0,:], points[1,:], label='points')


# def test_fourier_basis(plt):
#     #parameters
#     n_neurons = 100 # number of neurons
#     domain_dim = 1
#     range_dim = 1
#     radius = 1

#     FS = Fourier(gaussian, domain_dim, [Uniform(-1, 1)],
#                  n_functions=n_neurons)

#     true = FS.fns[:, 10]
#     model = FS.reconstruct(FS.signal_coeffs(FS.fns[:, 10])).flatten()

#     plt.figure('Testing Fourier Basis')
#     plt.plot(FS.domain, true, label='Function')
#     plt.plot(FS.domain,
#              model,
#              label='reconstruction')
#     plt.legend(loc='best')
#     plt.savefig('utils.test_function_space.test_fourier_basis.pdf')

#     assert np.allclose(true, model, atol=0.2)

# def test_function_values(Simulator, nl):

#     #parameters
#     n_neurons = 100 # number of neurons
#     domain_dim = 1
#     range_dim = 1
#     radius = 1

#     FS = Gen_Function_Space(gaussian, domain_dim, [Uniform(-1, 1)],
#                         n_functions=n_neurons)

#     input_func = lambda x: x ** 2
#     signal_coeffs = FS.signal_coeffs(input_func(FS.domain)).flatten()
#     encoders = FS.encoder_coeffs()

#     # eval_points = FS.eval_points()

#     f_radius = np.linalg.norm(signal_coeffs) # radius to use for function rep.

#     # #figure out what the value of the function should be at x
#     # #relies on x being in a uniformly discretized domain, but can generalize
#     # #this to multiple dimensions
#     # def func_apply(f_by_x):
#     #     coeff = f_by_x[0:FS.n_basis]
#     #     f = FS.reconstruct(coeff)
#     #     x = f_by_x[-domain_dim:]
#     #     index = min(int(x / FS.dx) + len(FS.domain) / 2, len(FS.domain) - 1)
#     #     return f[index]

#     # n_samples = 300
#     # eval_points_fx = [] #eval_points for f_by_x ensemble
#     # for point in eval_points:
#     #     # n_samples argument samples for each function
#     #     for _ in range(n_samples):
#     #         eval_points_fx.append(np.append(point, np.random.uniform(-1, 1)))

#     # eval_points_fx = np.array(eval_points_fx)

#     with nengo.Network() as model:
#         #represents the function
#         f = nengo.Ensemble(n_neurons=n_neurons, dimensions=FS.n_basis,
#                            encoders=encoders, radius=f_radius, label='f')
#                            # eval_points=eval_points, label='f')
#         #represents the argument
#         x = nengo.Ensemble(n_neurons=200, dimensions=domain_dim,
#                            radius=radius, label='x')
#         #represents the function and the argument
#         f_by_x = nengo.Ensemble(n_neurons=n_neurons,
#                                 dimensions=FS.n_basis + domain_dim,
#                                 radius=f_radius + radius, label='f_by_x')
#                                 # eval_points=eval_points_fx)
#         #represents the function applied to the argument
#         fx = nengo.Ensemble(n_neurons=200, dimensions=range_dim,
#                             radius=radius, label='fx')

#         in_x = nengo.Node(output=lambda t: np.sin(t))
#         signal = nengo.Node(output=signal_coeffs)

#         nengo.Connection(signal, f)
#         nengo.Connection(in_x, x)
#         nengo.Connection(x, f_by_x[-1])
#         nengo.Connection(f, f_by_x[:FS.n_basis])
#         nengo.Connection(f_by_x, fx, function=func_apply)

#         probe_model = nengo.Probe(fx, synapse=0.1)
#         probe_f_by_x = nengo.Probe(f_by_x, synapse=0.1)

#     sim = Simulator(model)
#     sim.run(6)

#     true_val = input_func(sim.data[probe_f_by_x][:, -1])
#     reconstruction = FS.reconstruct(sim.data[probe_f_by_x][400, :FS.n_basis])
#     model_val = sim.data[probe_model]

#     true_f = input_func(FS.domain)

#     plt.figure()
#     plt.plot(sim.trange(), true_val, label='true')
#     plt.plot(sim.trange(), model_val, label='model')
#     plt.plot(sim.trange(), sim.data[probe_f_by_x][:, -1], label='x')
#     plt.legend(loc='best')
#     plt.savefig('utils.test_function_space.test_function'
#                 '_values_%s.pdf' % str(nl))

#     plt.figure()
#     plt.plot(FS.domain, reconstruction, label='model_f')
#     plt.plot(FS.domain, input_func(FS.domain), label='true_f')
#     plt.legend(loc='best')
#     plt.savefig('utils.test_function_space.test_function'
#                 '_representation_%s.pdf' % str(nl))

#     assert np.allclose(true_f, reconstruction, atol=0.2)
#     # assert np.allclose(true_val, model_val, atol=0.5)
