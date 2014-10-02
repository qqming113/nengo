import numpy as np
import matplotlib.pyplot as plt

from nengo.utils.function_space import *
from nengo.utils.testing import Plotter


def test_function_representation(Simulator, nl):

    #parameters
    n_neurons = 1000 # number of neurons
    domain_dim = 1
    range_dim = 1
    radius = np.pi
    d = 0.001 # discretization

    #setup functions and coefficients
    functions = gaussian_functions_1D(n_neurons, 0.5, radius)
    domain_points = uniform_cube(domain_dim, radius, d)
    FS = Function_Space(functions, domain_points, d)
    input_func = lambda x: 5* x**2 + 2 * x**3 - 3 * x** 5
    func_signal = input_func(domain_points)
    encoders = FS.encoder_coeffs()
    signal_coeffs = FS.signal_coeffs(func_signal)

    #pick evaluation points to be polynomial functions
    n_eval_points = 300
    coeff_matrix = np.random.uniform(low=-5, high=5,
                                     size=(n_eval_points, n_eval_points))
    poly_val = np.empty((len(domain_points), n_eval_points))
    for i in range(n_eval_points):
        poly_val[:, i] = np.polynomial.polynomial.polyval(domain_points.flatten(),
                                                          coeff_matrix[i, :])
    poly_coeff = FS.signal_coeffs(poly_val) # size (n_eval_points, n_basis)
    poly_coeff /= np.linalg.norm(poly_coeff, axis=1)[:, np.newaxis]

    f_radius = np.linalg.norm(signal_coeffs) # radius to use for function rep.

    with nengo.Network() as model:
        f = nengo.Ensemble(n_neurons=n_neurons, dimensions=FS.n_basis,
                           encoders=encoders, radius=f_radius)
                           #eval_points=poly_coeff)
        signal = nengo.Node(output=signal_coeffs)
        nengo.Connection(signal, f)
        probe = nengo.Probe(f, synapse=0.1)

    sim = Simulator(model)
    sim.run(0.5)

    true_val = func_signal
    model_val = FS.reconstruct(sim.data[probe][300,:])

    # with Plotter(Simulator) as plt:
    #     plt.plot(domain_points, true_val, label='true')
    #     plt.plot(domain_points, model_val, label='model')
    #     plt.legend(loc='best')
    #     plt.savefig('utils.test_function_space.test_function'
    #                 '_representation_%s.pdf' % str(nl))
    #     plt.close()

    return true_val, model_val, domain_points
   # assert np.allclose(true_val, model_val, atol=0.2)

true_val, model_val, domain_points = test_function_representation(nengo.Simulator, nengo.Direct())
plt.plot(domain_points, true_val, label='true')
plt.plot(domain_points, model_val, label='model')
plt.show()

# def test_function_values(Simulator, nl):

#     #parameters
#     n_neurons = 500 # number of neurons
#     domain_dim = 1
#     range_dim = 1
#     radius = np.pi
#     d = 0.001 # discretization

#     #setup functions and coefficients
#     functions = gaussian_functions_1D(n_neurons, 0.5, radius)
#     domain_points = uniform_cube(domain_dim, radius, d)

#     FS = Function_Space(functions, domain_points, d)

#     input_func = np.sin
#     signal = input_func(domain_points)

#     encoders = FS.encoder_coeffs()
#     signal_coeffs = FS.signal_coeffs(signal)

#     f_radius = np.linalg.norm(signal_coeffs)

#     with nengo.Network() as model:
#         f = nengo.Ensemble(n_neurons=n_neurons, dimensions=FS.n_basis,
#                            encoders=encoders, radius=f_radius)
#         x = nengo.Ensemble(n_neurons=100, dimensions=domain_dim,
#                            radius=radius)
#         f_by_x = nengo.Ensemble(n_neurons=n_neurons,
#                                 dimensions=FS.n_basis + domain_dim,
#                                 radius=f_radius + radius)
#         fx = nengo.Ensemble(n_neurons=500, dimensions=range_dim,
#                             radius=f_radius + radius)

#         step = nengo.Node(output=lambda t: t - 0.5)
#         signal = nengo.Node(output=signal_coeffs)

#         nengo.Connection(signal, f)
#         nengo.Connection(step, x)
#         nengo.Connection(x, f_by_x[-1])
#         nengo.Connection(f, f_by_x[:n_basis])
#         nengo.Connection(f_by_x, fx, function=lambda x: input_func(x))

#         # probe = nengo.Probe(f, synapse=0.1)
#         probe_model = nengo.Probe(fx, synapse=0.1)
#         probe_arg = nengo.Probe(x, synapse=0.1)

#         sim = Simulator(model)
#         sim.run(0.5)

#         true_val = input_func(domain_points)
#         model_val = FS.reconstruct(sim.data[probe][300,:])

#         with Plotter(Simulator) as plt:
#             plt.plot(domain_points, true_val,
#                      domain_points, model_val)
#             plt.savefig('utils.test_function_space.test_function'
#                         '_representation_%s.pdf' % nl)
#             plt.close()

#         assert np.allclose(true_val, model_val, atol=0.2)

# plt.plot(sim.trange(), sim.data[probe_model], label='function_values')
# plt.plot(sim.trange(), input_func(sim.data[probe_arg]), label='true funcx')
# plt.plot(sim.trange(), sim.data[probe_arg], label='x')
# plt.legend(loc='best')
# plt.show()

# plt.plot(reconstruct(orthonorm_basis_fns, signal_coeff))
# plt.show()

# plt.plot(S)
# plt.show()

# plt.plot(domain_points, function_vals)
# plt.show()

########### Storing and loading data ###########
# encoders = np.load('coeff.npz')['encoders']
# signal_coeff = np.load('coeff.npz')['signal_coeff']
# orthonorm_basis_fns = np.load('coeff.npz')['ortho']
# np.savez('coeff', encoders=encoders, signal_coeff=signal_coeff, ortho=orthonorm_basis_fns)

# def func_apply(f_by_x):
#     #should use the reconstructed function here right?
#     coeff = f_by_x[0:n_basis]
#     f = reconstruct(orthonorm_basis_fns, coeff)
#     x = f_by_x[-domain_dim:]
#     index = min(int(x / dx) + len(domain_points) / 2, len(domain_points) - 1)
#     return f[index]
