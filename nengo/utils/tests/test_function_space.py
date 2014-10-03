import numpy as np
import matplotlib.pyplot as plt

from nengo.utils.function_space import *
from nengo.utils.testing import Plotter

def test_function_space(Simulator, nl):

    #parameters
    n_neurons = 1500 # number of neurons
    domain_dim = 1
    range_dim = 1
    radius = 1
    d = 0.001 # discretization

    #setup functions and coefficients
    functions = gaussian_functions_1D(n_neurons, 0.5, radius)
    domain_points = uniform_cube(domain_dim, radius, d)
    FS = Function_Space(functions, domain_points, d)
    input_func = lambda x: x ** 2
    signal_coeffs = FS.signal_coeffs(domain_points ** 2)
    encoders = FS.encoder_coeffs()

    #pick evaluation points to be polynomial functions
    n_eval_points = 400
    eval_points = FS.eval_points(n_eval_points)

    f_radius = np.linalg.norm(signal_coeffs) # radius to use for function rep.

    #figure out what the value of the function should be at x
    #relies on x being in a uniformly discretized domain, but can generalize
    #this to multiple dimensions
    def func_apply(f_by_x):
        coeff = f_by_x[0:FS.n_basis]
        f = FS.reconstruct(coeff)
        x = f_by_x[-domain_dim:]
        index = min(int(x / FS.dx) + len(domain_points) / 2, len(domain_points) - 1)
        return f[index]

    n_samples = 300
    eval_points_fx = [] #eval_points for f_by_x ensemble
    for point in eval_points:
        # n_samples argument samples for each function
        for _ in range(n_samples):
            eval_points_fx.append(np.append(point, np.random.uniform(-1, 1)))

    eval_points_fx = np.array(eval_points_fx)

    with nengo.Network() as model:
        #represents the function
        f = nengo.Ensemble(n_neurons=n_neurons, dimensions=FS.n_basis,
                           encoders=encoders, radius=f_radius,
                           eval_points=eval_points, label='f')
        #represents the argument
        x = nengo.Ensemble(n_neurons=100, dimensions=domain_dim,
                           radius=radius, label='x')
        #represents the function and the argument
        f_by_x = nengo.Ensemble(n_neurons=n_neurons,
                                dimensions=FS.n_basis + domain_dim,
                                radius=f_radius + radius, label='f_by_x',
                                eval_points=eval_points_fx)
        #represents the function applied to the argument
        fx = nengo.Ensemble(n_neurons=200, dimensions=range_dim,
                            radius=radius, label='fx')

        in_x = nengo.Node(output=lambda t: np.sin(t))
        signal = nengo.Node(output=signal_coeffs)

        nengo.Connection(signal, f)
        nengo.Connection(in_x, x)
        nengo.Connection(x, f_by_x[-1])
        nengo.Connection(f, f_by_x[:FS.n_basis])
        nengo.Connection(f_by_x, fx, function=func_apply)

        probe_model = nengo.Probe(fx, synapse=0.1)
        probe_f_by_x = nengo.Probe(f_by_x, synapse=0.1)

    sim = Simulator(model)
    sim.run(6)

    true_val = input_func(sim.data[probe_f_by_x][:, -1])
    reconstruction = FS.reconstruct(sim.data[probe_f_by_x][400, :FS.n_basis])
    model_val = sim.data[probe_model]

    true_f = input_func(domain_points)

    with Plotter(Simulator) as plt:
        plt.figure()
        plt.plot(sim.trange(), true_val, label='true')
        plt.plot(sim.trange(), model_val, label='model')
        plt.plot(sim.trange(), sim.data[probe_f_by_x][:, -1], label='x')
        plt.legend(loc='best')
        plt.savefig('utils.test_function_space.test_function'
                    '_values_%s.pdf' % str(nl))
        plt.close()

        plt.figure()
        plt.plot(domain_points, reconstruction, label='model_f')
        plt.plot(domain_points, input_func(domain_points), label='true_f')
        plt.legend(loc='best')
        plt.savefig('utils.test_function_space.test_function'
                        '_representation_%s.pdf' % str(nl))
        plt.close()

    assert np.allclose(true_f, reconstruction, atol=0.2)
    assert np.allclose(true_val, model_val, atol=0.5)
