import numpy as np
import pytest

import nengo
from nengo.utils.compat import range
from nengo.utils.numpy import rms
from nengo.utils.testing import Plotter


def test_sine_waves(Simulator, nl):
    radius = 2
    dim = 5
    product = nengo.networks.Product(
        200, dim, radius, neuron_type=nl(), seed=63)

    func_A = lambda t: radius*np.sin(np.arange(1, dim+1)*2*np.pi*t)
    func_B = lambda t: radius*np.sin(np.arange(dim, 0, -1)*2*np.pi*t)
    pstc = 0.003
    with product:
        input_A = nengo.Node(func_A)
        input_B = nengo.Node(func_B)
        nengo.Connection(input_A, product.A)
        nengo.Connection(input_B, product.B)
        p = nengo.Probe(product.output, synapse=pstc)

    sim = Simulator(product)
    sim.run(1.0)

    t = sim.trange()
    AB = np.asarray(list(map(func_A, t))) * np.asarray(list(map(func_B, t)))
    delay = 0.011
    offset = np.where(t > delay)[0]

    with Plotter(Simulator) as plt:
        for i in range(dim):
            plt.subplot(dim+1, 1, i+1)
            plt.plot(t + delay, AB[:, i], label="$A \cdot B$")
            plt.plot(t, sim.data[p][:, i], label="Output")
            plt.legend()
        plt.savefig('test_product.test_sine_waves.pdf')
        plt.close()

    assert rms(AB[:len(offset), :] - sim.data[p][offset, :]) < 0.3


@pytest.mark.benchmark
def test_benchmark(Simulator):
    nl = nengo.LIF
    n_neurons = 50
    N = 50
    # N = 10
    rng = np.random

    radius = np.sqrt(2)

    n_trials = 50
    # n_trials = 10
    t_trial = 0.1
    t_check = 0.03
    tau_out = 0.01
    tau_probe = 0.01
    values = rng.uniform(low=-1, high=1, size=(n_trials, N, 2))
    values = np.sign(values) * np.abs(values)**(1./2)  # make product dist. more uniform

    def present(t):
        return values[int(t / t_trial)].flatten()

    encoders = np.tile([[1, 1], [-1, 1], [1, -1], [-1, -1]],
                       (np.ceil(n_neurons / 2.), 1))[:2 * n_neurons]

    with nengo.Network() as model1:
        u = nengo.Node(output=present)
        a = nengo.networks.EnsembleArray(2 * n_neurons, N, 2, radius=radius, encoders=encoders)

        nengo.Connection(u, a.input, synapse=None)
        b = a.add_output('product', lambda x: x[0] * x[1], synapse=tau_out)

        up = nengo.Probe(u, synapse=tau_probe)
        bp = nengo.Probe(b, synapse=tau_probe)

    sim = nengo.Simulator(model1)
    sim.run(n_trials * t_trial)

    t = sim.trange()
    tmask = (t % t_trial) > (t_trial - t_check)

    x1, z1 = sim.data[up], sim.data[bp]
    y1 = x1[:, 0::2] * x1[:, 1::2]
    rmse1 = rms(z1[tmask] - y1[tmask])

    # --- now using quarter-square
    with nengo.Network() as model2:
        u = nengo.Node(output=present)
        a = nengo.networks.EnsembleArray(n_neurons, N, 1)  # 0.5 * (x + y)
        b = nengo.networks.EnsembleArray(n_neurons, N, 1)  # 0.5 * (x - y)
        d = nengo.Node(size_in=N)

        nengo.Connection(u, a.input, function=lambda x: 0.5 * (x[0::2] + x[1::2]), synapse=None)
        nengo.Connection(u, b.input, function=lambda x: 0.5 * (x[0::2] - x[1::2]), synapse=None)
        a.add_output('square', lambda x: x**2)
        b.add_output('square', lambda x: -x**2)

        nengo.Connection(a.square, d, synapse=tau_out)
        nengo.Connection(b.square, d, synapse=tau_out)

        up = nengo.Probe(u, synapse=tau_probe)
        dp = nengo.Probe(d, synapse=tau_probe)

    sim = nengo.Simulator(model2)
    sim.run(n_trials * t_trial)

    t = sim.trange()
    tmask = (t % t_trial) > (t_trial - t_check)

    x2, z2 = sim.data[up], sim.data[dp]
    y2 = x2[:, 0::2] * x2[:, 1::2]
    rmse2 = rms(z2[tmask] - y2[tmask])
    print rmse1, rmse2

    with Plotter(Simulator) as plt:
        r, c = 2, 2
        x = [x1, x2]
        y = [y1, y2]
        z = [z1, z2]

        for i, [xi, yi, zi] in enumerate(zip(x, y, z)):
            plt.subplot(r, c, i+1)
            plt.plot(t, xi)
            plt.subplot(r, c, c + i+1)
            plt.plot(t, yi, 'k')
            plt.plot(t, zi)

        # plt.show()
        plt.savefig('test_product.test_benchmark.pdf')
        plt.close()


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
