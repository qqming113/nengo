import numpy as np

from nengo.builder.builder import Builder
from nengo.builder.signal import Signal
from nengo.builder.operator import Operator
from nengo.synapses import Synapse


class SimSynapse(Operator):
    """Simulate a Synapse object."""
    def __init__(self, input, output, synapse):
        self.input = input
        self.output = output
        self.synapse = synapse

        self.sets = []
        self.incs = []
        self.reads = [input]
        self.updates = [output]

    def make_step(self, signals, dt, rng):
        input = signals[self.input]
        output = signals[self.output]
        step_f = self.synapse.make_step(dt, output)

        def step(input=input):
            step_f(input)

        return step


@Builder.register(Synapse)
def build_synapse(model, synapse, input, output=None):
    if output is None:
        output = Signal(np.zeros(input.shape),
                        name="%s.%s" % (input.name, synapse))

    model.add_op(SimSynapse(input, output, synapse))
    return output
