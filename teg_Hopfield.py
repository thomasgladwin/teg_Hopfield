import numpy as np

class Hopfield:
    def __init__(self, nNeurons, bias_default=0):
        self.nNeurons = nNeurons
        self.bias = np.ones((self.nNeurons, 1)) * bias_default
        self.neuron_activations = np.zeros((self.nNeurons, 1))
        self.W = self.get_random_weights()
        self.iNeuron = 0

    def get_random_weights(self):
        W = -1 + 2 * np.random.randn(self.nNeurons ** 2)
        W = W.reshape(self.nNeurons, self.nNeurons)
        W[range(self.nNeurons), range(self.nNeurons)] = 0
        W = W + W.transpose()
        return W

    def calc_neuron_activation(self, iNeuron):
        input_to_neuron = np.dot(self.neuron_activations.reshape(-1), self.W[:, iNeuron])
        activation_prob = (1 + np.sign(input_to_neuron + self.bias[iNeuron])) * 0.5
        die = np.random.rand()
        if die < activation_prob:
            activation = 1
        else:
            activation = -1
        return activation

    def set_activations(self, neuron_activations):
        self.neuron_activations = neuron_activations

    def set_activations_random(self):
        self.neuron_activations = -1 + 2 * np.floor(2*np.random.rand(self.nNeurons, 1))

    def update(self):
        # iNeuron = np.random.randint(0, self.nNeurons)
        iNeuron = self.iNeuron
        self.iNeuron = (self.iNeuron + 1) % self.nNeurons
        activation = self.calc_neuron_activation(iNeuron)
        self.neuron_activations[iNeuron] = activation

    def run_to_convergence(self, convergence_crit=0.001, verbose=False):
        convergence_steps = self.nNeurons
        converged_period = 0
        old_E = np.nan
        old_d = np.nan
        while converged_period < convergence_steps:
            old_activation = self.neuron_activations.copy()
            self.update()
            new_activation = self.neuron_activations.copy()
            E = -np.dot(old_activation.transpose(), new_activation)
            d = np.nan
            if not np.isnan(old_E):
                d = E - old_E
                if not np.isnan(old_d):
                    if np.abs(d - old_d) < convergence_crit:
                        converged_period = converged_period + 1
                    else:
                        converged_period = 0
                old_d = d
            old_E = E
            if verbose:
                print(E, d, converged_period, self.neuron_activations)

    def train(self, Patterns):
        self.W = 0 * self.W
        Patterns = (Patterns - np.mean(Patterns, axis=0)) / np.sqrt(np.var(Patterns, axis=0))
        nP = Patterns.shape[0]
        C = (Patterns.transpose() @ Patterns) / nP
        C[np.isnan(C)] = 0
        self.W = C - np.identity(self.nNeurons)

#
#
#

class Hopfield_cont:
    def __init__(self, Patterns, bias_default=0, beta=2.0):
        self.X = Patterns # Neurons x Patterns
        self.nNeurons = Patterns.shape[0]
        self.nPatterns = Patterns.shape[1]
        self.bias = np.ones((self.nNeurons, 1)) * bias_default
        self.beta = beta
        self.y = np.zeros((self.nNeurons, 1))

    def set_random_activation(self):
        self.y = np.random.randn(len(self.y), 1)

    def set_activation(self, y):
        self.y = np.array(y).reshape((len(y), 1))

    def softmax(self, v):
        if np.sum(np.exp(v)) > 0:
            c = np.max(v)
            val = np.exp(v - c) / np.sum(np.exp(v - c))
        else:
            val = np.ones(v.shape) / len(v)
        return val

    def update(self):
        arg0 = self.beta * self.X.transpose() @ self.y
        new_y = self.X @ self.softmax(arg0)
        self.y = new_y.copy()

    def lse(self, beta, x):
        val = (1/beta) * np.log(np.sum(np.exp(beta*x)))
        return val

    def calc_energy(self):
        M = np.max(np.sqrt(np.var(self.X, axis=0)))
        term1 = -self.lse(self.beta, self.X.transpose() @ self.y)
        term2 = (1/2) * self.y.transpose() @ self.y
        term3 = (1/self.beta) * np.log(self.nPatterns)
        term4 = (1/2) * (M ** 2)
        e = term1 + term2 + term3 + term4
        e = e[0][0]
        return e

    def run_to_convergence(self, verbose=False):
        # Should converge in a single update.
        e_initial = self.calc_energy()
        self.update()
        e_convergence = self.calc_energy()
        return e_initial, e_convergence
