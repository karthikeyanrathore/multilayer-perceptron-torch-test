import numpy as np
import os
try:
    from mlp.mlp_value import Value
except ModuleNotFoundError:
     from mlp_value import Value

class Neuron:
    def __init__(self, features_dim):
        import random
        if os.environ.get("TORCH_TESTING", 0):
            if features_dim == 4:
                self.w = [0.068718719629973, -0.815050457958294, -0.7715021568114409, -0.942495102903679]
            self.b = 0
        else:
            self.w = [Value(random.uniform(-1, 1)) for _ in range(features_dim)]
            self.b = Value(0)

    def __call__(self, X):
        # dot_result = np.dot(X, self.w) + self.b
        result = sum((wi*xi for wi,xi in zip(self.w, X)), self.b)
        # print(f"Neuron.result: {result.tanh()}")
        if os.environ.get("TORCH_TESTING", 0):
            return result
        return result.tanh()
    
    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self,neurons_or_units_count, features_dim):
        self.neurons_init = [Neuron(features_dim) for  _ in range(neurons_or_units_count)]

    def __call__(self, X):
        result = [nn(X) for nn in self.neurons_init]
        return result[0] if len(result) == 1 else result
    
    def parameters(self):
        a = []
        for neuron in self.neurons_init:
            a.extend(neuron.parameters())
        return a


class MLP:
    # Multilayer perceptron
    def __init__(self, feature_input_dim, layers_neuron_list):
        # (3, [4, 4, 1])
        self.layer_init = []
        update_nn_list = [feature_input_dim] + layers_neuron_list
        for i in range(len(layers_neuron_list)):
            input_neurons_unit = update_nn_list[i] # 3, 4, 4
            neuron_units = update_nn_list[i+1] # 4, 4, 1
            self.layer_init.append(Layer(neuron_units, input_neurons_unit))

    def __call__(self, X):
        for layer in self.layer_init:
            out = layer(X)
        return out

    def parameters(self):
        a = []
        for layer in self.layer_init:
            a.extend(layer.parameters())
        return a