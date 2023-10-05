import numpy as np

from mlp_value import Value

class Neuron:
    def __init__(self, features_dim):
        import random
        self.w = [Value(random.uniform(-1, 1)) for _ in range(features_dim)]
        self.b = Value(random.uniform(-10, 10))
    
    def __call__(self, X):
        # dot_result = np.dot(X, self.w) + self.b
        result = sum((wi*xi for wi,xi in zip(self.w, X)), self.b)
        return result.tanh()


class Layer:
    def __init__(self,neurons_or_units_count, features_dim):
        self.neurons_init = [Neuron(features_dim) for  _ in range(neurons_or_units_count)]

    def __call__(self, X):
        result = [nn(X) for nn in self.neurons_init]
        return result
    

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
        layer_outputs = [layer(X) for layer in self.layer_init]
        return layer_outputs[-1]