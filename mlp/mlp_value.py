class Value:
    def __init__(self, data, children=(), _operation="", variable=""):
        self.data = data
        self.gradient = 0.0
        self.children = set(children)
        self.operation = _operation
        self.variable = variable
        self.backward_propagation = lambda: None
        self.debug = None

    def __repr__(self):
        return f"Value(data={self.data}, gradient={self.gradient})"

    def __add__(self, value_to_add):
        if not isinstance(value_to_add, Value):
            value_to_add = Value(value_to_add)
        result = Value(self.data + value_to_add.data, (self, value_to_add), "+")
        # result is b
        def _backward_propagation():
            self.gradient += 1.0 * result.gradient # a
            value_to_add.gradient += 1.0 * result.gradient # a
        result.backward_propagation = _backward_propagation
        return result

    def __mul__(self, value_to_mult):
        if not isinstance(value_to_mult, Value):
            value_to_mult = Value(value_to_mult)
        result = Value(self.data * value_to_mult.data, (self, value_to_mult), "*")
        def _backward_propagation():
            self.gradient += value_to_mult.data * result.gradient
            value_to_mult.gradient += self.data * result.gradient
        result.backward_propagation = _backward_propagation
        return result

    def __rmul__(self, other):
        # a = Value(1.0)
        # a = 2 * a fails if we don't have __rmul__ function to swap it.
        # 2.__mul__(Value(1.0)) ???
        return self * other # Value(1.0).__mul__(2)

    def tanh(self):
        import math
        tanh_function =(math.exp(2*self.data) - 1) / (math.exp(2*self.data) + 1)
        result = Value(tanh_function, (self, ), "tanh")

        # print(self.data)
        def _backward_propagation():
            # self.debug= self.data
            self.gradient +=  (1 - result.data**2) * result.gradient
        result.backward_propagation = _backward_propagation
        return result


    def __pow__(self, other):
        # other should not be Value object.
        assert isinstance(other, (int, float)), "value can be either float/int."
        result = Value(self.data**other, (self, ), f"pow{other}")
        def _backward_propagation():
            self.gradient += other * ((self.data)**(other - 1)) * result.gradient
        result.backward_propagation = _backward_propagation
        return result

    def __truediv__(self, other):
        return self * other**-1
    
    def __rtruediv__(self, other):
        # makes (int/float)/Value possible
        return other * self **-1

    def exp(self):
        import math
        result = Value(math.exp(self.data), (self, ), "exp")
        def _backward_propagation():
            self.gradient += math.exp(self.data) * result.gradient
        result.backward_propagation = _backward_propagation
        return result

    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __radd__(self, other):
        # makes int + Value possible
        return self + other

    def backward(self):
        self.gradient = 1.0
        for v in reversed(self.deepwalk()):
            v.backward_propagation()

    def deepwalk(self):
        # toposort
        def _deepwalk(vertex, visited, nodes):
            if vertex not in visited:
                visited.add(vertex)
                for previous_vertex in vertex.children:
                    _deepwalk(previous_vertex, visited, nodes)
                nodes.append(vertex)
            return nodes
        return _deepwalk(self, set(), [])
