#!/usr/bin/env python3

from mlp_model import MLP

X = [2.0, 3.0, 4.0] # input features

nn = MLP(3, [4, 4, 1]) # (input feature dimension, list of neurons)

# forward pass
print(f"[1] Forward Pass result:{nn(X)}")