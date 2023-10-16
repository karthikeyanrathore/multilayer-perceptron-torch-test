#!/usr/bin/env python3

def mean_squared_error_loss(y_actual, y_prediction):
    error = 0
    for ya, yp in zip(y_actual, y_prediction):
        diff = (yp - ya)
        error += diff ** 2
    return  error / len(y_actual)

def calculate_loss(mlp, X, y, iter=10):
    for i in range(iter):
        # forward pass
        y_prediction =  [mlp(x) for x in X]
        loss = mean_squared_error_loss(y, y_prediction)
        if i % 5 == 0:
            print(f"[{i}] loss: {loss.data}")

        # After each backward pass 
        # update gradient to 0.0 for forward pass
        for p in mlp.parameters():
            p.gradient = 0.0
        # backward pass
        loss.backward()

        # update
        for p in mlp.parameters():
            p.data =  p.data -  0.01 * p.gradient

if __name__ == "__main__":
    try:
        from mlp.mlp_model import MLP
    except ModuleNotFoundError:
        from mlp_model import MLP

    mlp = MLP(3, [4, 4, 1]) # model

    X = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ] # 4 forward passes
    y = [1.0, -1.0, -1.0, 1.0] # desired targets
    
    calculate_loss(mlp, X, y, iter=100)