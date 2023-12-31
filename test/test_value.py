
from mlp.mlp_value import Value
from mlp.mlp_model import Neuron, MLP
from mlp.mlp_test import mean_squared_error_loss
import numpy as np
import torch
import os
import random


def test_value_operations():
    # supports +, -, *, /, **
    # float/int <op> Value works and reverse too.
    a = Value(-1.0)
    b = Value(3.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).tanh()
    d += 3 * d + (b - a).tanh()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f

    assert round(g.data) == 39440 # Value(data=39439.889108858835, gradient=0.0)

    g.backward()

    assert g.gradient == 1.0 # makes sense :P
    # shrinking it to round about
    assert round(c.gradient) == -281
    assert round(f.gradient) ==  0 # 
    assert round(a.gradient) == 9347
    assert round(b.gradient) == 86583


def test_value_against_pytorch():

    # tanh
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.tanh() + z * x
    h = (z * z).tanh()
    y = h + q + q * x
    y.backward()
    value_x, value_y = x, y

    x = torch.Tensor([-4.0]).double() # 64-bit
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.tanh() + z * x
    h = (z * z).tanh()
    y = h + q + q * x
    y.backward()

    torch_x, torch_y = x, y
    
    # forward pass
    assert value_y.data == torch_y.data.item()

    # backward pass
    assert value_x.gradient == torch_x.grad.item()


def test_forward_and_backward_pass():

    x = Value(-2.0)
    y = Value(-1.0)

    w = -1
    b = 0
    f = w * x + b
    loss = ((f - y)**2)
    # print(f"loss: {loss}")
    f.backward()

    xtorch = torch.Tensor([-2.0]).double() # 64-bit
    xtorch.requires_grad = True
    w = -1
    b = 0
    ftorch = w * xtorch + b
    ftorch.backward()

    # forward
    assert f.data == ftorch.data.item()
    # backward
    assert x.gradient == xtorch.grad.item()

def test_multiple_value_tensors():

    x = [Value(1.0), Value(10.0), Value(-9.0), Value(-2.0)]
    w = [(random.uniform(-1, 1)) for _ in range(4)]
    # print(w)
    b = 0
    # hack found :P
    f = sum((Value(wi) * xi for xi, wi in zip(x, w)), b)

    f.backward()

    # xtorch = [
    #     torch.Tensor([1.0]).double(),
    #     torch.Tensor([10.0]).double(),
    #     torch.Tensor([-9.0]).double(),
    #     torch.Tensor([-4.0]).double(),
    # ] # 64-bit

    x_torch = torch.Tensor([[1.0], [10.0], [-9.0], [-2]]).double()
    # for i  in range(4):
    #     xtorch[i].requires_grad = True
    x_torch.requires_grad = True

    ftorch = sum((wi * xi for xi, wi in zip(x_torch, w)), b)
    ftorch.backward()

    # print(x)

    # forward good.
    assert f.data == ftorch.data.item()

    # backward should pass
    assert x_torch.grad[0][0].item() == x[0].gradient
    assert x_torch.grad[1][0].item() == x[1].gradient
    assert x_torch.grad[2][0].item() == x[2].gradient
    assert x_torch.grad[3][0].item() == x[3].gradient



def test_neuron_class_for_regression():
    x = [Value(1.0), Value(10.0), Value(-9.0), Value(-2.0)]

    neuron = Neuron(4)
    res = neuron(x)
    # AHH!, tanh is bit funky in Value that's fails when used against torch.tanh().
    # very minor diff though.
    final = res.tanh()
    final.backward()


    # pytorch
    x_torch = torch.Tensor([[1.0], [10.0], [-9.0], [-2.0]]).double()
    x_torch.requires_grad = True

    res = neuron(x_torch)
    assert isinstance(res, torch.Tensor) == True
    torch_tanh = res.tanh()
    torch_tanh.backward()

    # print(x_torch.grad)
    # print(neuron.w[0].grad)

    # # forward good.
    assert abs(final.data - torch_tanh.data.item()) < 1e-5

    # # backward should pass
    # difference should be very minor < 1e-5
    assert abs(x_torch.grad[0][0].item() - x[0].gradient) < 1e-5
    assert abs(x_torch.grad[1][0].item() - x[1].gradient) < 1e-5
    assert abs(x_torch.grad[2][0].item() - x[2].gradient) < 1e-5
    assert abs(x_torch.grad[3][0].item() - x[3].gradient) < 1e-5

    

def test_regression_including_mseloss():

    # 3 features
    # 2 data pts. 

    # Value implementation
    os.environ["PYTORCH"] = "0"
    x = [
        [1.0, 2.0, 3.0, 4.0],
        [9.0, -1.0, 5.0, 6.0],
    ]
    y = [-1.0, 1.0]

    # 3 features NN
    nn = Neuron(len(x[0])) # working with w values between (-1,1)

    assert len(nn.parameters()) == 5
    # nn = MLP(3, [4, 4, 1])
    epochs = 100
    for epoch in range(epochs):
        predic = [nn(pt) for pt in x]
        # MES loss
        mse = sum((pi - yi)**2 for pi, yi in zip(predic, y)) / len(y)    
        for p in nn.parameters():
            p.gradient = 0.0
        mse.backward()
        # print(f"value mse: {mse}")
        # print(f"mse: {mse}")
        # updating weights
        for p in nn.parameters():
            # print(p.data)
            p.data = p.data - 0.001 * p.gradient
        if epoch == epochs -1:
            val_res = mse


    # pytorch implementation 
    os.environ["PYTORCH"] = "1"
    nn = Neuron(4)
    x_torch = torch.Tensor([[1.0, 2.0, 3.0, 4.0], [9.0, -1.0, 5.0, 6.0]]).double()
    x_torch.requires_grad = True
    y = torch.tensor([-1.0, 1.0])
    w_for_torch = nn.w
    b = nn.b
    epochs = 100
    for epoch in range(epochs):
        fw = x_torch @ w_for_torch.T + b 
        error = fw - y
        mse = torch.sum(error**2 ) / error.numel()
        mse.backward()
        # print(f"torch mse: {mse}")
        # print(w_for_torch.grad)
        with torch.no_grad():
            w_for_torch -= 0.001 * w_for_torch.grad 
            b -= 0.001 * b.grad
        w_for_torch.grad.zero_()
        b.grad.zero_()
        if epoch == epochs - 1:
            tr_res = mse
    
    print(f"tr_res.data.item(): {tr_res.data.item()}")
    print(f"val.data: {val_res.data}")
    assert abs(tr_res.data.item() - val_res.data) <  1e-4

