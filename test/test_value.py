
from mlp.mlp_value import Value
import torch

def test_value_operations():
    # supports +, -, *, /
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

    



