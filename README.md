## multilayer-perceptron-torch-test

Testing custom MLP model written from scratch in python in < 200 loc against pytorch library.

## forward / backward pass comparison against pytorch

In mlp/mlp_value.py

```python3
>>> from mlp.mlp_value import Value
>>> x = Value(-4.0)
>>> z = 2 * x + 2 + x
>>> q = z.tanh() + z * x
>>> h = (z * z).tanh()
>>> y = h + q + q * x
>>> y.backward()
>>> x.gradient
104.99999992992078
>>> y.data
-116.00000001236691
```

In Pytorch

```python3
>>> import torch
>>> x = torch.Tensor([-4.0]).double() # 64-bit
>>> x.requires_grad = True
>>> z = 2 * x + 2 + x
>>> q = z.tanh() + z * x
>>> h = (z * z).tanh()
>>> y = h + q + q * x
>>> y.backward()
>>> x.grad.item()
104.99999992992078
>>> y.data
tensor([-116.0000], dtype=torch.float64)
```

## build and run mlp_test.py
```bash
docker-compose build; \
docker-compose run custom-mlp-model
```
## testing in local
can't docker-compose as it requires torch package.
```bash
export TORCH_TESTING=1;pwd test; coverage run -m pytest
```

## TODO
- ~test cases~
- add more test case related to MLP/Layers/Neuron model
- add case for Value.exp.
- add more activation functions to Value class.
- explore pytorch doc: https://pytorch.org/docs/stable/torch.html
- see if i can change tanh backward pass XD: https://discuss.pytorch.org/t/change-the-tanh-backward-code/148609/3
- testing tinygrad ofc: https://github.com/tinygrad/tinygrad/tree/master
- add support for matrices (build nn on top of numpy lib)
