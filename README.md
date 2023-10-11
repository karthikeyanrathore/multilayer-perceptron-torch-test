## multilayer-perceptron-torch-test

Testing custom MLP model written from scratch in python in < 200 loc against pytorch library.

## build and run mlp_test.py
```bash
docker-compose build; \
docker-compose run custom-mlp-model
```
## testing in local
can't docker-compose as it requires torch package.
```bash
pwd test; coverage run -m pytest
```

## TODO
- ~test cases~
- add more test case related to MLP/Layers/Neuron model
- add case for Value.exp.
- add more activation functions to Value class.
