# McTorch
McTorch is a Python package that adds manifold optimization functionality to [PyTorch](https://github.com/pytorch/pytorch).  

McTorch:

 - Leverages tensor computation and GPU acceleration from PyTorch.
 - Enables optimization on manifold constrained tensors to address nonlinear optimization problems.
 - Facilitates constrained weight tensors in deep learning layers.

Sections:
- [More about McTorch](#more-about-mctorch)
  - [Using McTorch for Optimization](#using-mctorch-for-optimization)
  - [Using McTorch for Deep Learning](#using-mctorch-for-deep-learning)
- [Functionality Supported](#functionality-supported)
- [Installation](#installation)
- [Release and Contribution](#release-and-contribution)
- [Team](#team)

## More about McTorch
McTorch builds on top of PyTorch and supports all PyTorch functions in addition to Manifold optimization. This is done to ensure researchers and developers using PyTorch can easily experiment with McTorch functions. McTorch's manifold implementations and optimization methods are derived from the Matlab toolbox [Manpot](http://manopt.org/) and the Python toolbox [Pymanopt](https://pymanopt.github.io/).

### Using McTorch for Optimization

1. **Initialize Parameter** - McTorch manifold parameters are same as PyTorch parameters (`torch.nn.Parameter`) and requires just addition of one property to parameter initialization to constrain the parameter values. 
2. **Define Cost** - Cost function can be any PyTorch function using the above parameter mixed with non constrained parameters.
3. **Optimize** - Any optimizer from `torch.optim` can be used to optimize the cost function using same functionality as any PyTorch code.

**PCA Example**
```python
import torch
import torch.nn as nn

# Random data with high variance in first two dimension
X = torch.diag(torch.FloatTensor([3,2,1])).matmul(torch.randn(3,200))

# 1. Initialize Parameter
manifold_param = nn.Parameter(manifold=nn.Stiefel(3,2))

# 2. Define Cost - squared reconstruction error
def cost(X, w):
    wTX = torch.matmul(w.transpose(1,0), X)
    wwTX = torch.matmul(w, wTX)
    return torch.sum((X - wwTX)**2)

# 3. Optimize
optimizer = torch.optim.Adagrad(param = [manifold_param], lr=1e-4)

for epoch in range(30):
    cost_step = cost(X, manifold_param)
    print(cost_step)
    cost_step.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Using McTorch for Deep Learning
**Multi Layer Perceptron Example**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# a torch module using constrained linear layers
class ManifoldMLP(nn.Module):
    def __init__(self):
        super(ManifoldMLP, self).__init__()
        self.layer1 = nn.Linear(in_features=28*28, out_features=100, weight_manifold=nn.Stiefel)
        self.layer2 = nn.Linear(in_features=100, out_features=100, weight_manifold=nn.PositiveDefinite)
        self.output = nn.Linear(in_features=100, out_features=10, weight_manifold=nn.Stiefel)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.log_softmax(self.output(x), dim=0)
        return x

# create module object and compute cost by applying module on inputs
mlp_module = ManifoldMLP()
cost = mlp_module(inputs)

```

## Functionality Supported
This would be an ever increasing list of features. McTorch currently supports:

### Manifolds
- Stiefel
- Positive Definite

All manifolds support k multiplier as well.

### Optimizers
- SGD
- Adagrad
- Conjugate Gradient

### Layers
- Linear
- Conv1d, Conv2d, Conv3d
- Conv1d\_transpose, Conv2d\_transpose, Conv3d\_transpose



## Installation
This is same as PyTorch installation from source.

### Linux
```bash
git clone --recursive https://github.com/mctorch/mctorch
cd mctorch
python setup.py install
```
For other os and optional dependencies go through [Installation](pytorch-README.md#installation)

## Release and Contribution
McTorch is currently under development and any contributions, suggestions and feature requests are welcome. We'd closely follow PyTorch stable versions to keep the base updated and will have our own versions for other additions.

McTorch is released under the open source [3-clause BSD License](LICENSE).

## Team 
- [Mayank Meghwanshi](https://github.com/mayank127/)
- [Bamdev Mishra](https://github.com/bamdevm)
- [Pratik Jawanpuria](https://pratikjawanpuria.com)
- [Hiroyuki Kasai](https://github.com/hiroyuki-kasai)
- [Anoop Kunchukuttan](https://github.com/anoopkunchukuttan)