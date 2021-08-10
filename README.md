# McTorch Lib, a manifold optimization library for deep learning 

McTorch is a Python library that adds manifold optimization functionality to [PyTorch](https://github.com/pytorch/pytorch).  

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
- [Reference](#reference)

## More about McTorch
McTorch builds on top of PyTorch and supports all PyTorch functions in addition to Manifold optimization. This is done to ensure researchers and developers using PyTorch can easily experiment with McTorch functions. McTorch's manifold implementations and optimization methods are derived from the Matlab toolbox [Manopt](http://manopt.org/) and the Python toolbox [Pymanopt](https://pymanopt.github.io/).

### Using McTorch for Optimization

1. **Initialize Parameter** - McTorch manifold parameters are same as PyTorch parameters (`mctorch.nn.Parameter`) and requires just addition of one property to parameter initialization to constrain the parameter values. 
2. **Define Cost** - Cost function can be any PyTorch function using the above parameter mixed with non constrained parameters.
3. **Optimize** - Any optimizer from `mctorch.optim` can be used to optimize the cost function using same functionality as any PyTorch code.

**PCA Example**
```python
import torch
import mctorch.nn as mnn
import mctorch.optim as moptim

# Random data with high variance in first two dimension
X = torch.diag(torch.FloatTensor([3,2,1])).matmul(torch.randn(3,200))

# 1. Initialize Parameter
manifold_param = mnn.Parameter(manifold=mnn.Stiefel(3,2))

# 2. Define Cost - squared reconstruction error
def cost(X, w):
    wTX = torch.matmul(w.transpose(1,0), X)
    wwTX = torch.matmul(w, wTX)
    return torch.sum((X - wwTX)**2)

# 3. Optimize
optimizer = moptim.rAdagrad(params = [manifold_param], lr=1e-2)

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
import mctorch.nn as mnn
import torch.nn.functional as F

# a torch module using constrained linear layers
class ManifoldMLP(nn.Module):
    def __init__(self):
        super(ManifoldMLP, self).__init__()
        self.layer1 = mnn.rLinear(in_features=28*28, out_features=100, weight_manifold=mnn.Stiefel)
        self.layer2 = mnn.rLinear(in_features=100, out_features=100, weight_manifold=mnn.PositiveDefinite)
        self.output = mnn.rLinear(in_features=100, out_features=10, weight_manifold=mnn.Stiefel)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.log_softmax(self.output(x), dim=0)
        return x

# create module object and compute cost by applying module on inputs
mlp_module = ManifoldMLP()
cost = mlp_module(inputs)

```

More examples added - [here](examples)

## Functionality Supported
This would be an ever increasing list of features. McTorch currently supports:

### Manifolds
- Stiefel
- Positive Definite
- Hyperbolic
- Doubly Stochastic

All manifolds support k multiplier as well.

### Optimizers
- rSGD
- rAdagrad
- rASA
- rConjugateGradient

### Layers
- Linear
- Conv1d, Conv2d, Conv3d


## Installation
After installing PyTorch can be installed with `python setup.py install`

### Linux
```bash
source activate myenv
conda install numpy setuptools
# Add LAPACK support for the GPU if needed
conda install -c pytorch magma-cuda90 # or [magma-cuda80 | magma-cuda92 | magma-cuda100 ] depending on your cuda version
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch # or cudatoolkit=10.0 | cudatoolkit=10.1 | .. depending on your cuda version
pip install mctorch-lib
```

## Release and Contribution
McTorch is currently under development and any contributions, suggestions and feature requests are welcome. We'd closely follow PyTorch stable versions to keep the base updated and will have our own versions for other additions.

McTorch is released under the open source [3-clause BSD License](LICENSE).

## Team 
- [Mayank Meghwanshi](https://github.com/mayank127/)
- [Satyadev Ntv](https://github.com/satyadevntv/)
- [Bamdev Mishra](https://github.com/bamdevm)
- [Pratik Jawanpuria](https://pratikjawanpuria.com)
- [Hiroyuki Kasai](https://github.com/hiroyuki-kasai)
- [Anoop Kunchukuttan](https://github.com/anoopkunchukuttan)

## Reference
Please cite [[1](https://arxiv.org/abs/1810.01811)] if you found this code useful.
#### McTorch, a manifold optimization library for deep learning
[1] M. Meghawanshi, P. Jawanpuria, A. Kunchukuttan, H. Kasai, and B. Mishra, [McTorch, a manifold optimization library for deep learning](https://arxiv.org/abs/1810.01811)

```
@techreport{meghwanshi2018mctorch,
  title={McTorch, a manifold optimization library for deep learning},
  author={Meghwanshi, Mayank and Jawanpuria, Pratik and Kunchukuttan, Anoop and Kasai, Hiroyuki and Mishra, Bamdev},
  institution={arXiv preprint arXiv:1810.01811},
  year={2018}
}
```
