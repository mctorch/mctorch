# McTorch Examples

Examples for using McTorch will be added here with respective source in example files.
- [pca.py](pca.py) and [pca_gpu.py](pca_gpu.py): Example showing use of manifold based parameter directly in the optmizer of pytorch using either SGD or Adagrad.
- [1 - Multilayer Perceptron.ipynb](1%20-%20Multilayer%20Perceptron.ipynb): Notebook file showing a sample multi-layer perceptron applied on MNIST dataset. [source](https://github.com/bentrevett/pytorch-image-classification/blob/master/1%20-%20Multilayer%20Perceptron.ipynb) (with minor modifications)
- [1 - Multilayer Perceptron-Manifold Stiefel.ipynb](1%20-%20Multilayer%20Perceptron-Manifold%20Stiefel.ipynb): Same as file above replacing all normal linear layers with manifold constraint linear layers.
- [2 - LeNet.ipynb](2%20-%20LeNet.ipynb): Notebook file showing a sample convolutional network applied on MNIST dataset. [source](https://github.com/bentrevett/pytorch-image-classification/blob/master/2%20-%20LeNet.ipynb)(with minor modifications)
- [2 - LeNet-Manifold Stiefel.ipynb](2%20-%20LeNet-Manifold%20Stiefel.ipynb): Same as file above replacing all convolutional and linear layers to use manifold constraint layers.