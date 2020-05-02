# McTorch Examples

Examples for using McTorch will be added here with respective source in example files.
- [pca.py](pca.py) and [pca_gpu.py](pca_gpu.py): Example showing use of manifold based parameter directly in the optmizer of pytorch using either SGD or Adagrad.
- [1 - Multilayer Perceptron.ipynb](1%20-%20Multilayer%20Perceptron.ipynb): Notebook file showing a sample multi-layer perceptron applied on MNIST dataset. [source](https://github.com/bentrevett/pytorch-image-classification/blob/master/1%20-%20Multilayer%20Perceptron.ipynb) (with minor modifications)
- [1 - Multilayer Perceptron-Manifold Stiefel.ipynb](1%20-%20Multilayer%20Perceptron-Manifold%20Stiefel.ipynb): Same as file above replacing all normal linear layers with manifold constraint linear layers.
- [2 - LeNet.ipynb](2%20-%20LeNet.ipynb): Notebook file showing a sample convolutional network applied on MNIST dataset. [source](https://github.com/bentrevett/pytorch-image-classification/blob/master/2%20-%20LeNet.ipynb)(with minor modifications)
- [2 - LeNet-Manifold Stiefel.ipynb](2%20-%20LeNet-Manifold%20Stiefel.ipynb): Same as file above replacing all convolutional and linear layers to use manifold constraint layers.

## How to Run
After installation of McTorch with GPU support is completed - 
``` bash
pip install torchvision==0.6.0 --no-deps # no deps is required since there is a hard dependency on torch 1.5.0 and it tries to install torch instead of using mctorch
pip install Pillow
pip install sklearn
pip install matplotlib
pip install jupyter
jupyter notebook # this will open jupyter notebook in the browser, where you can run and view the exisiting notebook files
```
## References
Some more great examples of using pytorch for image classification here -
- https://github.com/bentrevett/pytorch-image-classification