import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(0)
cuda = torch.device('cuda') 

# Random data with high variance in first two dimension
X_ = torch.diag(torch.FloatTensor([3,2,1])).matmul(torch.randn(3,200))
X_ -= X_.mean(axis=0)
X = X_.to(device=cuda)
# 1. Initialize Parameter
manifold_param = nn.Parameter(data=torch.randn((3,2), device=cuda), manifold=nn.Stiefel(3,2))

# 2. Define Cost - squared reconstruction error
def cost(X, w):
    wTX = torch.matmul(w.transpose(1,0), X)
    wwTX = torch.matmul(w, wTX)
    return torch.norm((X - wwTX)**2)

# 3. Optimize
# optimizer = torch.optim.Adagrad(params = [manifold_param], lr=1)
optimizer = torch.optim.SGD(params = [manifold_param], lr=1e-2)

cost_step = None
for epoch in range(1000):
    cost_step = cost(X, manifold_param)
    # print(cost_step)
    cost_step.backward()
    optimizer.step()
    optimizer.zero_grad()
print(cost_step)

np_X = X_.detach().numpy()
np_w = manifold_param.detach().cpu().numpy()

# 4. Test Results

estimated_projector = np_w @ np_w.T

eigenvalues, eigenvectors = np.linalg.eig(np_X @ np_X.T)
indices = np.argsort(eigenvalues)[::-1][:2]
span_matrix = eigenvectors[:, indices]
projector = span_matrix @ span_matrix.T

print("Frobenius norm error between estimated and closed-form projection "
          "matrix:", np.linalg.norm(projector - estimated_projector))