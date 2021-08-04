import torch
import mctorch.nn as mnn
import mctorch.optim as moptim
from mctorch.nn.manifolds import doublystochastic as ds

k, n, m = 3, 100, 1000

A = torch.empty((k, n, m))
p = torch.empty((k, n))
q = torch.empty((k, m))
for i in range(k):
    p0 = torch.rand(n)
    q0 = torch.rand(m)
    A0 = torch.rand(n, m)[None, :]
    p[i] = p0 / torch.sum(p0)
    q[i] = q0 / torch.sum(q0)
    A[i] = ds.SKnopp(A0, p[i], q[i], n+m)

# 1. Initialize Parameter
manifold_param = mnn.Parameter(manifold=mnn.DoublyStochastic(n, m, k=k, p=p, q=q))

# 2. Define Cost
def cost(x):
    return 0.5 * (torch.linalg.norm(x - A)**2)

# 3. Optimize
optimizer = moptim.ConjugateGradient(params = [manifold_param], lr=1e-2)

for epoch in range(30):
    cost_step = cost(manifold_param)
    print(cost_step)
    cost_step.backward()
    optimizer.step()
    optimizer.zero_grad()
