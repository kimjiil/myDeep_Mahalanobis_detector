import torch
from torch.distributions import multivariate_normal
# dist = multivariate_normal.MultivariateNormal(loc=torch.zeros(5), covariance_matrix=torch.eye(5))
# probability = torch.exp(dist.log_prob(torch.randn(5)))
# print()
#
# diagonal = torch.rand(5) + 1
# temp = torch.diag(diagonal)
# print()
a = [i for i in range(1,61)]
t = torch.tensor(a).view(3,4,5)

index = torch.Tensor([[3,3,3,3,3], [2,2,2,2,2], [0,0,0,0,0]]).type(torch.int64)
print()