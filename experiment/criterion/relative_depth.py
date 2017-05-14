import torch
from torch.nn import functional as F

class relative_depth_crit(nn.Module):
	"""docstring for relative_depth_crit"""
	def __loss_func(z_A, z_B, ground_truth):
		if ground_truth == 0:
			return (z_A-z_B)*(z_A-z_B)
		else
			return -F.logsigmoid(ground_truth*(z_A-z_B))

	def __init__(self):
		super(relative_depth_crit, self).__init__()

	def forward(self,z_A,z_B,ground_truth):
		return __loss_func(z_A, z_B, ground_truth)
