import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

class relative_depth_crit(nn.Module):
	"""docstring for relative_depth_crit"""
	def __loss_func(self, z_A, z_B, ground_truth):
		if ground_truth == 0:
			return (z_A-z_B)*(z_A-z_B)
		else:
			return -F.logsigmoid(ground_truth*(z_A-z_B))

	def __loss_func_arr(self, z_A, z_B, ground_truth):
		mask = torch.abs(ground_truth)
		return -F.logsigmoid(ground_truth*(z_A-z_B))*mask + (1-mask)*(z_A-z_B)*(z_A-z_B)

	def __init__(self):
		super(relative_depth_crit, self).__init__()

	def forward(self,z_A,z_B,ground_truth):
		return self.__loss_func_arr(z_A, z_B, ground_truth)

		#TODO
if __name__ == '__main__':
	crit = relative_depth_crit()
	print(crit)
	z_A = Variable(torch.zeros(5), requires_grad = True)
	z_B = Variable(torch.ones(5), requires_grad = True)
	ground_truth = Variable(torch.ones(5))
	print(crit(z_A, z_B, ground_truth))
	mean = crit(z_A, z_B, ground_truth).mean()
	mean.backward()
	print(z_A.grad)
	print(z_B.grad)