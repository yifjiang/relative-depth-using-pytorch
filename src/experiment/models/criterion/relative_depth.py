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
		# return -F.logsigmoid(-ground_truth*(z_A-z_B))*mask + (1-mask)*(z_A-z_B)*(z_A-z_B)
		return mask*torch.log(1+torch.exp(-ground_truth*(z_A-z_B)))+(1-mask)*(z_A-z_B)*(z_A-z_B)

	def __init__(self):
		super(relative_depth_crit, self).__init__()

	# def forward(self,z_A,z_B,ground_truth):
	# 	return self.__loss_func_arr(z_A, z_B, ground_truth)

	def forward(self, input, target):
		output = Variable(torch.Tensor([0])).cuda()
		n_point_total = 0
		cpu_input = input
		for batch_idx in range(0,cpu_input.size()[0]):
			n_point_total+=target[batch_idx]['n_point']

			x_A_arr = target[batch_idx]['x_A']
			y_A_arr = target[batch_idx]['y_A']
			x_B_arr = target[batch_idx]['x_B']
			y_B_arr = target[batch_idx]['y_B']

			batch_input = cpu_input[batch_idx, 0]
			z_A_arr = batch_input.index_select(1, x_A_arr.long()).gather(0, y_A_arr.view(1,-1).long())
			z_B_arr = batch_input.index_select(1, x_B_arr.long()).gather(0, y_B_arr.view(1,-1).long())

			ground_truth_arr = target[batch_idx]['ordianl_relation']
			output += torch.sum(self.__loss_func_arr(z_A_arr, z_B_arr, ground_truth_arr))

		return output/n_point_total#n_point_total should be of type int or IntTensor


		#TODO
if __name__ == '__main__':
	crit = relative_depth_crit().cuda()
	print(crit)
	# z_A = Variable(torch.zeros(5), requires_grad = True)
	# z_B = Variable(torch.ones(5), requires_grad = True)
	# ground_truth = Variable(torch.ones(5))
	# print(crit(z_A, z_B, ground_truth))
	# mean = crit(z_A, z_B, ground_truth).mean()
	# mean.backward()
	# print(z_A.grad)
	# print(z_B.grad)
	x = Variable(torch.rand(1,1,320,320).cuda(), requires_grad = True)
	target = {}
	target[0] = {}
	target[0]['x_A'] = Variable(torch.Tensor([0,1,2,3,4,5])).cuda()
	target[0]['y_A'] = Variable(torch.Tensor([0,1,2,3,4,5])).cuda()
	target[0]['x_B'] = Variable(torch.Tensor([5,4,3,2,1,0])).cuda()
	target[0]['y_B'] = Variable(torch.Tensor([5,4,3,2,1,0])).cuda()
	target[0]['ordianl_relation'] = Variable(torch.Tensor([0,1,-1,0,-1,1])).cuda()
	target[0]['n_point'] = 1
	loss = crit(x,target)
	print(loss)
	loss.backward()
	print(x.grad)
