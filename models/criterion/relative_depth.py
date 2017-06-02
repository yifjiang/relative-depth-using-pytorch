import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

class relative_depth_crit(nn.Module):

	def __loss_func_arr(self, z_A, z_B, ground_truth):
		mask = torch.abs(ground_truth)
		# print(mask)
		z_A = z_A[0]
		a_B = z_B[0]
		# print(z_A.size())
		# return -F.logsigmoid(-ground_truth*(z_A-z_B))*mask + (1-mask)*(z_A-z_B)*(z_A-z_B)
		# print((z_A-z_B)*(z_A-z_B))
		# print ((mask*torch.log(1+torch.exp(-ground_truth*(z_A-z_B)))+(1-mask)*(z_A-z_B)*(z_A-z_B)).size())
		return mask*torch.log(1+torch.exp(-ground_truth*(z_A-z_B)))+(1-mask)*(z_A-z_B)*(z_A-z_B)

	def __init__(self):
		super(relative_depth_crit, self).__init__()

	# def forward(self,z_A,z_B,ground_truth):
	# 	return self.__loss_func_arr(z_A, z_B, ground_truth)

	def forward(self, input, target):
		self.input = input
		self.target = target
		self.output = Variable(torch.Tensor([0])).cuda()
		n_point_total = 0
		cpu_input = input
		for batch_idx in range(0,cpu_input.size()[0]):
			n_point_total+=target[batch_idx]['n_point']

			x_A_arr = target[batch_idx]['x_A']
			# print(torch.max(x_A_arr))
			y_A_arr = target[batch_idx]['y_A']
			# print('y=',torch.max(y_A_arr))
			x_B_arr = target[batch_idx]['x_B']
			y_B_arr = target[batch_idx]['y_B']

			batch_input = cpu_input[batch_idx, 0]
			z_A_arr = batch_input.index_select(1, x_A_arr.long()).gather(0, y_A_arr.view(1,-1).long())
			z_B_arr = batch_input.index_select(1, x_B_arr.long()).gather(0, y_B_arr.view(1,-1).long())

			ground_truth_arr = target[batch_idx]['ordianl_relation']
			self.output += torch.sum(self.__loss_func_arr(z_A_arr, z_B_arr, ground_truth_arr))

		return self.output/n_point_total#n_point_total should be of type int or IntTensor

	def _grad_loss_func(self, z_A, z_B, ground_truth):
		mask = torch.abs(ground_truth)
		z_A_z_B = z_A - z_B
		d = z_A_z_B * z_A_z_B

		grad_A1 = z_A_z_B*2
		grad_B1 = - grad_A1

		denom = torch.exp(z_A_z_B*ground_truth)+1
		grad_A2 = -ground_truth/denom
		grad_B2 = ground_truth/denom

		grad_A = mask*grad_A2 + (1-mask)*grad_A1
		grad_B = mask*grad_B2 + (1-mask)*grad_B1

		return grad_A, grad_B

	# def backward(self, grad_output):
	# 	# grad_input = Variable(torch.zeros(self.input.data.size()));

	# 	n_point_total = 0
	# 	cpu_input = self.input
	# 	target = self.target

	# 	self.buffer = Variable(torch.Tensor(cpu_input.data.size())).float().cuda()
	# 	self.buffer.data.zero_()

	# 	for batch_idx in range(0,self.input.size()[0]):
	# 		n_point_total+=target[batch_idx]['n_point']
	# 		x_A_arr = target[batch_idx]['x_A']
	# 		y_A_arr = target[batch_idx]['y_A']
	# 		x_B_arr = target[batch_idx]['x_B']
	# 		y_B_arr = target[batch_idx]['y_B']

	# 		batch_input = cpu_input[batch_idx, 0]
	# 		z_A_arr = batch_input.index_select(1, x_A_arr.long()).gather(0, y_A_arr.view(1,-1).long())
	# 		z_B_arr = batch_input.index_select(1, x_B_arr.long()).gather(0, y_B_arr.view(1,-1).long())

	# 		ground_truth_arr = target[batch_idx]['ordianl_relation']

	# 		grad_A , grad_B = self._grad_loss_func(z_A_arr, z_B_arr, ground_truth_arr)

	# 		p2 = Variable(torch.Tensor(cpu_input.size()[2], target[batch_idx]['n_point'])).cuda()
	# 		p1 = Variable(torch.Tensor(cpu_input.size()[2], cpu_input.size()[3])).cuda()

	# 		p2.data.zero_()
	# 		p1.data.zero_()
	# 		print(p2.size())
	# 		print(grad_A.size())
	# 		p2.scatter_(0, y_A_arr.view(1,-1).long(), grad_A.view(1,-1))
	# 		p1.index_add_(1, x_A_arr.long(), p2)

	# 		self.buffer[batch_idx, 0, :,:] = self.buffer[batch_idx, 0, :,:]+ p1.float()

	# 		p1.data.zero_()
	# 		p2.data.zero_()
	# 		p2.scatter_(0, y_B_arr.view(1,-1).long(), grad_B.view(1,-1))
	# 		p1.index_add_(1, x_B_arr.long(), p2)

	# 		self.buffer[batch_idx, 0, :,:] = self.buffer[batch_idx, 0, :,:]+ p1.float()

	# 	return self.buffer/n_point_total





		#TODO
if __name__ == '__main__':
	crit = relative_depth_crit()
	print(crit)
	# z_A = Variable(torch.zeros(5), requires_grad = True)
	# z_B = Variable(torch.ones(5), requires_grad = True)
	# ground_truth = Variable(torch.ones(5))
	# print(crit(z_A, z_B, ground_truth))
	# mean = crit(z_A, z_B, ground_truth).mean()
	# mean.backward()
	# print(z_A.grad)
	# print(z_B.grad)
	x = Variable(torch.zeros(1,1,6,6).cuda(), requires_grad = True)
	target = {}
	target[0] = {}
	target[0]['x_A'] = Variable(torch.Tensor([0,1,2,3,4,5])).cuda()
	target[0]['y_A'] = Variable(torch.Tensor([0,1,2,3,4,5])).cuda()
	target[0]['x_B'] = Variable(torch.Tensor([0,0,0,0,0,0])).cuda()
	target[0]['y_B'] = Variable(torch.Tensor([5,4,3,2,1,0])).cuda()
	target[0]['ordianl_relation'] = Variable(torch.Tensor([-1,0,1,1,-1,-1])).cuda()
	target[0]['n_point'] = 6
	loss = crit.forward(x,target)
	print(loss)
	loss.backward()
	# a = crit.backward(1.0)
	# print(a)
	print(x.grad)
	# print(x.creator)
