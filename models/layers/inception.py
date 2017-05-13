import torch
from torch import nn
from torch.nn import ModuleList
from torch.autograd import Variable

class inception(nn.Module):
	def __init__(self, input_size, config):
		super(inception,self).__init__()
		self.convs = ModuleList()

		# Base 1*1 conv layer
		self.convs.add_module('conv1',nn.Sequential(
			nn.Conv2d(input_size, config[0][0],1),
			nn.BatchNorm2d(config[0][0],affine=False),
			nn.ReLU(True),
		))

		# Additional layers
		for i in range(1, len(config)):
			conv = nn.Sequential()
			filt = config[i][0]
			pad = (filt-1)/2
			out_a = config[i][1]
			out_b = config[i][2]
			# Reduction
			conv.add_module('conv1times1', nn.Conv2d(input_size, out_a,1))
			conv.add_module('norm1', nn.BatchNorm2d(out_a,affine=False))
			conv.add_module('relu1', nn.ReLU(True))
			# Spatial Convolution
			conv.add_module('conv', nn.Conv2d(out_a, out_b, filt,padding=pad))
			conv.add_module('norm2', nn.BatchNorm2d(out_b,affine=False))
			conv.add_module('relu2', nn.ReLU(True))
			self.convs.add_module('conv'+str(i+1),conv)

	def forward(self, x):
		ret = []
		for conv in self.convs:
			ret.append(conv(x))
		return torch.cat(ret,dim=1)

if __name__ == '__main__':
	testModule = inception(256, [[64], [3,32,64], [5,32,64], [7,32,64]])
	print(testModule)