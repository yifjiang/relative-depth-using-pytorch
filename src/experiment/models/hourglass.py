import torch
from torch import nn
from torch.autograd import Variable
from layers.inception import inception

class Channels1(nn.Module):
	def __init__(self):
		super(Channels1, self).__init__()
		self.list = nn.ModuleList()
		self.list.append(
			nn.Sequential(
				inception(256,[[64],[3,32,64],[5,32,64],[7,32,64]]),
				inception(256,[[64],[3,32,64],[5,32,64],[7,32,64]])
				)
			) #EE
		self.list.append(
			nn.Sequential(
				nn.AvgPool2d(2),
				inception(256,[[64],[3,32,64],[5,32,64],[7,32,64]]),
				inception(256,[[64],[3,32,64],[5,32,64],[7,32,64]]),
				inception(256,[[64],[3,32,64],[5,32,64],[7,32,64]]), 
				nn.UpsamplingNearest2d(scale_factor=2)
				)
			) #EEE

	def forward(self,x):
		return self.list[0](x)+self.list[1](x)

class Channels2(nn.Module):
	def __init__(self):
		super(Channels2, self).__init__()
		self.list = nn.ModuleList()
		self.list.append(
			nn.Sequential(
				inception(256, [[64], [3,32,64], [5,32,64], [7,32,64]]), 
				inception(256, [[64], [3,64,64], [7,64,64], [11,64,64]])
				)
			)#EF
		self.list.append( 
			nn.Sequential(
				nn.AvgPool2d(2),
				inception(256, [[64], [3,32,64], [5,32,64], [7,32,64]]), 
				inception(256, [[64], [3,32,64], [5,32,64], [7,32,64]]), 
				Channels1(),
				inception(256, [[64], [3,32,64], [5,32,64], [7,32,64]]), 
				inception(256, [[64], [3,64,64], [7,64,64], [11,64,64]]),
				nn.UpsamplingNearest2d(scale_factor=2)
				)
			)#EE1EF

	def forward(self,x):
		return self.list[0](x)+self.list[1](x)

class Channels3(nn.Module):
	def __init__(self):
		super(Channels3, self).__init__()
		self.list = nn.ModuleList()
		self.list.append(
			nn.Sequential(
				nn.AvgPool2d(2),
				inception(128, [[32], [3,32,32], [5,32,32], [7,32,32]]),
				inception(128, [[64], [3,32,64], [5,32,64], [7,32,64]]),
				Channels2(),
				inception(256, [[64], [3,32,64], [5,32,64], [7,32,64]]), 
				inception(256, [[32], [3,32,32], [5,32,32], [7,32,32]]), 
				nn.UpsamplingNearest2d(scale_factor=2)
				)
			)#BD2EG
		self.list.append(
			nn.Sequential(
				inception(128, [[32], [3,32,32], [5,32,32], [7,32,32]]), 
				inception(128, [[32], [3,64,32], [7,64,32], [11,64,32]])
				)
			)#BC

	def forward(self,x):
		return self.list[0](x)+self.list[1](x)

class Channels4(nn.Module):
	def __init__(self):
		super(Channels4, self).__init__()
		self.list = nn.ModuleList()
		self.list.append(
			nn.Sequential(
				nn.AvgPool2d(2),
				inception(128, [[32], [3,32,32], [5,32,32], [7,32,32]]),
				inception(128, [[32], [3,32,32], [5,32,32], [7,32,32]]),
				Channels3(),
				inception(128, [[32], [3,64,32], [5,64,32], [7,64,32]]),
				inception(128, [[16], [3,32,16], [7,32,16], [11,32,16]]),
				nn.UpsamplingNearest2d(scale_factor=2)
				)
			)#BB3BA
		self.list.append(
			nn.Sequential(
				inception(128, [[16], [3,32,16], [7,32,16], [11,32,16]])
				)
			)#A

	def forward(self,x):
		return self.list[0](x)+self.list[1](x)


class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()

		# first layer
		self.blockH = nn.Sequential(
			nn.Conv2d(3,128,7,padding=3), 
			nn.BatchNorm2d(128), 
			nn.ReLU(True)
			)

		self._4channels = Channels4()

		self.final = nn.Conv2d(64,1,3,padding=1)

	def forward(self,x):
		return self.final(self._4channels(self.blockH(x)))

def get_model():
	return Model()

from criterion.relative_depth import relative_depth_crit
def get_criterion():
	return relative_depth_crit()

def f_depth_from_model_output():
	print(">>>>>>>>>>>>>>>>>>>>>>>>>    depth = model_output")
	return ____get_depth_from_model_output

def ____get_depth_from_model_output(model_output):
	return model_output

		
if __name__ == '__main__':
	test = Model().cuda()
	print(test)
	x = Variable(torch.rand(1,3,320,320).cuda())
	print(test(x))