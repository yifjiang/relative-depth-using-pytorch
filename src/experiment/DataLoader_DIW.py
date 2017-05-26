import os
import h5py
import math
import random
import torch
import csv
from common.NYU_params import *
from DataPointer import DataPointer
from torchvision import transforms
from PIL import Image
from math import floor

class DataLoader(object):
	"""docstring for DataLoader"""
	def __init__(self, relative_depth_filename):
		super(DataLoader, self).__init__()
		print(">>>>>>>>>>>>>>>>> Using DataLoader")
		self.parse_depth(relative_depth_filename)
		self.data_ptr_relative_depth = DataPointer(self.n_relative_depth_sample)
		
		print("DataLoader init: \n \t{} relative depth samples \n ".format(self.n_relative_depth_sample))

	def parse_DIW_csv(self, _filename):
		f = open(_filename, 'r')
		csv_file_handle = list(csv.reader(f))
		_n_lines = len(csv_file_handle)

		_handle = {}

		_sample_idx = 0
		_line_idx = 0
		while _line_idx < _n_lines:
			_handle[_sample_idx] = {}
			_handle[_sample_idx]['img_filename'] = csv_file_handle[_line_idx][0]
			_handle[_sample_idx]['n_point'] = 1
			_handle[_sample_idx]['img_filename_line_idx'] = _line_idx

			_line_idx += _handle[_sample_idx]['n_point']
			_line_idx += 1
			_sample_idx += 1

		_handle['csv_file_handle'] = csv_file_handle

		print("{}: number of sample = {}".format(_filename, _sample_idx))

		return _handle

	def parse_one_coordinate_line(self, csv_file_handle, _line_idx):
		orig_img_width = float(csv_file_handle[_line_idx][5])
		orig_img_height = float(csv_file_handle[_line_idx][6])

		y_A_float_orig = (float(csv_file_handle[_line_idx][0])-1)/orig_img_height
		x_A_float_orig = (float(csv_file_handle[_line_idx][1])-1)/orig_img_width
		y_B_float_orig = (float(csv_file_handle[_line_idx][2])-1)/orig_img_height
		x_B_float_orig = (float(csv_file_handle[_line_idx][3])-1)/orig_img_width

		y_A = min(g_input_height-1, max(0, floor(y_A_float_orig * g_input_height )))
		x_A = min(g_input_width -1, max(0, floor(x_A_float_orig * g_input_width  )))
		y_B = min(g_input_height-1, max(0, floor(y_B_float_orig * g_input_height )))
		x_B = min(g_input_width -1, max(0, floor(x_B_float_orig * g_input_width  )))

		if (y_A == y_B) and (x_A == x_B):#check this
			if y_A_float_orig > y_B_float_orig:
				y_A+=1
			elif y_A_float_orig > y_B_float_orig:
				y_A-=1
			if x_A_float_orig > x_B_float_orig:
				x_A+=1
			elif x_A_float_orig < x_B_float_orig:
				x_B-=1

		ordi = csv_file_handle[_line_idx][4][0]

		if ordi == '>':
			ordi = 1
		elif ordi == '<':
			ordi = -1
		elif ordi == '=':
			print('Error in _read_one_sample()! The ordinal_relationship should never be = !!!!')
			assert(False)
		else:
			print(ordi)
			print('Error in _read_one_sample()! The ordinal_relationship does not read correctly!!!!')
			assert(False)

		# print("Original:{}, {}, {}, {}".format(int(csv_file_handle[_line_idx][0])-1, int(csv_file_handle[_line_idx][1])-1, int(csv_file_handle[_line_idx][2])-1, int(csv_file_handle[_line_idx][3])-1))
		# print("Size    : height:{}, width{}".format(orig_img_height, orig_img_width))
		# print("Float  :{}, {}, {}, {}".format(y_A_float_orig, x_A_float_orig, y_B_float_orig, x_B_float_orig))
		# print("Scaled: {}, {}, {}, {}".format(y_A, x_A, y_B, x_B, ordi))
		# print("relationship: {}".format(ordi))

		return y_A, x_A, y_B, x_B, ordi

	def parse_depth(self, relative_depth_filename):
		if relative_depth_filename is not None:
			self.relative_depth_handle = self.parse_DIW_csv(relative_depth_filename)
		else:
			self.relative_depth_handle = {}

		self.n_relative_depth_sample = len(self.relative_depth_handle)-1

	def close(self):
		pass

	def mixed_sample_strategy1(self, batch_size):
		n_depth = random.randint(0, batch_size-1)
		return n_depth, batch_size - n_depth

	def mixed_sample_strategy2(self, batch_size):
		n_depth = floor(batch_size/2)
		return n_depth, batch_size - n_depth

	def load_indices(self, depth_indices):
		if depth_indices is not None:
			n_depth = len(depth_indices)
		else:
			n_depth = 0

		batch_size = n_depth
		color = torch.Tensor(batch_size, 3, g_input_height, g_input_width)

		_batch_target_relative_depth_gpu = {}
		_batch_target_relative_depth_gpu['n_sample'] = n_depth

		loader = transforms.Compose([
			transforms.Scale((g_input_width, g_input_height)),
			transforms.ToTensor(),
			# transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) # may not need this
			])

		csv_file_handle = self.relative_depth_handle['csv_file_handle']

		for i in range(0,n_depth):
			chosen_idx = depth_indices[i]
			_batch_target_relative_depth_gpu[i] = {}
			img_name = self.relative_depth_handle[chosen_idx]['img_filename']
			n_point = self.relative_depth_handle[chosen_idx]['n_point']

			img = Image.open(img_name)
			img = loader(img).float()
			if img.size()[0] == 1:
				print(img_name, ' is gray')
				color[i,0,:,:].copy_(img)
				color[i,1,:,:].copy_(img)
				color[i,2,:,:].copy_(img)
			else:
				color[i,:,:,:].copy_(img)

			_line_idx = self.relative_depth_handle[chosen_idx]['img_filename_line_idx']+1
			y_A, x_A, y_B, x_B, ordi = self.parse_one_coordinate_line(csv_file_handle, _line_idx)

			_batch_target_relative_depth_gpu[i]['y_A'] = torch.autograd.Variable(torch.Tensor([y_A])).cuda()
			_batch_target_relative_depth_gpu[i]['x_A'] = torch.autograd.Variable(torch.Tensor([x_A])).cuda()
			_batch_target_relative_depth_gpu[i]['y_B'] = torch.autograd.Variable(torch.Tensor([y_B])).cuda()
			_batch_target_relative_depth_gpu[i]['x_B'] = torch.autograd.Variable(torch.Tensor([x_B])).cuda()
			_batch_target_relative_depth_gpu[i]['ordianl_relation'] = torch.autograd.Variable(torch.Tensor([ordi])).cuda()
			_batch_target_relative_depth_gpu[i]['n_point'] = n_point

		return torch.autograd.Variable(color.cuda()), _batch_target_relative_depth_gpu

	def load_next_batch(self, batch_size):
		depth_indices = self.data_ptr_relative_depth.load_next_batch(batch_size)
		return self.load_indices(depth_indices)

	def reset(self):
		self.current_pos = 1