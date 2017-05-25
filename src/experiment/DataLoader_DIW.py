import os
import h5py
import math
import random
import torch
from common.NYU_params import *
from DataPointer import DataPointer
from torchvision import transforms
from PIL import Image

class DataLoader(object):
	"""docstring for DataLoader"""
	def __init__(self, relative_depth_filename):
		super(DataLoader, self).__init__()
		print(">>>>>>>>>>>>>>>>> Using DataLoader")
		self.parse_depth(relative_depth_filename)
		self.data_ptr_relative_depth = DataPointer(self.n_relative_depth_sample)
		
		print("DataLoader init: \n \t{} relative depth samples \n ".format(self.n_relative_depth_sample))

	def parse_DIW_csv(_filename):
		f = open(_filename, 'r')
		csv_file_handle = list(csv.reader(f))
		_n_lines = len(csv_file_handle)

		_handle = {}

		_sample_idx = 0
		for _line_idx in range(0,_n_lines):
			_handle[_sample_idx] = {}
			_handle[_sample_idx]['img_filename'] = csv_file_handle[_line_idx][0]
			_handle[_sample_idx]['img_filename']['n_point'] = 1
			_handle[_sample_idx]['img_filename_line_idx'] = _line_idx
			_sample_idx += 1

			_line_idx += _handle[_sample_idx]['img_filename']['n_point']

		_handle['csv_file_handle'] = csv_file_handle

		print("{}: number of sample = {}".format(_filename, _sample_idx))

		return _handle

	def parse_one_coordinate_line(csv_file_handle, _line_idx):
		orig_img_width = float(csv_file_handle[_line_idx][5])
		orig_img_height = float(csv_file_handle[_line_idx][6])

		y_A_float_orig = float(csv_file_handle[_line_idx][0])/orig_img_height
		x_A_float_orig = float(csv_file_handle[_line_idx][1])/orig_img_width
		y_B_float_orig = float(csv_file_handle[_line_idx][2])/orig_img_height
		x_B_float_orig = float(csv_file_handle[_line_idx][3])/orig_img_width

		# y_A = min(g_input_height, max(0, floor(y)))#todo