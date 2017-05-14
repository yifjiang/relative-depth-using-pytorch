import os
import h5py
from common.NYU_params import *
from DataPointer import DataPointer


class DataLoader(object):
	"""docstring for DataLoader"""
	def __init__(self, relative_depth_filename):
		super(DataLoader, self).__init__()
		print(">>>>>>>>>>>>>>>>> Using DataLoader")
		self.parse_depth(relative_depth_filename)
		self.data_ptr_relative_depth = DataPointer(self.n_relative_depth_sample)
		print("DataLoader init: \n \t{} relative depth samples \n ".format(self.n_relative_depth_sample))

	def parse_relative_depth_line(self, line):
		splits = line.split(',')
		sample = {}
		sample['img_filename'] = splits[0]
		sample['n_point'] = int(splits[2])
		return sample

	def parse_csv(self, filename, parsing_func):
		_handle = {}

		if filename == None:
			return _handle

		_n_lines = 0
		f = open(filename, 'r')
		for _ in filename:
			_n_lines+=1
		f.close()

		csv_file_handle = open(filename, 'r')
		_sample_idx = 0
		while _sample_idx < _n_lines:
			this_line = csv_file_handle.readline()
			_handle[_sample_idx] = parsing_func(this_line)
			_sample_idx+=1

		csv_file_handle.close()

		return _handle

	def parse_depth(self, relative_depth_filename):
		if relative_depth_filename is not None:
			_simplified_relative_depth_filename = relative_depth_filename.replace('.csv', '_name.csv')
			if os.path.isfile(_simplified_relative_depth_filename):
				print(_simplified_relative_depth_filename+" already exists.")
			else:
				command = "grep '.png' "+ relative_depth_filename + " > " + _simplified_relative_depth_filename
				print("executing:{}".format(command))
				os.system(command)

				self.relative_depth_handle = parse_csv(_simplified_relative_depth_filename, parse_relative_depth_line)

				hdf5_filename = relative_depth_filename.replace(',csv', '.h5')
				self.relative_depth_handle['hd5_handle'] = h5py.File(hdf5_filename, 'r')

		else:
			self.relative_depth_handle = {}

		self.n_relative_depth_sample = len(self.relative_depth_handle)

	def close():
		pass

	