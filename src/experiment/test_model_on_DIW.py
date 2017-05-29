from PIL import Image
from math import sqrt
import torch
import os
import csv
import numpy as np
import argparse
import h5py
from torchvision import transforms
from torch.autograd import Variable

def _read_data_handle(_filename):
	f = open(_filename, 'r')
	csv_file_handle = list(csv.reader(f))
	_n_lines = len(csv_file_handle)

	_data = {}
	_line_idx = 0
	_sample_idx = 0
	while _line_idx < _n_lines:

		dic = {};
		dic['img_filename'] = csv_file_handle[_line_idx][0]
		dic['n_point'] = 1
		dic['img_filename_line_idx'] = _line_idx

		_line_idx+=dic['n_point']
		_line_idx+=1

		_data[_sample_idx] = dic

		_sample_idx+=1

	print('number of sample =', len(_data))
	_data['csv_file_handle'] = csv_file_handle

	return _sample_idx, _data

def _read_one_sample(_sample_idx, handle):
	_data = {}
	n_point = handle[_sample_idx]['n_point']
	_data['img_filename'] = handle[_sample_idx]['img_filename']
	_data['n_point'] = handle[_sample_idx]['n_point']

	_data['y_A'] = []
	_data['y_B'] = []
	_data['x_A'] = []
	_data['x_B'] = []
	_data['ordianl_relation'] = []

	_line_idx = handle[_sample_idx]['img_filename_line_idx']+1

	for point_idx in range(0,handle[_sample_idx]['n_point']):
		_data['y_A'].append(int(handle['csv_file_handle'][_line_idx][0])-1)
		_data['x_A'].append(int(handle['csv_file_handle'][_line_idx][1])-1)
		_data['y_B'].append(int(handle['csv_file_handle'][_line_idx][2])-1)
		_data['x_B'].append(int(handle['csv_file_handle'][_line_idx][3])-1)

		if _data['y_A'][point_idx] == _data['y_B'][point_idx] and _data['x_A'][point_idx] == _data['x_B'][point_idx]:
			print('The coordinates shouldn not be equal!!!!')
			assert(False)

		ordi = handle['csv_file_handle'][_line_idx][4][0]

		if ordi == '>':
			_data['ordianl_relation'].append(1)
		elif ordi == '<':
			_data['ordianl_relation'].append(-1)
		elif ordi == '=':
			print('Error in _read_one_sample()! The ordinal_relationship should never be = !!!!')
			assert(False)
		else:
			print('Error in _read_one_sample()! The ordinal_relationship does not read correctly!!!!')
			assert(False)

		_line_idx+=1

	return _data

def inpaint_pad_output_eigen(output):
	pass#todo

def _evaluate_correctness(_batch_output, _batch_target, record):
	assert(_batch_target['n_point'] == 1)
	for point_idx in range(0,_batch_target['n_point']):
		x_A = _batch_target['x_A'][point_idx]
		y_A = _batch_target['y_A'][point_idx]
		x_B = _batch_target['x_B'][point_idx]
		y_B = _batch_target['y_B'][point_idx]

		z_A = _batch_output[0,0,y_A,x_A]
		z_B = _batch_output[0,0,y_B,x_B]

		ground_truth = _batch_target['ordianl_relation'][point_idx]

		if ((z_A-z_B) * ground_truth)>0:
			if ground_truth > 0:
				record['n_gt_correct']+=1
			else:
				record['n_lt_correct']+=1

		if ground_truth>0:
			record['n_gt']+=1
		elif ground_truth < 0:
			record['n_lt']+=1
		elif ground_truth == 0:
			print('The input should not contain equal terms!')
			assert(False)

def print_result(record):
	print('Less_than correct ratio = {}, n_lt_correct = {}, n_lt = {}'.format(record['n_lt_correct']/record['n_lt'], record['n_lt_correct'], record['n_lt']))
	print('Greater_than correct ratio = {}, n_gt_correct = {}, n_gt = {}'.format(record['n_gt_correct']/record['n_gt'], record['n_gt_correct'], record['n_gt']))
	print('Overall correct ratio = {}'.format((record['n_lt_correct']+record['n_gt_correct'])/(record['n_lt']+record['n_gt'])))

###### main ######

parser = argparse.ArgumentParser()
parser.add_argument('-num_iter', default=1, type = int, help ='number of testing iteration')
parser.add_argument('-prev_model_file', required=True, help='Absolute/relative path to the previous model file. Resume training from this file')
parser.add_argument('-test_model', default = 'our', help = 'eigen, our or debug')
parser.add_argument('-vis', default = False, type = bool, help='visualize output')
parser.add_argument('-output_folder', default = './output_imgs_DIW', help = 'image output folder')
cmd_params = parser.parse_args()

csv_file_name = '../../data/DIW_test.csv'
print('loading csv file...')
n_sample, data_handle = _read_data_handle(csv_file_name)

print('Hyper params: ')
print('csv_file_name:', csv_file_name)
print('N test samples:', n_sample)

num_iter = min(n_sample, cmd_params.num_iter)
print('num_iter:', cmd_params.num_iter)

if cmd_params.test_model == 'debug':
	pass#todo

if cmd_params.test_model == 'eigen':
	pass#todo

if cmd_params.test_model == 'our':
	_network_input_width = 320
	_network_input_height = 240

	prev_model_file = cmd_params.prev_model_file
	model = torch.load(prev_model_file)

	print('Model file:', prev_model_file)

	our_result = {}
	our_result['n_gt_correct'] = 0
	our_result['n_gt'] = 0

	our_result['n_lt_correct'] = 0
	our_result['n_lt'] = 0

	our_result['n_eq_correct'] = 0
	our_result['n_eq'] = 0

	_batch_input_cpu = torch.Tensor(1,3,_network_input_height, _network_input_width)	

	t = transforms.Compose([
		transforms.Scale((_network_input_width, _network_input_height)),
		transforms.ToTensor()
		])

	for i in range(0,num_iter):
		print(i)
		thumb_filename = data_handle[i]['img_filename']
		orig_img = Image.open(thumb_filename).convert('RGB')
		orig_width, orig_height = orig_img.size


		print('Processing sample', thumb_filename)
		img = t(orig_img)

		if img.size()[0] == 1:
			print(data_handle[i]['img_filename'], 'is gray')
			_batch_input_cpu[0,0,:,:].copy_(img)
			_batch_input_cpu[0,1,:,:].copy_(img)
			_batch_input_cpu[0,2,:,:].copy_(img)
		else:
			_batch_input_cpu[0,:,:,:].copy_(img)

		img = None #get rid of its reference

		_single_data = {}
		_single_data[0] = _read_one_sample(i, data_handle)

		batch_output = model(Variable(_batch_input_cpu).cuda()).cpu().data

		batch_output_min = torch.min(batch_output)
		batch_output_max = torch.max(batch_output) - batch_output_min

		orig_size_output = torch.Tensor(1,1,orig_height, orig_width)
		scale = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Scale((orig_width, orig_height)),
			transforms.ToTensor()
			])
		batch_output-=batch_output_min
		batch_output/=batch_output_max
		temp = scale(batch_output[0])
		temp*=batch_output_max
		temp+=batch_output_min
		orig_size_output[0].copy_(temp)

		_evaluate_correctness(orig_size_output, _single_data[0], our_result)

		if i%100 == 0 and i != 0:
			print_result(our_result)


		if cmd_params.vis:
			orig_size_output = orig_size_output[0]
			orig_size_output = orig_size_output - torch.min(orig_size_output)
			orig_size_output = orig_size_output / torch.max(orig_size_output)
			t_back = transforms.ToPILImage()
			orig_size_output = t_back(orig_size_output)

			new_image = Image.new('RGB', (orig_width*2, orig_height))
			new_image.paste(orig_img, (0,0))
			new_image.paste(orig_size_output, (orig_width, 0))

			new_image.save(os.path.join(cmd_params.output_folder, str(i+1)+'.png'))

print("Summary:=========================================================================")
if cmd_params.test_model == 'our':
	print_result(our_result)