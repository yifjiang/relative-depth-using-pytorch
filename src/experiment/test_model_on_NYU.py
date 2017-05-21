from PIL import Image
import torch
import os
import csv
import numpy as np
import argparse
import h5py
from torchvision import transforms
from torch.autograd import Variable

def _read_data_handle(_filename):
	f = open(_filename, 'rb')
	csv_file_handle = csv.reader(f)
	_n_lines = len(csv_file_handle)

	_data = []
	_line_idx = 1
	while _line_idx < _n_lines:

		dic = {};
		dic['img_filename'] = csv_file_handle[_line_idx][0]
		dic['n_point'] = int(csv_file_handle[_line_idx][2])

		dic['y_A'] = []
		dic['y_B'] = []
		dic['x_A'] = []
		dic['x_B'] = []
		dic['ordianl_relation'] = []

		_line_idx+=1

		for point_idx in range(0,dic['n_point']):
			dic['y_A'].append(int(csv_file_handle[_line_idx][0]))
			dic['x_A'].append(int(csv_file_handle[_line_idx][1]))
			dic['y_B'].append(int(csv_file_handle[_line_idx][2]))
			dic['x_B'].append(int(csv_file_handle[_line_idx][3]))

			if csv_file_handle[_line_idx][4] == '>':
				dic['ordianl_relation'].append(1)
			elif csv_file_handle[_line_idx][4] == '<':
				dic['ordianl_relation'].append(-1)
			elif csv_file_handle[_line_idx][4] == '=':
				dic['ordianl_relation'].append(0)

			_line_idx+=1

		_data.append(dic)

	return len(_data), _data

def _evaluate_correctness_out(_batch_output, _batch_target, WKDR, WKDR_eq, WKDR_neq):
	n_gt_correct = torch.zeros(n_thresh)
	n_gt = 0

	n_lt_correct = torch.zeros(n_thresh)
	n_lt = 0

	n_eq_correct = torch.zeros(n_thresh)
	n_eq = 0

	for point_idx in range(0,_batch_target['n_point']):
		x_A = _batch_target['x_A'][point_idx]
		y_A = _batch_target['y_A'][point_idx]
		x_B = _batch_target['x_B'][point_idx]
		y_B = _batch_target['y_B'][point_idx]

		z_A = _batch_output[0,0,y_A,x_A]
		z_B = _batch_output[0,0,y_B,x_B]

		ground_truth = _batch_target['ordianl_relation'][point_idx]

		z_A_z_B = (z_A - z_B).data[0]

		for thresh_idx in range(0,n_thresh):
			if z_A_z_B > thresh[thresh_idx]:
				_classify_res = 1
			elif z_A_z_B < -thresh[thresh_idx]:
				_classify_res = -1
			else:
				_classify_res = 0

			if _classify_res == 0 and ground_truth == 0:
				n_eq_correct[thresh_idx]+=1
			elif _classify_res == 1 and ground_truth == 1:
				n_gt_correct[thresh_idx]+=1
			elif _classify_res == -1 and ground_truth == -1:
				n_lt_correct[thresh_idx]+=1

		if ground_truth > 0:
			n_gt += 1
		elif ground_truth < 0:
			n_lt+=1
		elif ground_truth == 0:
			n_eq+=1

	for i in range(0,n_thresh):
		WKDR[i] = (1-(n_eq_correct[i]+n_lt_correct[i]+n_gt_correct[i]).float()/(n_eq+n_lt+n_gt))
		WKDR_eq[i] = (1 - n_eq_correct[i].float()/n_eq)
		WKDR_neq[i] = (1 - (n_lt_correct[i] + n_gt_correct[i]).float()/(n_lt+n_gt))

def inpaint_pad_output_our(output, img_original_width, img_original_height):
	crop = cmd_params.crop
	resize_height = img_original_height - 2*crop
	resize_width = img_original_width -2*crop
	scale = transforms.Compose([
		transforms.ToPILImage(),
		transforms.Scale((resize_width, resize_height)),
		transforms.ToTensor()
		]) 
	resize_output = scale(torch.Tensor(output[0]))
	padded_output = torch.Tensor(1,img_original_height, img_original_width)

	padded_output[0,crop:img_original_height-crop, crop:img_original_width-crop].copy_(resize_output)

	for i in range(0,crop):
		padded_output[0, crop:img_original_height-crop, i].copy_(resize_output[:,0])
		padded_output[0, crop:img_original_height-crop, img_original_width-1-i].copy_(resize_output[:,resize_width-1])

	for i in range(0,crop):
		padded_output[0, i, :].copy_(padded_output[0,crop,:])
		padded_output[0, img_original_height-1-i, :].copy_(padded_output[0, resize_height+crop-1 ,:])

	return padded_output

### main entry ###

parser = argparse.ArgumentParser()
parser.add_argument('-num_iter', default=1, type = int, help ='number of training iteration')
parser.add_argument('-prev_model_file', required=True, type=string, help='Absolute/relative path to the previous model file. Resume training from this file')
parser.add_argument('-vis', default = False, type = bool, help='visualize output')
parser.add_argument('-output_folder', default = './output_imgs', help = 'image output folder')
parser.add_argument('-mode', default='validate', help = 'mode: test or validate')
parser.add_argument('-valid_set', default='45_NYU_validate_imgs_points_resize_240_320.csv', help = 'validation file name')
parser.add_argument('-test_set', default = '654_NYU_MITpaper_test_imgs_orig_size_points.csv', help = 'test file name')
parser.add_argument('-crop', default = 10, type = int, help = 'cropping size')
parser.add_argument('-thresh', default = -1, help = 'threshold for determining WKDR. Obtained from validations set.')

cmd_params = parser.parse_args()

if cmd_params.mode == 'test':
	csv_file_name = os.path.join('../../data/',cmd_params.test_set)
elif cmd_params.mode  == 'validate':
	csv_file_name = os.path.join('../../data/',cmd_params.valid_set)
preload_pt_filename = csv_file_name.replace('.csv','.pt')

f = open(preload_pt_filename, 'r')
if f is None:
	print('loading csv file...')
	n_sample, data_handle = _read_data_handle(csv_file_name)
	torch.save(data_handle, preload_pt_filename)
else:
	f.close()
	print('loading preload pt file...')
	data_handle = torch.load(preload_pt_filename)
	n_sample = len(data_handle)

print('Hyper params: ')
print('csv_file_name:', csv_file_name)
print('N test samples:', n_sample)
n_iter = min(n_sample, cmd_params.num_iter)
print('n_iter = {}'.format(n_iter))

prev_model_file = cmd_params.prev_model_file
model = torch.load(prev_model_file)
print('Model file:', prev_model_file)

from common.NYU_params import g_input_height as network_input_height
from common.NYU_params import g_input_width as network_input_width
_batch_input_cpu = torch.Tensor(1,3,network_input_height,network_input_width)

n_thresh = 140
thresh = torch.Tensor(n_thresh)
for i in range(0,n_thresh):
	thresh[i] = 0.1 + (i+1) * 0.01

WKDR = torch.zeros(n_iter, n_thresh)
WKDR_eq = torch.zeros(n_iter,n_thresh)
WKDR_neq = torch.zeros(n_iter,n_thresh)
fmse = torch.zeros(n_iter)
fmselog = torch.zeros(n_iter)
flsi = torch.zeros(n_iter)
fabsrel = torch.zeros(n_iter)
fsqrrel = torch.zeros(n_iter)

t = transforms.Compose([
	transforms.Scale((network_input_width, network_input_height)),
	transforms.toTensor()
	])

for i in range(0, n_iter):
	img = Image.open(data_handle[i].img_filename).convert('RGB')
	img_original_width, img_original_height = img.size

	crop = transforms.CenterCrop((img_original_height- 2*cmd_params.crop, img_original_width - 2*cmd_params.crop ))

	if cmd_params.mode == 'test':
		_batch_input_cpu[0] = t(crop(img))
	elif cmd_params.mode == 'validate':
		_batch_input_cpu[0] = t(img)

	_single_data = []
	_single_data.append(data_handle[i])

	batch_output = model(Variable(_batch_input_cpu).cuda())

	temp = batch_output

	batch_output = batch_output.data #now a tensor

	original_size_output = torch.Tensor(1,1,img_original_height, img_original_width)

	if cmd_params.mode == 'test':
		original_size_output[0].copy_(inpaint_pad_output_our(batch_output, img_original_width, img_original_height))

		_evaluate_correctness_out(original_size_output, _single_data[0], WKDR[i], WKDR_eq[i], WKDR_neq[i])

		# gtz_h5_ #todo
		
	elif cmd_params.mode == 'validate':
		scale = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Scale((img_original_width, img_original_height)),
			transforms.ToTensor()
			])
		original_size_output[0].copy_(scale(batch_output[0]))
		_evaluate_correctness_out(original_size_output, _single_data[0], WKDR[i], WKDR_eq[i], WKDR_neq[i])

WKDR = torch.mean(WKDR, 0)
WKDR_eq = torch.mean(WKDR_eq, 0)
WKDR_neq = torch.mean(WKDR_neq, 0)
overall_summary = torch.Tensor(n_thresh, 4)

min_max = 100
min_max_i = 0

for i in range(0,n_thresh):
	overall_summary[i,0] = thresh[i]
	overall_summary[i,1] = WKDR[0,i]#-TODO: check the dimension
	overall_summary[i,2] = WKDR_eq[0,i]
	overall_summary[i,3] = WKDR_neq[0,i]
	if max(WKDR_eq[0,i], WKDR_neq[0,i])<min_max:
		min_max = max(WKDR_eq[0,i], WKDR_neq[0,i])
		min_max_i = i

if cmd_params.thresh < 0:
	print(overall_summary)
	print("====================================================================")
	if min_max_i > 0:
		if min_max_i < n_thresh-1:
			print(overall_summary[min_max_i-1, min_max_i+1, :])
else:
	print('Result:\n')
	for i in range(0,n_thresh):
		if overall_summary[i,0] == cmd_params.thresh:
			print(' Thresh\tWKDR\tWKDR_eq\tWKDR_neq')
			print(overall_summary[i])
