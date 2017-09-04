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
			dic['y_A'].append(int(csv_file_handle[_line_idx][0])-1)
			dic['x_A'].append(int(csv_file_handle[_line_idx][1])-1)
			dic['y_B'].append(int(csv_file_handle[_line_idx][2])-1)
			dic['x_B'].append(int(csv_file_handle[_line_idx][3])-1)

			if csv_file_handle[_line_idx][4] == '>':
				dic['ordianl_relation'].append(1)
			elif csv_file_handle[_line_idx][4] == '<':
				dic['ordianl_relation'].append(-1)
			elif csv_file_handle[_line_idx][4] == '=':
				dic['ordianl_relation'].append(0)

			_line_idx+=1

		_data.append(dic)

	# print(_data[0], _data[1])

	return len(_data), _data

def _evaluate_correctness_out(_batch_output, _batch_target, WKDR, WKDR_eq, WKDR_neq):
	# print(_batch_output.size())

	# print(_batch_output)

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

		z_A_z_B = (z_A - z_B)

		# print(z_A_z_B, y_A,x_A, y_B,x_B, ground_truth)

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
		WKDR[i] = (1-(n_eq_correct[i]+n_lt_correct[i]+n_gt_correct[i])/(n_eq+n_lt+n_gt))
		WKDR_eq[i] = (1 - n_eq_correct[i]/n_eq)
		WKDR_neq[i] = (1 - (n_lt_correct[i] + n_gt_correct[i])/(n_lt+n_gt))

def inpaint_pad_output_our(output, img_original_width, img_original_height):
	crop = cmd_params.crop
	resize_height = img_original_height - 2*crop
	resize_width = img_original_width -2*crop
	output_min = torch.min(output)
	output_max = torch.max(output)-output_min
	output-=output_min
	output/=output_max
	scale = transforms.Compose([
		transforms.ToPILImage(),
		transforms.Scale((resize_width, resize_height)),
		transforms.ToTensor()
		]) 
	resize_output = scale(torch.Tensor(output[0]))
	resize_output*=output_max
	resize_output+=output_min
	padded_output = torch.Tensor(1,img_original_height, img_original_width)

	padded_output[0,crop:img_original_height-crop, crop:img_original_width-crop].copy_(resize_output)

	for i in range(0,crop):
		# print(resize_output[0, :,0])
		padded_output[0, crop:img_original_height-crop, i].copy_(resize_output[0, :,0])
		# print(resize_output[0,:,resize_width-1])
		padded_output[0, crop:img_original_height-crop, img_original_width-1-i].copy_(resize_output[0,:,resize_width-1])

	for i in range(0,crop):
		# print(padded_output[0,crop,:])
		padded_output[0, i, :].copy_(padded_output[0,crop,:])
		# print(padded_output[0, resize_height+crop-1 ,:])
		padded_output[0, img_original_height-1-i, :].copy_(padded_output[0, resize_height+crop-1 ,:])

	return padded_output

def metric_error(gtz, z):
	gtz = torch.Tensor(gtz).cuda()
	z = z.cuda()
	fmse = torch.mean(torch.pow(gtz-z,2))
	fmselog = torch.mean(torch.pow(torch.log(gtz)-torch.log(z),2))
	flsi = torch.mean(torch.pow(torch.log(z)-torch.log(gtz) + torch.mean(torch.log(gtz)-torch.log(z)), 2))
	fabsrel = torch.mean(torch.abs(z-gtz)/gtz)
	fsqrrel = torch.mean(torch.pow(z-gtz, 2)/gtz)

	return fmse, fmselog, flsi, fabsrel, fsqrrel

def normalize_output_depth_with_NYU_mean_std(input):
	std_of_NYU_training = 0.6148231626
	mean_of_NYU_training = 2.8424594402

	transformed_z = input.clone()
	transformed_z -= torch.mean(transformed_z)
	transformed_z /= torch.std(transformed_z)
	transformed_z *= std_of_NYU_training
	transformed_z += mean_of_NYU_training

	if torch.sum(transformed_z<0) > 0:
		transformed_z = (transformed_z<0).float()*(torch.min(transformed_z>0)+0.00001)+transformed_z*(1- transformed_z<0).float()

	return transformed_z

### main entry ###

parser = argparse.ArgumentParser()
parser.add_argument('-num_iter', default=1, type = int, help ='number of testing iteration')
parser.add_argument('-prev_model_file', required=True, help='Absolute/relative path to the previous model file. Resume training from this file')
parser.add_argument('-vis', default = False, type = bool, help='visualize output')
parser.add_argument('-output_folder', default = './output_imgs', help = 'image output folder')
parser.add_argument('-mode', default='validate', help = 'mode: test or validate')
parser.add_argument('-valid_set', default='45_validate_from_795_NYU_MITpaper_train_imgs_800_points_resize_240_320.csv', help = 'validation file name')
parser.add_argument('-test_set', default = '654_NYU_MITpaper_test_imgs_orig_size_points.csv', help = 'test file name')
parser.add_argument('-crop', default = 16, type = int, help = 'cropping size')
parser.add_argument('-thresh', default = -1, help = 'threshold for determining WKDR. Obtained from validations set.')

cmd_params = parser.parse_args()

if cmd_params.mode == 'test':
	csv_file_name = os.path.join('../../data/',cmd_params.test_set)
elif cmd_params.mode  == 'validate':
	csv_file_name = os.path.join('../../data/',cmd_params.valid_set)
preload_pt_filename = csv_file_name.replace('.csv','.pt')

if not os.path.exists(cmd_params.output_folder):
	os.mkdir(cmd_params.output_folder)

try:
	f = open(preload_pt_filename, 'r')
	f.close()
	print('loading preload pt file...')
	data_handle = torch.load(preload_pt_filename)
	n_sample = len(data_handle)
except Exception as e:
	print('loading csv file...')
	n_sample, data_handle = _read_data_handle(csv_file_name)
	torch.save(data_handle, preload_pt_filename)


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
	transforms.ToTensor(),
	# transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))#may not need this
	])

for i in range(0, n_iter):
	print(i)
	img = Image.open(data_handle[i]['img_filename']).convert('RGB')
	img_original_width, img_original_height = img.size

	crop = transforms.CenterCrop((img_original_height- 2*cmd_params.crop, img_original_width - 2*cmd_params.crop ))

	if cmd_params.mode == 'test':
		_batch_input_cpu[0] = t(crop(img))
	elif cmd_params.mode == 'validate':
		_batch_input_cpu[0] = t(img)

	_single_data = []
	_single_data.append(data_handle[i])

	batch_output = model(Variable(_batch_input_cpu).cuda())

	# temp = batch_output

	batch_output = batch_output.cpu().data #now a tensor
	batch_output_min = torch.min(batch_output)
	batch_output_max = torch.max(batch_output)-batch_output_min

	original_size_output = torch.Tensor(1,1,img_original_height, img_original_width)

	if cmd_params.mode == 'test':
		original_size_output[0].copy_(inpaint_pad_output_our(batch_output, img_original_width, img_original_height))

		_evaluate_correctness_out(original_size_output, _single_data[0], WKDR[i], WKDR_eq[i], WKDR_neq[i])

		gtz_h5_handle = h5py.File(os.path.join(os.path.dirname(data_handle[i]['img_filename']),str(i+1)+'_depth.h5'), 'r') #todo
		gtz = gtz_h5_handle['/depth']
		# print(gtz)
		assert(gtz.shape[0] == 480)
		assert(gtz.shape[1] == 640)

		transformed_z_orig_size = normalize_output_depth_with_NYU_mean_std(original_size_output[0,0,:,:])

		metric_test_crop = 16
		transformed_z_orig_size = transformed_z_orig_size[metric_test_crop:(img_original_height- metric_test_crop), metric_test_crop:(img_original_width- metric_test_crop)]
		gtz = gtz[metric_test_crop:(img_original_height- metric_test_crop), metric_test_crop:(img_original_width- metric_test_crop)]
		# print(gtz)

		fmse[i], fmselog[i], flsi[i], fabsrel[i], fsqrrel[i] = metric_error(gtz,transformed_z_orig_size)

		gtz_h5_handle.close()
		
	elif cmd_params.mode == 'validate':
		scale = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Scale((img_original_width, img_original_height)),
			transforms.ToTensor()
			])#temporal solution
		batch_output -= batch_output_min
		batch_output /= batch_output_max
		temp = scale(batch_output[0])
		temp *= batch_output_max
		temp += batch_output_min
		original_size_output[0].copy_(temp)
		_evaluate_correctness_out(original_size_output, _single_data[0], WKDR[i], WKDR_eq[i], WKDR_neq[i])

	if cmd_params.vis:
		orig_size_output = original_size_output[0]
		orig_size_output = orig_size_output - torch.min(orig_size_output)
		orig_size_output = orig_size_output / torch.max(orig_size_output)
		t_back = transforms.ToPILImage()
		orig_size_output = t_back(orig_size_output)

		new_image = Image.new('RGB', (img_original_width*2, img_original_height))
		new_image.paste(img, (0,0))
		new_image.paste(orig_size_output, (img_original_width, 0))

		new_image.save(os.path.join(cmd_params.output_folder, str(i+1)+'.png'))


WKDR = torch.mean(WKDR, 0)
WKDR_eq = torch.mean(WKDR_eq, 0)
WKDR_neq = torch.mean(WKDR_neq, 0)
overall_summary = torch.Tensor(n_thresh, 4)

min_max = 100
min_max_i = 0
min_WKDR = 100
min_WKDR_i = 0

for i in range(0,n_thresh):
	overall_summary[i,0] = thresh[i]
	overall_summary[i,1] = WKDR[0,i]#-TODO: check the dimension
	overall_summary[i,2] = WKDR_eq[0,i]
	overall_summary[i,3] = WKDR_neq[0,i]
	if max(WKDR_eq[0,i], WKDR_neq[0,i])<min_max:
		min_max = max(WKDR_eq[0,i], WKDR_neq[0,i])
		min_max_i = i
	if WKDR[0,i] < min_WKDR:
		min_WKDR = WKDR[0,i]
		min_WKDR_i = i

if cmd_params.thresh < 0:
	print(overall_summary)
	print("====================================================================")
	if min_max_i > 0:
		if min_max_i < n_thresh-1:
			print(overall_summary[min_max_i-1:min_max_i+2, :])
	print(overall_summary[min_WKDR_i])
else:
	print('Result:\n')
	for i in range(0,n_thresh):
		if overall_summary[i,0] == cmd_params.thresh:
			print(' Thresh\tWKDR\tWKDR_eq\tWKDR_neq')
			print(overall_summary[i])

if cmd_params.mode == 'test':
	print("====================================================================")
	print("rmse:\t{}".format(sqrt(torch.mean(fmse))))
	print("rmselog:{}".format(sqrt(torch.mean(fmselog))))
	print("lsi:\t{}".format(sqrt(torch.mean(flsi))))
	print("absrel:\t{}".format(torch.mean(fabsrel)))
	print("sqrrel:\t{}".format(torch.mean(fsqrrel)))
