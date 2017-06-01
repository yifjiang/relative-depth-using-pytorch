import torch
import gc

def _classify(z_A,z_B,ground_truth,thresh):
	if z_A - z_B > thresh:
		_classify_res = 1
	elif z_A - z_B < -thresh:
		_classify_res = -1
	else: #elif z_A - z_B <= thresh and z_A - z_B >= -thresh: #this may be unecessary
		_classify_res = 0
	return (_classify_res == ground_truth)

def _count_correct(output, target, record):
	for point_idx in range(0,target['n_point']):
		x_A = target['x_A'][point_idx]
		y_A = target['y_A'][point_idx]
		x_B = target['x_B'][point_idx]
		y_B = target['y_B'][point_idx]

		z_A = output[0,0, y_A.data.int()[0], x_A.data.int()[0]].data[0] #zero-indexed
		z_B = output[0,0, y_B.data.int()[0], x_B.data.int()[0]].data[0]

		assert(x_A.data[0] != x_B.data[0] or y_A.data[0] != y_B.data[0])

		ground_truth = target['ordianl_relation'][point_idx].data[0]

		for tau_idx in range(0,record['n_thresh']):
			if _classify(z_A, z_B, ground_truth, record['thresh'][tau_idx]):
				if ground_truth == 0:
					record['eq_correct_count'][tau_idx] += 1
				else: #elif ground_truth == 1 or ground_truth == -1 #this may be unnecessary
					record['not_eq_correct_count'][tau_idx] += 1

		if ground_truth == 0:
			# print(ground_truth)
			record['eq_count'] += 1
		else:
			record['not_eq_count'] += 1


_eval_record = {}
_eval_record['n_thresh'] = 15
_eval_record['eq_correct_count'] = torch.Tensor(_eval_record['n_thresh'])
_eval_record['not_eq_correct_count'] = torch.Tensor(_eval_record['n_thresh'])
_eval_record['not_eq_count'] = 0
_eval_record['eq_count'] = 0
_eval_record['thresh'] = torch.Tensor(_eval_record['n_thresh'])
_eval_record['WKDR'] = torch.Tensor(_eval_record['n_thresh'], 4)
for i in range(0,_eval_record['n_thresh']):
	_eval_record['thresh'][i] = float(i)*0.1


def reset_record(record):
	record['eq_correct_count'].fill_(0)
	record['not_eq_correct_count'].fill_(0)
	record['WKDR'].fill_(0)
	record['not_eq_count'] = 0
	record['eq_count'] = 0

def evaluate(data_loader, model, criterion, max_n_sample):
	print('>>>>>>>>>>>>>>>>>>>>>>>>> Valid Crit Threshed: Evaluating on validation set...')
	print('Evaluate() Switch On!!!')
	# model.evaluate()
	reset_record(_eval_record)

	total_validation_loss = 0
	n_iters = min(data_loader.n_relative_depth_sample, max_n_sample)
	n_total_point_pair = 0

	print("Number of samples we are going to examine: {}".format(n_iters))

	for i in range(0,n_iters):
		batch_input, batch_target = data_loader.load_indices(torch.Tensor([i])) 

		relative_depth_target = batch_target[0]

		# print(i)
		batch_output = model.forward(batch_input)
		batch_loss = criterion.forward(batch_output, batch_target) 

		# output_depth = get_depth_from_model_output(batch_output) #get_depth_from_model_output is from main.py
		output_depth = batch_output #temporary solution

		_count_correct(output_depth, relative_depth_target, _eval_record)

		total_validation_loss += (batch_loss * relative_depth_target['n_point']).data[0]

		n_total_point_pair += relative_depth_target['n_point']

		gc.collect()
	
	print('Evaluate() Switch Off!!!')
	# model.training()

	max_min = 0
	max_min_i = 1
	for tau_idx in range(0,_eval_record['n_thresh']):
		_eval_record['WKDR'][tau_idx, 0] = _eval_record['thresh'][tau_idx]
		_eval_record['WKDR'][tau_idx, 1] = float(_eval_record['eq_correct_count'][tau_idx]+_eval_record['not_eq_correct_count'][tau_idx])/float(_eval_record['eq_count'] + _eval_record['not_eq_count'])
		_eval_record['WKDR'][tau_idx, 2] = float(_eval_record['eq_correct_count'][tau_idx])/float( _eval_record['eq_count'])
		# print(_eval_record['eq_count'], _eval_record['not_eq_count'])
		_eval_record['WKDR'][tau_idx, 3] = float(_eval_record['not_eq_correct_count'][tau_idx])/float(_eval_record['not_eq_count'])

		if min(_eval_record['WKDR'][tau_idx,2], _eval_record['WKDR'][tau_idx, 3])>max_min:
			max_min = min(_eval_record['WKDR'][tau_idx,2], _eval_record['WKDR'][tau_idx, 3])
			max_min_i = tau_idx

	# print(_eval_record['WKDR'])
	print(_eval_record['WKDR'][max_min_i])
	# print("\tEvaluation Completed. Loss = {}, WKDR = {}".format(total_validation_loss, 1 - max_min))

	return float(total_validation_loss) / float(n_total_point_pair), 1 - max_min
