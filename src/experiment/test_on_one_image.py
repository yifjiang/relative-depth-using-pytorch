from PIL import Image
import torch
import argparse
from torchvision import transforms

def parseArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument('-prev_model_file',required=True, help = 'model file definition')
	parser.add_argument('-input_image', required=True, 'path to the input image')
	parser.add_argument('-output_image', required=True, 'path ot the output image')
	args = parser.parse_args()
	return args

def main():
	cmd_params = parseArgs()
	_network_input_width = 320
	_network_input_height = 240

	prev_model_file = cmd_params.prev_model_file
	model = torch.load(prev_model_file)

	print('Model file:', prev_model_file)

	_batch_input_cpu = torch.Tensor(1,3,_network_input_height, _network_input_width)

	thumb_filename = cmd_params.input_image
	orig_img = Image.open(thumb_filename).convert('RGB')
	orig_height, orig_width = orig_img.size

	print('Processing sample ', thumb_filename)

	loader = transforms.Compose(
		transforms.Scale((_network_input_width, _network_input_height)),
		transforms.ToTensor()
		)
	img = loader(orig_img).float()
	_batch_input_cpu[0,:,:,:].copy_(img)

	_processed_input = _batch_input_cpu.cuda()
	batch_output = model(_processed_input)

	t_back = transforms.Compose(
		transforms.ToPILImage(),
		transforms.Scale((orig_width, orig_height))
		)
	orig_size_output = batch_output.data[0]
	orig_size_output = t_back(orig_size_output)

	orig_size_output.save(cmd_params.output_image)