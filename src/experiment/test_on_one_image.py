from PIL import Image
import torch
import argparse
from torchvision import transforms

def parseArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument('-prev_model_file',required=True, help = 'model file definition')
	parser.add_argument('-input_image', required=True, help = 'path to the input image')
	parser.add_argument('-output_image', required=True, help = 'path ot the output image')
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
	orig_width, orig_height = orig_img.size

	print('Processing sample ', thumb_filename)

	loader = transforms.Compose([
		transforms.Scale((_network_input_width, _network_input_height)),
		transforms.ToTensor(),
		# transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
		])
	img = loader(orig_img).float()
	# print(img)
	_batch_input_cpu[0,:,:,:] = (img)

	_processed_input = torch.autograd.Variable(_batch_input_cpu.cuda())
	batch_output = (model(_processed_input)).float()

	a = batch_output[0,:,120,:]
	# b = torch.autograd.Variable(torch.zeros(a.size()[0],a.size()[1]+1)).cuda()
	# b[0,0:-1]=b[0,0:-1]+a
	# b[0,1:]=b[0,1:]-a
	# print(a)

	# print(batch_output>0)

	t_back = transforms.Compose([
		transforms.ToPILImage(),
		transforms.Scale((orig_width, orig_height))
		])
	orig_size_output = batch_output.data[0].cpu()
	# print(orig_size_output[0,0])
	orig_size_output = orig_size_output - torch.min(orig_size_output)
	orig_size_output = orig_size_output / torch.max(orig_size_output)
	orig_size_output = t_back(orig_size_output)#.convert('RGB')

	# orig_size_output.save(cmd_params.output_image)
	new_image = Image.new('RGB', (orig_width*2, orig_height))
	new_image.paste(orig_img, (0,0))
	new_image.paste(orig_size_output, (orig_width, 0))
	# print(new_image)
	new_image.save(cmd_params.output_image)

if __name__ == '__main__':
	main()