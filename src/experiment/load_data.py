import sys
train_depth_path = None
valid_depth_path = None

folderpath = '../../data/'

if g_args.t_depth_file != '':
	train_depth_path = folderpath + g_args.t_depth_file

if g_args.v_depth_file != '':
	valid_depth_path = folderpath + g_args.v_depth_file


if train_depth_path is None:
	print("Error: Missing training file for depth!")
	sys.exit(1)

if valid_depth_path is None:
	print("Error: Missing validation file for depth!")
	sys.exit(1)


def TrainDataLoader():
	return DataLoader(train_depth_path)

def ValidDataLoader():
	return DataLoader(valid_depth_path)