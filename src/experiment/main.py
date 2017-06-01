import torch
import torch.optim as optim
import shutil
import os
import sys
import argparse
import time

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', default='hourglass', help='model file definition')
    parser.add_argument('-bs',default=4, type = int, help='batch size')
    parser.add_argument('-it', default=0, type = int, help='Iterations')
    parser.add_argument('-lt', default=10, type = int, help = 'Loss file saving refresh interval (seconds)')
    parser.add_argument('-mt', default=3000 , type = int, help = 'Model saving interval (iterations)')
    parser.add_argument('-et', default=1000 , type = int, help = 'Model evaluation interval (iterations)')
    parser.add_argument('-lr', default=1e-3 , type = float, help = 'Learning rate')
    parser.add_argument('-t_depth_file', default='', help = 'Training file for relative depth')
    parser.add_argument('-v_depth_file', default='' , help = 'Validation file for relative depth')
    parser.add_argument('-rundir', default='' , help = 'Running directory')
    parser.add_argument('-ep', default=10 , type = int , help = 'Epochs')
    parser.add_argument('-start_from', default='' , help = 'Start from previous model')
    parser.add_argument('-diw', default=False , type = bool , help = 'Is training on DIW dataset')
    parser.add_argument('-optim', default='RMSprop', help = 'choose the optimizer')
    g_args = parser.parse_args()
    return g_args

def default_feval():
    batch_input, batch_target = train_loader.load_next_batch(g_args.bs)

    optimizer.zero_grad()

    batch_output = g_model(batch_input)
    batch_loss = g_criterion.forward(batch_output, batch_target)
    batch_loss.backward()
    # dloss_dx = g_criterion.backward(1.0)
    # print(dloss_dx)
    # batch_output.backward(dloss_dx.data)
    optimizer.step()

    return batch_loss.data[0]

def save_loss_accuracy(t_loss, t_WKDR, v_loss, v_WKDR):
    _v_loss_tensor = torch.Tensor(v_loss)
    _t_loss_tensor = torch.Tensor(t_loss)
    _v_WKDR_tensor = torch.Tensor(v_WKDR)
    _t_WKDR_tensor = torch.Tensor(t_WKDR)#It doesn't matter whether t_WKDR is a Tensor

    _full_filename = os.path.join(g_args.rundir, 'loss_accuracy_record_period' + str(g_model.period) + '.h5')
    if os.path.isfile(_full_filename):
        os.remove(_full_filename)

    myFile = h5py.File(_full_filename, 'w')
    myFile.create_dataset('t_loss', data=_t_loss_tensor.numpy())
    myFile.create_dataset('v_loss', data=_v_loss_tensor.numpy())
    myFile.create_dataset('t_WKDR', data=_t_WKDR_tensor.numpy())
    myFile.create_dataset('v_WKDR', data=_v_WKDR_tensor.numpy())
    myFile.close()

def save_model(model, directory, current_iter, config):
    # model.clearState()
    model.config = config
    torch.save(model, directory+'/model_period'+str(model.period)+'_'+str(current_iter)+'.pt')

def save_best_model(model, directory, config, iteration):
    # model.clearState()
    model.config = config
    model.iter = iteration
    torch.save(model, os.path.join(directory,'Best_model_period'+str(model.period)+'.pt'))


## main
g_args = parseArgs()

if g_args.diw:
    exec(open('./DataLoader_DIW.py').read())#TODO
    from validation_crit.validate_crit_DIW import *
else:
    exec(open('./DataLoader.py').read())
    from validation_crit.validate_crit1 import *
exec(open('load_data.py').read())
train_loader = TrainDataLoader()
valid_loader = ValidDataLoader()

if g_args.it == 0:
    g_args.it = int(g_args.ep * (train_loader.n_relative_depth_sample)/g_args.bs)

# Run path
jobid = os.getenv('PBS_JOBID')
job_name = os.getenv('PBS_JOBNAME')
# assert(job_name is not None)
if g_args.rundir == '':
    if jobid == '' or jobid is None:
        jobid = 'debug'
    else:
        jobid = jobid.split('%.')[0]
    g_args.rundir = os.path.join('/home/yifan/dump/depth_pytorch/results/',g_args.m, str(job_name))
# if os.path.exists(g_args.rundir):
#     shutil.rmtree(g_args.rundir)
if not os.path.exists(g_args.rundir):
    os.mkdir(g_args.rundir)
torch.save(g_args ,g_args.rundir+'/g_args.pt')

# Model
config = {}
# temporary solution
if g_args.m == 'hourglass':
    from models.hourglass import *
if g_args.start_from != '':
    # import cudnn
    print(os.path.join(g_args.rundir, g_args.start_from))
    g_model = torch.load(os.path.join(g_args.rundir , g_args.start_from))
    if g_model.period is None:
        g_model.period = 1
    g_model.period += 1
    config = g_model.config
else:
    g_model = Model()
    g_model.period = 1
# g_model.training()?
config['learningRate'] = g_args.lr

if get_criterion is None: #Todo
    print("Error: no criterion specified!!!!!!!")
    sys.exit(1)

get_depth_from_model_output = f_depth_from_model_output()
if get_depth_from_model_output is None:
    print('Error: get_depth_from_model_output is undefined!!!!!!!')
    sys.exit(1)

g_criterion = get_criterion()
g_model = g_model.cuda()
# g_params = g_model.parameters() # get parameters
if g_args.optim == 'RMSprop':
    optimizer = optim.RMSprop(g_model.parameters(), lr=g_args.lr) #optimizer
    print('Using RMSprop')
elif g_args.optim == 'Adam':
    optimizer = optim.Adam(g_model.parameters(), lr=g_args.lr)
    print('Using Adam')

feval = default_feval
best_valist_set_error_rate = 1.0
train_loss = []
train_WKDR = []
valid_loss = []
valid_WKDR = []
lfile = open(g_args.rundir+'/training_loss_period'+str(g_model.period)+'.txt', 'w')

total_loss = 0.0
for i in range(0,g_args.it):
    # start = time.time()
    running_loss = feval()
    total_loss += running_loss
    # end = time.time()
    print(('loss = {}'.format(running_loss)))
    lfile.write('loss = {}\n'.format(running_loss))
    # print('time_used = {}'.format(end-start))

    if i % g_args.mt == 0 and i!=0:
        print('Saving model at iteration {}...'.format(i))
        save_model(g_model, g_args.rundir, i, config)

    if i % g_args.et == 0:
        print('Evaluatng at iteration {}'.format(i))
        train_eval_loss, train_eval_WKDR = evaluate(train_loader, g_model, g_criterion, 100) #TODO
        valid_eval_loss, valid_eval_WKDR = evaluate(valid_loader, g_model, g_criterion, 100)
        print("train_eval_loss:",train_eval_loss, "; train_eval_WKDR:" ,train_eval_WKDR)
        print("valid_eval_loss:", valid_eval_loss, "; valid_eval_WKDR:", valid_eval_WKDR)

        train_loss.append(running_loss)
        valid_loss.append(valid_eval_loss)
        train_WKDR.append(train_eval_WKDR)
        valid_WKDR.append(valid_eval_WKDR)

        save_loss_accuracy(train_loss, train_WKDR, valid_loss, valid_WKDR)

        if best_valist_set_error_rate > valid_eval_WKDR:
            best_valist_set_error_rate = valid_eval_WKDR
            save_best_model(g_model, g_args.rundir, config, i)