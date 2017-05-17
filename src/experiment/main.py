import torch
import os
import sys
import argparse

print('test')

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', default='hourglass', help='model file definition')
    parser.add_argument('-bs',default=4, type = int, help='batch size')
    parser.add_argument('-it', default=0, type = int, help='Iterations')
    parser.add_argument('-lt', default=1000, type = int, help = 'Loss file saving refresh interval (seconds)')
    parser.add_argument('-mt', default=10000 , type = int, help = 'Model saving interval (iterations)')
    parser.add_argument('-et', default=3000 , type = int, help = 'Model evaluation interval (iterations)')
    parser.add_argument('-lr', default=1e-2 , type = float, help = 'Learning rate')
    parser.add_argument('-t_depth_file', default='', help = 'Training file for relative depth')
    parser.add_argument('-v_depth_file', default='' , help = 'Validation file for relative depth')
    parser.add_argument('-rundir', default='' , help = 'Running directory')
    parser.add_argument('-ep', default=10 , type = int , help = 'Epochs')
    parser.add_argument('-start_from', default='' , help = 'Start from previous model')
    parser.add_argument('-diw', default=False , type = bool , help = 'Is training on DIW dataset')
    g_args = parser.parse_args()
    return g_args

def default_feval(c):
    pass

def main():
    g_args = parseArgs()
    
    if g_args.diw:
        exec(open('./DataLoader_DIW.py').read())#TODO
        from validation_crit.validate_crit_DIW import *
    else:
        exec(open('./DataLoader.py').read())
        from validation_crit.validate_crit1 import *
    train_loader = TrainDataLoader()
    valid_loader = ValidDataLoader()

    if g_args.it == 0:
        g_args.it = g_args.ep * (train_loader.n_relative_depth_sample)/g_args.bs

    # Run path
    jobid = os.getenv('PBS_JOBID')
    job_name = os.getenv('PBS_JOBNAME')
    assert(job_name is not None)
    if g_args.rundir == '':
        if jobid == '':
            jobid = 'debug'
        else:
            jobid = jobid.split('%.')[0]
        g_args.rundir = os.path.join('/home/yifan/dump/depth_pytorch/results/',g_args.m, job_name)
    os.mkdir(g_args.rundir)
    torch.save(g_args ,g_args.rundir+'/g_args.t7')
    
    # Model
    config = {}
    # temporary solution
    if g_args.m == 'hourglass':
        from models.hourglass import *
    if g_args.start_from != '':
        # import cudnn
        print(g_args.rundir + g_args.start_from)
        g_model = torch.load(g_args.rundir + g_args.start_from)
        if g_model.period is None:
            g_model.period = 1
        g_model.period += 1
        config = g_model.config
    else:
        g_model = Model()
        g_model.period = 1
    # g_model.training()?
    config.learningRate = g_args.lr

    if get_criterion is None: #Todo
        print("Error: no criterion specified!!!!!!!")
        sys.exit(1)

    get_depth_from_model_output = f_depth_from_model_output()
    if get_depth_from_model_output is None:
        print('Error: get_depth_from_model_output is undefined!!!!!!!')
        sys.exit(1)

    g_criterion = get_criterion().cuda()
    g_model = g_model.cuda()
    # get parameters?




if __name__ == '__main__':
    main()