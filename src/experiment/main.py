import torch
import os
import sys
import argparse

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', default='hourglass3', help='model file definition')
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

def main():
    g_args = parseArgs()
    # print(args.bs)
    



if __name__ == '__main__':
    main()