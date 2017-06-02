from __future__ import division

import numpy as np
import time
import torch
import re
from graphviz import Digraph
from torch.autograd import Variable


def minmax_scale_image(image):
    assert len(image.shape) in (2, 3)

    if len(image.shape) == 3:
        image_min = np.min(np.min(image, axis=0, keepdims=True), axis=1, keepdims=True)
        image_max = np.max(np.max(image, axis=0, keepdims=True), axis=1, keepdims=True)
        if image_max == image_min: return image - image_min
        return (image - image_min) / (image_max - image_min)
    else:
        image_min = np.min(image)
        image_max = np.max(image)
        if image_max == image_min: return image - image_min
        return (image - image_min) / (image_max - image_min)


__clock = [time.time()]
def tictoc(label, sync_cuda=True):
    if sync_cuda:
        torch.cuda.synchronize()
    new_time = time.time()
    print('{}\t:{}'.format(label, new_time - __clock[0]))
    __clock[0] = new_time


def save_dot(filename, *var):
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def add_nodes(var):
        if var not in seen:
            if isinstance(var, Variable):
                value = '('+(', ').join(['%d'% v for v in var.size()])+')'
                dot.node(str(id(var)), str(value), fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'previous_functions'):
                for u in var.previous_functions:
                    dot.edge(str(id(u[0])), str(id(var)))
                    add_nodes(u[0])
    for v in var:
        add_nodes(v.creator)
    dot.render(filename)


'''
Load a PFM file into a Numpy array. Note that it will have
a shape of H x W, not W x H. Returns a tuple containing the
loaded image and the scale factor from the file.
'''
# Source: https://gist.github.com/chpatrick/8935738
def load_pfm(fname):
    with open(fname, 'rb') as f:
        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = f.readline().rstrip()
        if header == b'PF':
            color = True    
        elif header == b'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(rb'^(\d+)\s(\d+)\s$', f.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(f.readline().rstrip())
        if scale < 0: # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>' # big-endian

        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        return np.reshape(data, shape) * abs(scale)
