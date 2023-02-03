#统计模型运算量 torch.fx只支持静态控制流
from thop import profile
# flops, params = profile(net, (input, ))

import argparse
import os, sys
import math
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import pprint
import time
import torch
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import amp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np



import lib.dataset as dataset
from lib.config import cfg
from lib.config import update_config
from lib.core.loss import get_loss
from lib.core.function import train
from lib.core.function import validate
from lib.core.general import fitness
from lib.models import get_net
from lib.utils import is_parallel
from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger, select_device
from lib.utils import run_anchor

# pip install pthflops
from pthflops import count_ops

def main():
    # Create a network and a corresponding input
    device = 'cuda:0'
    model = get_net(cfg).to(device).half()
    model.eval()
    inp = torch.rand(1,3,1152,1920).to(device).half()
    # flops_224 = count_ops(model, inp) #1.83gflops

    
    with torch.no_grad():
        flops, params = profile(model,(inp,))
        print('flops:{}'.format(flops))

main()