from model import *
from helper import *
import torch

device = torch.device(f"cuda:1" if torch.cuda.is_available() else "cpu")

modf = relKKT_general(mode = 'linf')
tar = '/home/lxyang/git/pdqpnet/pkl/8906_valid/QPLIB_8906_17.mps.pkl'
sol_check(tar,device,modf)