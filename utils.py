import os
import torch
from torch.autograd import Variable

def make_folder(root_path, version,file_path):
    if not os.path.exists(os.path.join(root_path,version,file_path)):
        os.makedirs(os.path.join(root_path,version,file_path))

def tensor2var(x, grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=grad)

def var2tensor(x):
    return x.data.cpu()

def var2numpy(x):
    return x.data.cpu().numpy()

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)
