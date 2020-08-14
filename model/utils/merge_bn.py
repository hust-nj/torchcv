import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

def fuse_module(m):
    def fuse_helper(cur, lastm, c, cn):
        children = list(cur.named_children())

        for name, child in children:
            if isinstance(child, nn.BatchNorm2d):
                bc = fuse_conv_bn_eval(c, child)
                lastm._modules[cn] = bc
                cur._modules[name] = nn.Identity()
                c = None
                cn = None
            elif isinstance(child, nn.Conv2d):
                c = child
                cn = name
                lastm = cur
            else:
                fuse_helper(child, lastm, c, cn)
    fuse_helper(m, None, None, None)

