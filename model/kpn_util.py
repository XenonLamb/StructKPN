import torch
from torch import nn
import math, re, functools
import torch.nn.functional as F
import numpy as np  

class LocalConv2d_No(nn.Module):
    
    def __init__(self, in_channels=1, out_channels=1,
                 kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super(LocalConv2d_No, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (self.kernel_size -1) // 2
        self.dilation = dilation
        self.bias = bias

    def forward(self, input, w_gen):
        '''
        Local Convolution

        inputs:
            input: N, Ci, H, W
            w_gen: N, Co*(Ci*k*k + 1), H, W
        returns
            out: N, Co, H, W
        '''
        n, c, h, w = input.shape
        if self.kernel_size == 1:
            input_cat = input.view(n, self.kernel_size ** 2, c, h, w).contiguous()  # N, kk, Cin, H, W
            input_cat = input_cat.permute(0, 2, 1, 3, 4).contiguous().view(n, 1, c, -1, h, w).contiguous()  #N, Cin, kk, H, W --> N,1,Cin,kk,H,W
            if self.bias == True:

                cout = w_gen.shape[1] // (c*self.kernel_size**2 +1) #Co*(Cin*kk +1)/(Cin*kk) = Co
                w_gen = w_gen.view(n, cout, -1, h, w).contiguous()  #N,Co, Cin*kk+1, H, W
                b_gen = w_gen[:, :, -1, :, :]  #
                w_gen = w_gen[:, :, :-1, :, :].view(n, cout, c, -1, h, w).contiguous() # N,Co, Cin, kk, H, W
                out = (((input_cat * w_gen).sum(3).sum(2)) + b_gen).contiguous()
            else:
                cout = w_gen.shape[1] // (c*self.kernel_size**2) #Co*(Cin*kk +1)/(Cin*kk) = Co
                w_gen = w_gen.view(n, cout, -1, h, w).contiguous()  #N,Co, Cin*kk, H, W
                w_gen = w_gen[:, :, :, :, :].view(n, cout, c, -1, h, w).contiguous() # N,Co, Cin, kk, H, W
                out = (((input_cat * w_gen).sum(3).sum(2))).contiguous()
        else:
            input_allpad = F.pad(input, [self.padding, self.padding, self.padding, self.padding], mode='reflect').contiguous()
            # Roll out the local patches at each pixel position
            input_im2cl_list = []
            for i in range(self.kernel_size):
                for j in range(self.kernel_size):
                    input_im2cl_list.append(input_allpad[:, :, i:(h+i), j:(w+j)].contiguous())
            input_cat = torch.cat(input_im2cl_list, 1)
            input_cat = input_cat.view(n, self.kernel_size ** 2, c, h, w).contiguous()  # N, kk, Cin, H, W
            input_cat = input_cat.permute(0, 2, 1, 3, 4).contiguous().view(n, 1, c, -1, h, w).contiguous()  #N, Cin, kk, H, W --> N,1,Cin,kk,H,
          
            if self.bias == True:
                cout = w_gen.shape[1] // (c*self.kernel_size**2 +1) #Co*(Cin*kk +1)/(Cin*kk) = Co
                w_gen = w_gen.view(n, cout, -1, h, w).contiguous()  #N,Co, Cin*kk+1, H, W
                b_gen = w_gen[:, :, -1, :, :]  #
                w_gen = w_gen[:, :, :-1, :, :].view(n, cout, c, -1, h, w).contiguous() # N,Co, Cin, kk, H, W
                out = (((input_cat * w_gen).sum(3).sum(2)) + b_gen).contiguous()
            else:
                cout = w_gen.shape[1] // (c*self.kernel_size**2) #Co*(Cin*kk +1)/(Cin*kk) = Co
                w_gen = w_gen.view(n, cout, -1, h, w).contiguous()  #N,Co, Cin*kk, H, W
                w_gen = w_gen[:, :, :, :, :].view(n, cout, c, -1, h, w).contiguous() # N,Co, Cin, kk, H, W
                out = (((input_cat * w_gen).sum(3).sum(2))).contiguous()

        return out
