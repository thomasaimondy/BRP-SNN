import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal

thresh = 0.5
lens = 0.5
decay = 0.2
if_bias = True

class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()

    # @staticmethod
    # def backward(ctx, grad_h):
    #     z = ctx.saved_tensors
    #     s = torch.sigmoid(z[0])
    #     d_input = (1 - s) * s * grad_h
    #     return d_input

act_fun = ActFun.apply

def mem_update(ops, x, mem, spike, lateral = None):
    mem = mem * decay * (1. - spike) + ops(x)
    if lateral:
        mem += lateral(spike)
    spike = act_fun(mem)
    return mem, spike

class SNN(nn.Module):
    def __init__(self, batch_size, input_size, num_classes):
        super(SNN, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = 500
        self.fc1 = nn.Linear(self.input_size, self.hidden_size, bias = if_bias)
        self.fc2 = nn.Linear(self.hidden_size, self.num_classes, bias = if_bias)

    def forward(self, input, task, time_window):
        h1_mem = h1_spike = h1_sumspike = torch.zeros(self.batch_size, self.hidden_size).cuda()
        h2_mem = h2_spike = h2_sumspike = torch.zeros(self.batch_size, self.num_classes).cuda()

        for step in range(time_window):
            if task == 'mnist':
                x = input > torch.rand(input.size()).cuda()
            elif task == 'nettalk':
                x = input.cuda()
            elif task == 'gesture':
                x = input[:, step, :]
            x = x.float()
            x = x.view(self.batch_size, -1)
            h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike)
            h1_sumspike += h1_spike
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike)
            h2_sumspike += h2_spike

        outputs = h2_sumspike / time_window
        return outputs