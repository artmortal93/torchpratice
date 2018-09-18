import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]


cell=nn.RNN(input_size=4,hidden_size=2,batch_first=True)

hidden=Variable(torch.randn(1,1,2)) #numoflayerxdirection + batchnum + hidden(output size)

inputs=Variable(torch.Tensor([h,e,l,l,o]))  #not char h!!
for one in inputs:
    one=one.view(1,1,-1)  #batch x seqlen x input_size(one-hot length)(may be word/character)
    out, hidden = cell(one, hidden)
    print("one input size", one.size(), "out size", out.size())

inputs=inputs.view(1,5,-1)  #batch x seqlen x input_size(one-hot length)
out,hidden=cell(inputs,hidden)
print("sequence input size", inputs.size(), "out size", out.size())

hidden = Variable(torch.randn(1, 3, 2))  #numoflayerxdirection + batchnum + hidden(output size)

inputs = Variable(torch.Tensor([[h, e, l, l, o],
                                [e, o, l, l, l],
                                [l, l, e, e, l]]))
out, hidden = cell(inputs, hidden)  #(3,5,4)
print("batch input size", inputs.size(), "out size", out.size())



cell = nn.RNN(input_size=4, hidden_size=2)
inputs = inputs.transpose(dim0=0, dim1=1)
# Propagate input through RNN
# Input: (seq_len, batch_size, input_size) when batch_first=False (default)
# S x B x I
out, hidden = cell(inputs, hidden)
print("batch input size", inputs.size(), "out size", out.size())