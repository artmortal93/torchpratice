import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import sys

torch.manual_seed(777)  # reproducibility
#            0    1    2    3    4
idx2char = ['h', 'i', 'e', 'l', 'o']



x_data = [0, 1, 0, 2, 3, 3]   # hihell
one_hot_lookup = [[1, 0, 0, 0, 0],  # 0
                  [0, 1, 0, 0, 0],  # 1
                  [0, 0, 1, 0, 0],  # 2
                  [0, 0, 0, 1, 0],  # 3
                  [0, 0, 0, 0, 1]]

y_data = [1, 0, 2, 3, 3, 4]# ihello
x_one_hot = [one_hot_lookup[x] for x in x_data]

inputs = Variable(torch.Tensor(x_one_hot))
labels = Variable(torch.LongTensor(y_data))
num_classes = 5
input_size = 5  # one-hot size
hidden_size = 5  # output from the RNN. 5 to directly predict one-hot
batch_size = 1   # one sentence
sequence_length = 1  # One by one
num_layers = 1

class SimpleRnn(torch.nn.Module):
    def __init__(self):
        super(SimpleRnn, self).__init__()
        self.rnn=nn.RNN(input_size=input_size,hidden_size=hidden_size,batch_first=True)
    def forward(self, hidden,x):
        x=x.view(batch_size,sequence_length,input_size) #input sizxe
        out,hidden=self.rnn(x,hidden)#numoflayerxdirection + batchnum + hidden(output size)
        return hidden,out.view(-1,num_classes)
    def init_hidden(self):
        return Variable(torch.zeros(num_layers,batch_size,hidden_size))

model=SimpleRnn()
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


for epoch in range(100):
    optimizer.zero_grad()
    loss=0
    hidden=model.init_hidden()
    sys.stdout.write("predicted string:")
    for input,label in zip(inputs,labels):
        hidden,output=model(hidden,input)
        val,idx=output.max(1)  #val in most occuation  are not one-hot vector
        sys.stdout.write(idx2char[idx.data[0]])
        loss+=criterion(output,label)

    print(", epoch: %d, loss: %1.3f" % (epoch + 1, loss.data[0]))
    loss.backward()
    optimizer.step()

print("learning finished")
#hihell->ihello