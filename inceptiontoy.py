import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

batch_size=64

train_dataset=datasets.MNIST(root='./data/',
                             train=True,
                             transforms=transforms.ToTensor(),
                             download=True)
test_dataset = datasets.MNIST(root='./mnist_data/',
                              train=False,
                              transform=transforms.ToTensor())
train_loader=torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_loader=torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True
)
class InceptionBlock(nn.Module):
    def __init__(self,in_channels):
        super(InceptionBlock, self).__init__()
        self.branch1x1=nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch5x5_1=nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch5x5_2=nn.Conv2d(16,24,kernel_size=5,padding=2)
        self.branch3x3dbl_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3dbl_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)


    def forward(self, x):
        branch1x1=self.branch1x1(x)
        branch5x5=self.branch5x5_1(x)
        branch5x5=self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool=F.avg_pool2d(x,kernel_size=3,stride=1,padding=1)
        branch_pool = self.branch_pool(branch_pool)


        outputs=[branch1x1,branch5x5,branch3x3dbl,branch_pool]

        return torch.cat(outputs,1)

class Inception(nn.Module):
     def __init__(self):
         super(Inception, self).__init__()
         self.conv1=nn.Conv2d(1,10,kernel_size=5)
         self.conv2=nn.Conv2d(88,20,kernel_size=5)

         self.incept1=InceptionBlock(in_channels=10)
         self.incept2=InceptionBlock(in_channels=20)
         self.mp=nn.MaxPool2d(2)
         self.fc=nn.Linear(1408,10)

     def forward(self, x):
         in_size=x.size(0)
         x = F.relu(self.mp(self.conv1(x)))
         x = self.incept1(x)
         x = F.relu(self.mp(self.conv2(x)))
         x = self.incept2(x)
         x = x.view(in_size, -1)  # flatten the tensor
         x = self.fc(x)
         return F.log_softmax(x)

model = Inception()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))



def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))




for epoch in range(1, 10):
    train(epoch)
    test()
