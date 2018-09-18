import torch
import torch.nn.functional as F
from torch.autograd import Variable

x_data=Variable(torch.Tensor([[1.0],[2.0],[3.0],[4.0]]))
y_data=Variable(torch.Tensor([[0.],[0.],[1.],[1.]]))
class logModel(torch.nn.Module):
    def __init__(self):
        super(logModel, self).__init__()
        self.linear=torch.nn.Linear(1,1)


    def forward(self, x):
        y_pred=F.sigmoid(self.linear(x))
        return y_pred



lModel=logModel()
criterion=torch.nn.BCELoss(size_average=True)
optimizer=torch.optim.SGD(lModel.parameters(),lr=0.01)


for epoch in range(1000):
    y_pred=lModel(x_data)
    loss=criterion(y_pred,y_data)
    if epoch%100==0:
        print(epoch,loss.data[0])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


hour_var=Variable(torch.Tensor([[1.0]]))
print(lModel(hour_var).data[0][0]>0.5)
