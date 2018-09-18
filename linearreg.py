import torch
from torch.autograd import Variable

x_data=Variable(torch.Tensor([[1.0],[2.0],[3.0]]))
y_data=Variable(torch.Tensor([[2.0],[4.0],[6.0]]))  #3x1 matrix

class LinearModel(torch.nn.Module):
    def __index__(self):
        super(LinearModel, self).__init__()
        self.linear=torch.nn.Linear(1,1)   #w,no need to know batch size
    def forward(self, x):
        y_pred=self.linear(x)
        return y_pred

Lmodel=LinearModel()
criterion=torch.nn.MSELoss(size_average=False)
optimizer=torch.optim.SGD(Lmodel.parameters(),lr=0.01)


for epoch in range(500):
    y_pred=Lmodel(x_data)
    loss=criterion(y_pred,y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



hour_var = Variable(torch.Tensor([[4.0]]))
y_pred = Lmodel(hour_var)
print("predict (after training)",  4, Lmodel(hour_var).data[0][0])