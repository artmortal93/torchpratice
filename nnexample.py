import torch
import torch.nn
from torch.autograd import Variable
#higher abstraction
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建随机张量来保存输入和输出,并将它们包装在变量中.
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)



model=torch.nn.Sequential(
    torch.nn.Linear(D_in,H),
    torch.nn.ReLU(),
    torch.nn.Linear(H,D_out)  #neeed two dim
)


loss_fn=torch.nn.MSELoss(size_average=False)

learning_rate=1e-4
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
for t in range(500):
    y_pred=model(x)
    loss=loss_fn(y_pred,y)
    print(t,loss.data[0])
    optimizer.zero_grad()#model.zero_grad
    loss.backward()
    #for param in model.parameters():
        #param.data-=learning_rate*param.grad.data
    optimizer.step()
    


