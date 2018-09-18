import torch
import random
from torch.autograd import Variable
import torch.nn.modules
class TwoLayerNet(torch.nn.modules):
    def __init__(self,D_in,H,D_out):
        super(TwoLayerNet,self).__init__()
        self.linear1=torch.nn.Linear(D_in,H)
        self.linear2=torch.nn.Linear(H,D_out)
    def forward(self,x):
        h_relu=self.linear1(x).clamp(min=0)
        y_pred=self.linear2(h_relu)  #don nedd operator
        return y_pred
class DynamicNet(torch.nn.modules):
    def __init__(self, D_in, H, D_out):
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)
    def forward(self,x):
        h_relu=self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0,3)):
            h_relu=self.middle_linear(h_relu).clamp(min=0)
        y_pred=self.output_linear(h_relu)
        return y_pred
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建随机张量来保存输入和输出,并将它们包装在变量中.
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

# 通过实例化上面定义的类来构建我们的模型
model = TwoLayerNet(D_in, H, D_out)

# 构建我们的损失函数和优化器.
# 对SGD构造函数中的model.parameters()的调用将包含作为模型成员的两个nn.Linear模块的可学习参数.
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(500):
    y_pred=model(x)
    loss=criterion(y_pred,y)
    print(t, loss.data[0])
    # 梯度置零, 执行反向传递并更新权重.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()