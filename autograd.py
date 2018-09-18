import torch
from torch.autograd import Variable

class MyRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input







dtype=torch.FloatTensor

N,D_in,H,D_out=64,1000,100,10

x=Variable(torch.rand(N,D_in).type(dtype),requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype),requires_grad=False)

# 随机初始化权重
w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

learning_rate = 1e-6

for i in range(500):
    h=torch.mm(x,w1)#x.mm only applies on variable and tensor
    h_relu=torch.clamp(h,min=0)
    y_pred=torch.mm(h_relu,w2)
    loss = (y_pred - y).pow(2).sum()
    print(i, loss.data[0])


    loss.backward()
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data
    # 更新权重后手动将梯度归零
    w1.grad.data.zero_()
    w2.grad.data.zero_()

