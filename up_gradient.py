import torch
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0   #random value

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)
def gradient(x, y):  # d_loss/d_w
    return 2 * x * (x * w - y)
print("predict (before training)",  4, forward(4))




for epoch in range(15):
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val)
        w = w - 0.01 * grad
        print("\tgrad: ", x_val, y_val, round(grad, 2))
        l = loss(x_val, y_val)

    print("progress:", epoch, "w=", round(w, 2), "loss=", round(l, 2))

print("predict (after training)",  "4 hours gets predict score", round(forward(4),2))




