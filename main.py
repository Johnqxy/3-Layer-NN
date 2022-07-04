import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
import math
from model import Net
data = torch.ones(1000, 1)
x00 = torch.normal(60 * data, 1)  # male 60kg
x01 = torch.normal(1.72 * data, 0.05)  # male 1.72m
x0 = torch.cat((x00, x01), 1)  # 1000*2
# print(x0)
y0 = torch.zeros(1000, 1)
x10 = torch.normal(50 * data, 1)  # female 50kg
x11 = torch.normal(1.65 * data, 0.05)  # female 1.65m
x1 = torch.cat((x10, x11), 1)  # 1000*2
# print(x1)
y1 = torch.ones(1000, 1)
x = torch.cat((x0, x1), 0)
y = torch.cat((y0, y1), 0)
plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=10)
plt.savefig('data')  # save visual data
model = Net()
loss = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(50000):
    x_data = Variable(x)
    y_data = Variable(y)
    out = model(x_data)
    epoch_loss=loss(out,y_data)
    loss_value=epoch_loss.item()
    mask = out.ge(0.5)
    correct = (mask == y_data).sum()
    acc = correct.item() / x_data.size(0)
    optimizer.zero_grad()
    epoch_loss.backward()
    optimizer.step()
    if (epoch + 1) % 1000 == 0:
        print('-------------------')
        print('epoch:', epoch + 1, 'loss:', loss_value, 'acc:', acc)
w11,w12=model.hidden.weight[0].detach().numpy()
w21,w22=model.hidden.weight[1].detach().numpy()
w31,w32=model.hidden.weight[2].detach().numpy()
w0,w1,w2=model.out.weight[0].detach().numpy()
b11,b12,b13=model.hidden.bias.detach().numpy()
b=model.out.bias.item()
while 1:
    print('please input weight,height')
    x = float(input())
    y = float(input())
    l1=w11*x+w12*y+b11
    l2 = w21 * x + w22 * y + b12
    l3 = w31 * x + w32 * y + b13
    res=w0*l1+w1*l2+w2*l3+b
    value=1/(1+math.exp(-res))
    print(value)
    if value < 0.5:
        print('male')
    else:
        print('female')
