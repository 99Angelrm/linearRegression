import torch
import torch.nn as nn
import matplotlib.pyplot as plt
#TODO CHANGE THIS TO YOUR MODEL
from NeuralNetwork import NeuralNetwork

model = NeuralNetwork()

model.load_state_dict(torch.load(".pth"))
model.eval()

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y_true = 2 * x + 1 + 0.2 * torch.rand(x.size())

with torch.no_grad():
    y_pred = model(x)

x_np = x.numpy()
y_true_np = y_true.numpy()
y_pred_np = y_pred.numpy()

plt.figure()
plt.scatter(x_np, y_true_np, label="True data", alpha=0.6)
plt.plot(x.numpy(), y_pred.numpy(), color='red', label='Fitted Line')
plt.legend()
plt.title("Linear Model Approximation")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()

#Weight and Bias from training process
#TODO UPDATE THIS WITH YOUR MODEL'S WEIGHTS AND BIASES
# w = 
# b = 

# yNN = w * x + b

# plt.figure()
# plt.scatter(x_np, y_true_np, label="True data")
# plt.plot(x.numpy(), yNN.numpy(), color='green', label='NN Line')
# plt.legend()
# plt.title("Neural Network Linear Approximation")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.grid()

plt.show()  