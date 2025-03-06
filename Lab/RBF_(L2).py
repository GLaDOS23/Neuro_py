import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


x = torch.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)
y = torch.sin(x)


class RBFNet(nn.Module):
    def __init__(self, num_centers, sigma=1.0):
        super(RBFNet, self).__init__()
        self.num_centers = num_centers
        self.sigma = sigma
        self.centers = nn.Parameter(torch.rand(num_centers, 1) * 2 * np.pi)  # Центры
        self.linear = nn.Linear(num_centers, 1)  # линейный слой

    def rbf(self, x):
        # вычисление радиальных базисных функций
        return torch.exp(-((x - self.centers.T) ** 2) / (2 * self.sigma ** 2))

    def forward(self, x):
        # применение RBF и линейного слоя
        rbf_output = self.rbf(x)
        return self.linear(rbf_output)


num_centers = 5#количество центров
model = RBFNet(num_centers=num_centers, sigma=0.5)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# обучение
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


model.eval()
with torch.no_grad():
    predicted = model(x).numpy()

plt.plot(x.numpy(), y.numpy(), label='Sin')
plt.plot(x.numpy(), predicted, label='Predict Sin')
plt.legend()
plt.show()



# центры
model.eval()
with torch.no_grad():
    rbf_output = model.rbf(x).numpy()

plt.figure(figsize=(10, 6))
for i in range(num_centers):
    plt.plot(x.numpy(), rbf_output[:, i], label=f'{i+1}')
plt.title('')
plt.legend()
plt.show()
