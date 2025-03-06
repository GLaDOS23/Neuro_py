import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


x = torch.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)
y = torch.sin(x)

#модель
class SineNet(nn.Module):
    def __init__(self):
        super(SineNet, self).__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# инициализация модели, функции потерь и оптимизатора
model = SineNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

#обучение 
epochs = 500
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
