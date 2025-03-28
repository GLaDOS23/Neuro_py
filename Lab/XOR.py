import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.layer1 = nn.Linear(2, 2)  # входной слой 2 входа, 2 скрытых нейрона
        self.layer2 = nn.Linear(2, 1)  # выходной слой 2 скрытых нейрона, 1 выход
        self.sigmoid = nn.Sigmoid()    # функция активации
        
    def forward(self, x):
        x = self.sigmoid(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x


model = XORModel()
criterion = nn.MSELoss()  #среднеквадратичная ошибка
optimizer = optim.SGD(model.parameters(), lr=0.1)


epochs = 10000#количество эпох
target_accuracy = 0.95  #точность 100% - 5%

#обучение модель может правильно обучиться не с первого раза, нужно просто презапустить код и он скажет если обучится правильно
for epoch in range(epochs):
    # прямой проход
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # обратный проход 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Проверка точности
    with torch.no_grad():
        predicted = (outputs > 0.5).float()
        accuracy = (predicted == y).float().mean()
        #Если модель достигла заданной точности до завершения всех эпох, то обучение останавливается, чтобы не было переобучени
        if accuracy >= target_accuracy:
            print(f'Обучение завершено на эпохе {epoch+1}')
            print(f'Достигнута точность: {accuracy.item()*100:.2f}%')
            break

    if (epoch + 1) % 1000 == 0:#для проверки обучения
        print(f'№{epoch+1}, ошибка: {loss.item():.4f}, точность: {accuracy.item()*100:.2f}%')


with torch.no_grad():# Окончательная проверка
    print("\nПроверка:")
    for i in range(len(X)):
        input_val = X[i]
        output_val = model(input_val)
        predicted = (output_val > 0.5).float()
        print(f"Вход: {input_val.tolist()} -> выход: {output_val.item():.4f} результат: {predicted.item()}")
