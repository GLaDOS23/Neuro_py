import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


def load_images_from_folder(folder, txt_file):
    images_dict = {}
    with open(txt_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    for line in lines:
        filename, class_id = line.strip().split(':')
        class_id = int(class_id)
        image_path = os.path.join(folder, filename)
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flattened_image = [element for sub_list in gray_image for element in sub_list]
        if class_id not in images_dict:
            images_dict[class_id] = []
        images_dict[class_id].append(flattened_image)
    images_array = []
    for class_id in sorted(images_dict.keys()):
        images_array.append(np.array(images_dict[class_id]))
    return np.array(images_array)


class ImageDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# Структура RNN
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)#Скрытый рекуррентный слой
        self.fc = nn.Linear(hidden_size, output_size)# Полносвязный слой на выходе

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.rnn.hidden_size) # Инициализация скрытого состояния
        out, _ = self.rnn(x, h0)#Пропускаем данные через нейросеть и получаем результат
        out = self.fc(out[:, -1, :])  # Используем только последний выход RNN
        return out



folder = 'data'
txt_file = 'q.txt'
images_array = load_images_from_folder(folder, txt_file)


X = np.concatenate(images_array)
y = np.concatenate([np.full(len(images_array[i]), i) for i in range(len(images_array))])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


train_dataset = ImageDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


input_size = X_train.shape[1]  # Размер входных данных (длина одномерного массива)
hidden_size = 128  # Размер скрытого слоя RNN
output_size = len(images_array)  # Количество классов
model = SimpleRNN(input_size, hidden_size, output_size)
num_epochs = 10

criterion = nn.CrossEntropyLoss()#Функции потерь
optimizer = optim.Adam(model.parameters(), lr=0.001)# Оптимизатор

model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs, labels

        # Добавляем размерность последовательности
        inputs = inputs.unsqueeze(1) 

        optimizer.zero_grad()#обнуление градиентов
        outputs = model(inputs)# Предсказания модели
        loss = criterion(outputs, labels)#функция потерь
        loss.backward()# вычисление градиентов
        optimizer.step()#Обновление параметров

        running_loss += loss.item()

        #print(f'№ {epoch+1}, ошибка: {running_loss/len(train_loader):.4f}')


print("Ошибка обучения: ", running_loss/len(train_loader))                                                                                                                                  
image = cv2.imread("S5.jpg")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
flattened_image = [element for sub_list in gray_image for element in sub_list]
image_tensor = torch.tensor(flattened_image , dtype=torch.float32).unsqueeze(0).unsqueeze(0)
model.eval()#режим оценки

with torch.no_grad():
    output = model(image_tensor)
    _, predicted_class = torch.max(output, 1)

print(predicted_class.item())
