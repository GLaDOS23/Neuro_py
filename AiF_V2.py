import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

class NeuralNetwork:
    def __init__(self, input_size, output_size, hidden_layers, hidden_neurons):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.weights = []
        self.biases = []
        self.activations = []
        self.initialize_weights_offsets(input_size, hidden_neurons, output_size, hidden_layers)
        
    def initialize_weights_offsets(self, input_size, hidden_neurons, output_size, hidden_layers):
        if hidden_layers > 0:
            self.weights.append(np.random.randn(input_size, hidden_neurons))
            self.biases.append(np.zeros((1, hidden_neurons)))
            for i in range(hidden_layers - 1):
                self.weights.append(np.random.randn(hidden_neurons, hidden_neurons))
                self.biases.append(np.zeros((1, hidden_neurons)))
            self.weights.append(np.random.randn(hidden_neurons, output_size))
            self.biases.append(np.zeros((1, output_size)))
        else:
            self.weights.append(np.random.randn(input_size, output_size))
            self.biases.append(np.zeros((1, output_size)))
    
    def sigmoid(self, x):#функция сигмойда
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):# Производная сигмойда
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def feedforward(self, X):# прямое распространение
        self.activations = []
        z = np.dot(X, self.weights[0]) + self.biases[0]# dot - скалярное произведение X и weights[0] + biases[0]
        a = self.sigmoid(z)#пропускаем через сигмойду получившееся число
        self.activations.append(a)#записываем результат в масив
        for i in range(1, len(self.weights)):#вычисления аналогичные предыдущим по всем весам
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)
            self.activations.append(a)
        return a
    
    def backpropagation(self, X, y, learning_rate):#обратное распространение
        # проходим прямое распространение
        self.feedforward(X)
        #Вычисляем ошибки и дельты для выходного слоя
        error = y - self.activations[-1]
        delta = error * self.sigmoid_derivative(self.activations[-1])
        # Обновление весов и смещений для выходного слоя
        self.weights[-1] += np.dot(self.activations[-2].T, delta) * learning_rate
        self.biases[-1] += np.sum(delta, axis=0, keepdims=True) * learning_rate
        #Вычисление ошибок и дельт для скрытых слоев
        for i in range(len(self.weights) - 2, -1, -1):
            error = np.dot(delta, self.weights[i + 1].T)
            delta = error * self.sigmoid_derivative(self.activations[i])
            # Обновление веса и смещения для скрытого слоя
            if i == 0:
                input_layer = X
            else:
                input_layer = self.activations[i - 1]
            self.weights[i] += np.dot(input_layer.T, delta) * learning_rate
            self.biases[i] += np.sum(delta, axis=0, keepdims=True) * learning_rate
    #запуск обучения известное количество раз (epochs)
    def train(self, X, y, epochs, learning_rate):
        for i in range(epochs):
            self.backpropagation(X, y, learning_rate)

# Функция для загрузки изображений и их классов
def load_images_from_folder(folder, txt_file):
    images = []
    labels = []
    with open(txt_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    for line in lines:
        filename, class_id = line.strip().split(':')
        class_id = int(class_id)
        image_path = os.path.join(folder, filename)
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flattened_image = [element for sub_list in gray_image for element in sub_list]
        images.append(flattened_image)
        labels.append(class_id)
    return np.array(images), np.array(labels)

# Функция для преобразования меток в one-hot encoding
def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]


folder = 'data'  # папка с jpg
txt_file = 'q.txt'  # txt 
images, labels = load_images_from_folder(folder, txt_file)

# Преобразование меток в one-hot encoding
num_classes = len(np.unique(labels))
y = one_hot_encode(labels, num_classes)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.2, random_state=42)

# Нормализация данных (приведение к диапазону [0, 1])
X_train = X_train / 255.0
X_test = X_test / 255.0

#параметры нейронной сети
input_size = X_train.shape[1]  
output_size = num_classes  
hidden_layers = 1  
hidden_neurons = 64  

nn = NeuralNetwork(input_size, output_size, hidden_layers, hidden_neurons)

#параметры обучения
epochs = 1000
learning_rate = 0.01

# Обучение модели
nn.train(X_train, y_train, epochs, learning_rate)

# Оценка модели на тестовой выборке
predictions = nn.feedforward(X_test)
predicted_classes = np.argmax(predictions, axis=1)
print(predicted_classes)
true_classes = np.argmax(y_test, axis=1)
accuracy = np.mean(predicted_classes == true_classes)
print(f"Точность на тестовой выборке: {accuracy * 100:.2f}%")
