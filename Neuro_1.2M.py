import numpy as np
import time
class NeuralNetwork:
    def __init__(self, input_size, output_size, hidden_layers, hidden_neurons):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.weights = []
        self.biases = []
        self.activations = []
        
    def initialize_weights_biases(self, input_size, hidden_neurons, output_size, hidden_layers):# заполнение весов и смещений
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

    def saving_weights_biases(self):#сохраняем веса и смещения в фаил 
        fw = open("weights.txt", "w")
        for i in range(hidden_layers+1):
            if i == 0:
                for i2 in range(input_size):
                    fw.write(" ".join([str(k) for k in self.weights[i].tolist()[i2]])+"\n")
                fw.write("!")
            if i > 0 and i < hidden_layers:
                for i3 in range(hidden_neurons):
                   fw.write(" ".join([str(k) for k in self.weights[i].tolist()[ i3 ]])+"\n")
                fw.write("!")
                
            if i == hidden_layers:
                for i4 in range(hidden_neurons ):                   
                    fw.write(" ".join([str(k) for k in self.weights[i].tolist()[i4]])+"\n")
                fw.write("!")    
        fw.close()
        fb = open("biases.txt", "w")
        for i in range(hidden_layers+1):
            fb.write(" ".join([str(k) for k in self.biases[i].tolist()[0]])+"\n")
        fb.close()
    def reading_from_a_file(self):#Читаем веса исмещения из файла
        f = open("weights.txt", "r")
        text = f.read().split("\n")
        f.close()
        mul =[]
        i = 0
        for line in text:
            arr = []
            for  num in line.split("!"):
                for  num1 in num.split(" "):
                    if num1 != "":
                        arr.append(float(num1))
                    else:
                        if i == 0:
                            out = np.array(mul)
                            i = 1
                        else :
                            if i == 1:
                                out =[out , np.array(mul)]
                                i =2
                            else:
                                out = [*out , np.array(mul)]
                        mul = []
            mul.append(arr)
        out = [*out , np.array(mul)]
        self.weights = out.copy()
        
        f = open("biases.txt", "r")
        text = f.read().split("\n")
        f.close()
        mul =[]
        i = 0
        for line in text:
            arra = []
            for h in line.split(" "):
                arra.append(float(h))
            if i == 0:
                out2 = np.array([arra])
                i = 1

            else :
                if i == 1:
                    out2 =[out2 , np.array([arra])]
                    i =2
                else:
                    out2 = [*out2 , np.array([arra])] 
        self.biases = out2.copy()




        
    def sigmoid(self, x):#функция сигмойда
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):#роизводная сигмойда
        return x * (1 - x)
    
    def feedforward(self, X):# прямое распространение
        self.activations = []
        z = np.dot(X, self.weights[0]) + self.biases[0] # dot - скалярное произведение X и weights[0] + biases[0]
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
        
        # вычисляем ошибки и дельты для выходного слоя
        error = y - self.activations[-1]
        #print(self.activations[-1])
        delta = error * self.sigmoid_derivative(self.activations[-1])
        
        # обновление веса и смещения для выходного слоя
        self.weights[-1] += np.dot(self.activations[-2].T, delta) * learning_rate
        self.biases[-1] += np.sum(delta, axis=0, keepdims=True) * learning_rate
        
        # вычисление ошибок и дельт для скрытых слоев
        for i in range(len(self.weights) - 2, -1, -1):
            error = np.dot(delta, self.weights[i + 1].T)
            delta = error * self.sigmoid_derivative(self.activations[i])
            
            # обновление веса и смещения для скрытого слоя
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






X = np.array([[1,1,0,1,1]])
y = np.array([[0,0,0,1,1,1]])
X2 = np.array([[1,1,0,1,1]])
#X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#y = np.array([[0], [1], [1],[0]])
#X2 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    
input_size=5
output_size=6
hidden_layers=3
hidden_neurons=10
epochs=10000
learning_rate=0.5
nn = NeuralNetwork(input_size, output_size, hidden_layers, hidden_neurons)
# инициализируйте веса и смещения для каждого слоя

nn.initialize_weights_biases(input_size, hidden_neurons, output_size, hidden_layers)


#nn.reading_from_a_file()

seconds1 = time.time()
nn.train(X, y, epochs, learning_rate)
seconds2 = time.time()
print("time: ",seconds2 - seconds1)
#print(nn.biases)
#print("/////////////////////////////////////////////////////////////////////")
#print(nn.weights)
#print("/////////////////////////////////////////////////////////////////////")
out = nn.feedforward(X2)


#nn.saving_weights_biases()

print(out)

'''
Настройка нейронки
input_size - количество входов
output_size - количество нейронов на выходном слое
hidden_layers - количество внутренних слоёв
hidden_neurons - количество нейронов на внутреннем слое
epochs - количество циклов обучения
learning_rate - скорость обучения


'''

