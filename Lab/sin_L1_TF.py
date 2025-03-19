import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Генерация данных (синусойда )
x = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)
y = np.sin(x)

#Модель
class SineNet(tf.keras.Model):
    def __init__(self):
        super(SineNet, self).__init__()
        self.fc1 = tf.keras.layers.Dense(32, activation='relu')# полносвязный слой №1 32 нейрона relu - функция активации
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')# полносвязный слой №2 64 нейрона relu - функция активации
        self.fc3 = tf.keras.layers.Dense(1)# полносвязный слой №3 1 нейрон функции активации нет

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# инициализация модели
model = SineNet()
criterion = tf.keras.losses.MeanSquaredError()#инициализация функции потерь
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)#инициализация оптимизатора learning_rate - скорость обучения(можно менять)

# Обучение
epochs = 500# Количество эпох
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        outputs = model(x)
        loss = criterion(y, outputs)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.numpy():.4f}')

# Прямой проход и вывод результатов
predicted = model(x).numpy()

plt.plot(x, y, label='Sin')
plt.plot(x, predicted, label='Predict Sin')
plt.legend()
plt.show()
