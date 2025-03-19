import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Генерация данных
x = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)
y =np.sin(x)#предсказываем функцию синуса
'''
Старков сказал, что обычний синус - это скучно, поэтому можно использовать
более сложную функцию, например эту: x**2 * np.sin(x).
Но тогда лучше ствить epochs = 2000.
'''
#y = x**2 * np.sin(x)


# Модель 
class RBFNet(tf.keras.Model):
    def __init__(self, num_centers, sigma=1.0):
        super(RBFNet, self).__init__()
        self.num_centers = num_centers
        self.sigma = sigma
        self.centers = tf.Variable(tf.random.uniform((num_centers, 1), minval=0, maxval=2 * np.pi), trainable=True)
        # тензор с центрами радиальных базисных функций, щт 0 до 2pi
        
        self.linear = tf.keras.layers.Dense(1)  # линейный слой с одним нейроном и без функции активации

    def rbf(self, x):
        # вычисление рбф
        return tf.exp(-tf.square(x - tf.transpose(self.centers)) / (2 * self.sigma**2))

    def call(self, x):
        rbf_output = self.rbf(x)
        return self.linear(rbf_output)

# параметры модели
num_centers = 20  # количество центров
model = RBFNet(num_centers=num_centers, sigma=0.5)# сгма - влияет на ширину рбф
criterion = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)#Скорость обучения

# Обучение
epochs = 500#количество эпох
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        outputs = model(x)
        loss = criterion(y, outputs)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.numpy():.4f}')

# прямой проход и результаты
predicted = model(x).numpy()

plt.plot(x, y, label='Sin')
plt.plot(x, predicted, label='Predict Sin')
plt.legend()
plt.show()

# график с центрами рбф
rbf_output = model.rbf(x).numpy()
plt.figure(figsize=(10, 6))
for i in range(num_centers):
    plt.plot(x, rbf_output[:, i], label=f'Center {i+1}')
plt.title('Center')
plt.legend()
plt.show()
