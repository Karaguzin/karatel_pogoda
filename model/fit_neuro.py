import numpy as np
from neuron import SingleNeuron

# Пример данных (X - входные данные, y - целевые значения)
# Здесь [температура, влажность, скорость ветра]
X = np.array([[25, 65, 3],
              [30, 70, 5],
              [22, 80, 2],
              [28, 60, 4],
              [26, 75, 1],
              [24, 85, 3],
              [27, 68, 2]])
# Пример целевого значения (0 - плохая погода, 1 - хорошая погода)
y = np.array([1, 1, 0, 1, 1, 0, 1])

# Инициализация и обучение нейрона
neuron = SingleNeuron(input_size=3)
neuron.train(X, y, epochs=5000, learning_rate=0.001)

# Сохранение весов в файл
neuron.save_weights('neuron_weights.txt')
