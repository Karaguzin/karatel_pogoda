import numpy as np
from neuron import SingleNeuron

# Загрузка весов из файла и тестирование
new_neuron = SingleNeuron(input_size=3)
new_neuron.load_weights('neuron_weights.txt')

# Пример использования
test_data1 = np.array([[25, 70, 3]])  # Температура, Влажность, Скорость ветра
predictions1 = new_neuron.forward(test_data1)
print("Предсказанные значения для тестовых данных 1:", predictions1, *np.where(predictions1 >= 0.5, 'Хорошая погода', 'Плохая погода'))

test_data2 = np.array([[15, 85, 7]])  # Температура, Влажность, Скорость ветра
predictions2 = new_neuron.forward(test_data2)
print("Предсказанные значения для тестовых данных 2:", predictions2, *np.where(predictions2 >= 0.5, 'Хорошая погода', 'Плохая погода'))
