import pickle
import numpy as np
from flask import Flask, render_template, url_for, request, jsonify
from model.neuron import SingleNeuron

app = Flask(__name__)

menu = [{"name": "Лаба 4", "url": "p_lab4"}]

# Загрузка весов из файла
new_neuron = SingleNeuron(input_size=3)  # Обновлено для 3 входов
new_neuron.load_weights('model/neuron_weights.txt')


@app.route("/")
def index():
    return render_template('index.html', title="Лабораторные работы, выполненные ФИО", menu=menu)


@app.route("/p_lab4", methods=['POST', 'GET'])
def p_lab4():
    if request.method == 'GET':
        return render_template('lab4.html', title="Определение погоды", menu=menu, class_model='')

    if request.method == 'POST':
        X_new = np.array([[float(request.form['temperature']),  # Температура
                           float(request.form['humidity']),  # Влажность
                           float(request.form['wind_speed'])]])  # Скорость ветра
        predictions = new_neuron.forward(X_new)
        print("Предсказанные значения:", predictions, *np.where(predictions >= 0.5, 'Хорошая погода', 'Плохая погода'))
        return render_template('lab4.html', title="Определение погоды", menu=menu,
                               class_model="Прогноз: " + str(
                                   *np.where(predictions >= 0.5, 'Хорошая погода', 'Плохая погода')))


if __name__ == "__main__":
    app.run(debug=True)
