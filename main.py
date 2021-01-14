from flask import Flask, request

import numpy as np
import pandas as pd

from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop, Adam

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pickle

import heapq

app = Flask(__name__)

model = load_model('model/keras_next_word_model.h5')
history = pickle.load(open("model/history.p", "rb"))
unique_ingredient_index = pickle.load(open("model/unique_ingridients.p", "rb"))
index_unique_ingredient = {index: value for value, index in unique_ingredient_index.items()}


def prepare_input(input_ingredients):
    x = np.zeros((1, len(input_ingredients), len(unique_ingredient_index)))
    for t, ingredient in enumerate(input_ingredients):
        x[0, t, unique_ingredient_index[ingredient]] = 1
    return x


def suggest_next_ingredient(input_ingredients, n=5):
    x = prepare_input(input_ingredients)
    preds = model.predict(x, verbose=0)[0]
    next_indices = heapq.nlargest(n, range(len(preds)), preds.take)
    return [index_unique_ingredient[idx] for idx in next_indices]


@app.route('/testing', methods=['GET'])
def hello_world():
    return 'Hello World!'


@app.route('/prediction_endpoint', methods=['GET','POST'])
def prediction_endpoint():

    if request.method == 'POST':
        return 'Hello World!'
    elif request.method == 'GET':
        ingridients = list(request.values.get('ingridients').split(','))
        prediction = suggest_next_ingredient(ingridients)
        return prediction[0]


if __name__ == '__main__':
    app.run()
