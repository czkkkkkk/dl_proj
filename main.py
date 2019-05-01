import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout
from keras.layers import TimeDistributed
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.optimizers import RMSprop
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import sys
import heapq
import seaborn as sns
from pylab import rcParams
import csv
import argparse
import sys
import re

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 12, 5

csv.field_size_limit(sys.maxsize)

SEQUENCE_LENGTH = 40
step = 3

def read_csv(input) :
    ret = []
    with open(input, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if(line_count != 0):
                ret.append(' '.join(re.sub(r"[^a-zA-Z0-9]", " ", row[9].lower()).split()))
            line_count += 1
    print('Number of row is ' + str(line_count))
    return ret

def train(input, model_path) :
    text = ' '.join(input)
    print('corpus length:', len(text))

    chars = sorted(list(set(text)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    print('unique chars: ' + str(len(chars)))
    with open('100_char', 'wb') as fp:
        pickle.dump(chars, fp)

    sentences = []
    next_chars = []
    for i in range(0, len(text) - SEQUENCE_LENGTH, step):
        sentences.append(text[i: i + SEQUENCE_LENGTH])
        next_chars.append(text[i + SEQUENCE_LENGTH])
    print('num training examples: ' + str(len(sentences)))

    X = np.zeros((len(sentences), SEQUENCE_LENGTH, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    print(X.shape)
    print(y.shape)

    model = Sequential()
    model.add(LSTM(128, input_shape=(SEQUENCE_LENGTH, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    history = model.fit(X, y, validation_split=0.05, batch_size=128, epochs=20, shuffle=True).history

    model.save(model_path)
    pickle.dump(history, open("history.p", "wb"))

    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left');
    plt.savefig('fig.png')

def prepare_input(text, char_indices, chars):
    x = np.zeros((1, SEQUENCE_LENGTH, len(chars)))
    for t, char in enumerate(text):
        x[0, t, char_indices[char]] = 1.
        
    return x

def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    
    return heapq.nlargest(top_n, range(len(preds)), preds.take)

def predict_completion(text):
    original_text = text
    generated = text
    completion = ''
    while True:
        x = prepare_input(text)
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, top_n=1)[0]
        next_char = indices_char[next_index]
        text = text[1:] + next_char
        completion += next_char
        
        if len(original_text + completion) + 2 > len(original_text) and next_char == ' ':
            return completion

def predict_completions(text, n=3):
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [indices_char[idx] + predict_completion(text[1:] + indices_char[idx]) for idx in next_indices]

def predict(input, model_path, train_step):
    print("Input size " + str(len(input)))
    with open('100_char', 'rb') as fp:
        chars = pickle.load(fp)
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    model = load_model(model_path)

    X = np.zeros((train_step, SEQUENCE_LENGTH, len(chars)), dtype=np.bool)
    y = np.zeros((train_step, len(chars)), dtype=np.bool)
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    acc = 0
    cnt = 0
    for i in range(0, len(input) - SEQUENCE_LENGTH):
        text = input[i: i + SEQUENCE_LENGTH]
        x = prepare_input(text, char_indices, chars)
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, top_n=1)[0]
        next_char = indices_char[next_index]
        if(next_char == input[i + SEQUENCE_LENGTH]) :
            acc += 1
        cnt += 1

        for t in range(0, SEQUENCE_LENGTH):
            X[i % train_step, t, char_indices[input[i + t]]] = 1
        y[i % train_step, char_indices[input[i + SEQUENCE_LENGTH]]] = 1
        if(i > 0 and i % train_step == 0):
            model.fit(X, y, batch_size=128, epochs=4, shuffle=True, verbose=1)
            X = np.zeros((train_step, SEQUENCE_LENGTH, len(chars)), dtype=np.bool)
            y = np.zeros((train_step, len(chars)), dtype=np.bool)
            print("acc " + str(acc) + " cnt " + str(cnt) + " : " + str(1. * acc / cnt))
    if(cnt > 0):
        print("Accuracy in test dataset " + str(1. * acc / cnt))
    else:
        print("Dataset too small")
     


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Word Prediction ')
    parser.add_argument('type', type=str, choices=['train', 'predict'],
                        help='Type: train, predict')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='input file')
    parser.add_argument('--model', '-m', type=str, required=True, help='Model path')
    parser.add_argument('--train_step', '-t', type=int, help='Train step in prediction', default=500)
    args = parser.parse_args()

    input = read_csv(args.input)
    if(args.type == 'train'):
        train(input, args.model)
    else :
        predict(input[0], args.model, args.train_step)
