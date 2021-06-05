# TODO: FIX THIS OLD IMPORTED CODE TO WORK WITH THE NEW CODE

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
from numpy.fft import fft, ifft
import time as systemClock
from tqdm import tqdm
import tensorflow as tf
import keras
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import keras.backend as K
from keras.optimizers import Adam, SGD, RMSprop
from tensorflow.python.client import device_lib
from datetime import datetime

compression_size = 2756

absolute_path = '/home/lemonorange/catRemix/'
attempt = 'attempt29'
data_path = os.path.join(absolute_path, 'LSTMmodels', attempt, 'att1.h5')
loss_data_path = os.path.join(absolute_path, 'LSTMmodels', attempt, 'loss_time_graph.txt')
fin = open(loss_data_path)
loss_data = list(map(np.float32, fin.readline().strip('\n').strip('[').strip(']').split(', ')))
loss_data1 = list(map(np.float32, fin.readline().strip('\n').strip('[').strip(']').split(', ')))
loss_data2 = list(map(np.float32, fin.readline().strip('\n').strip('[').strip(']').split(', ')))
loss_data3 = list(map(np.float32, fin.readline().strip('\n').strip('[').strip(']').split(', ')))

x = np.arange(len(loss_data))
zero = np.zeros(x.shape[0])
plt.plot(x, loss_data, color='blue', label='Train BINCE')
plt.plot(x, loss_data1, color='green', label='Train CUS')
plt.plot(x, loss_data2, color='orange', label='Val BINCE')
plt.plot(x, loss_data3, color='magenta', label='Val CUS')
plt.plot(x, zero, color='red', label='optimal loss')
plt.legend()
plt.show()

def read_input(path):
    fin = open(path)
    # assuming above command runs successfully
    data = np.array(list(map(int, fin.readline().split())))
    fin.close()
    return data

def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    max_arr = max(arr)
    min_arr = min(arr)
    diff_arr = max_arr - min_arr
    for i in arr:
        temp = (((i - min_arr)*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return np.array(norm_arr)

model = keras.models.load_model(data_path)
input_path = os.path.join(absolute_path, 'LSTMreadyData', 'input')
# input_file_name = os.listdir(input_path)[5]
input_file_name = '772-0-0.rawWav'
input_file_path = os.path.join(input_path, input_file_name)
input_data = read_input(input_file_path)
input_data = input_data[:220480]
input_data = normalize(input_data,-10,10)
input_data = tf.reshape(input_data, (int(input_data.shape[0]/compression_size), 1, compression_size))

gen_out = model(input_data)
sample = gen_out[0]
sample = np.reshape(sample, [-1])
plt.plot(np.arange(sample.shape[0]), sample, color='green')
sample = np.reshape(sample, (88))

gen_out = gen_out.numpy()
threshold = 0
gen_out.shape

noteToFreq = lambda note : np.float32(440 * 2 ** ((note-69)/12))

freqOut = []
for sample in gen_out:
    sample = np.reshape(sample, [-1])
    sampleFreqOut = []
    for i in range(sample.shape[0]):
        if(sample[i] > threshold and i > 10 and i < 100):
            sampleFreqOut.append(i)
    freqOut.append(np.array(sampleFreqOut))
freqOut = np.array(freqOut)

sampleRate = 44100
length = np.float32(compression_size/sampleRate)
t = np.linspace(0, length, int(sampleRate * length))  #  Produces a 5 second Audio-File
final = []
for sample in freqOut:
    if(sample.shape[0] == 0):
        final.append(np.sin(0*t))
        continue
    overall = np.sin(noteToFreq(sample[0]+21) * 2 * np.pi * t)
    for freq in sample[1:]:
        overall += np.sin(noteToFreq(freq+21) * 2 * np.pi * t)
    final.append(overall)

gen = np.concatenate(final)

input_data = tf.reshape(input_data, [-1])

plt.plot(np.arange(input_data.shape[0]), input_data, color='blue')

plt.plot(np.arange(gen.shape[0]), gen, color='red')

Audio(gen, rate=sampleRate)
Audio(input_data, rate=sampleRate)
