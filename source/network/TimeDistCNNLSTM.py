from termcolor import colored
print(colored('Loading libraries...', attrs=['bold']))
print(colored('Loading tensorflow...', attrs=['bold']))
import tensorflow as tf
print(colored('Loading keras...', attrs=['bold']))
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, Bidirectional, TimeDistributed, Conv1D, MaxPooling1D, Flatten, InputLayer, Input
import os
import sys
import math
from keras.optimizers import Adam, SGD, RMSprop, Adagrad
print(colored('Loading numpy...', attrs=['bold']))
import numpy as np
sys.path.append('./source/network')
print(colored('Loading datasync...', attrs=['bold']))
from Dataloader import datasync as sync
from Dataloader import loader
print(colored('Loading math stuff...', attrs=['bold']))
import matplotlib.pyplot as plt
from functools import reduce
from time import sleep
import time
import librosa

from tensorflow.python.client import device_lib

# load in testing data
absolute_path = os.path.join('/home/lemonorange/catRemixV2')
data_root_path = os.path.join(absolute_path, 'data')
input_path = os.path.join(data_root_path, 'wav')
label_path = os.path.join(data_root_path, 'rawMid')
ground_truth_data_path = os.path.join(data_root_path, 'rawMid')

data, sr = loader.parse_input('7480_6.wav', input_path, norm=False) # NORMALIZATION MUST BE DISABLED
label, bpm = loader.parse_label('7480_6.wav', label_path)

data2, sr2 = loader.parse_input('1_0.wav', input_path, norm=False) # NORMALIZATION MUST BE DISABLED
label2, bpm2 = loader.parse_label('1_0.wav', label_path)

ML = loader.get_mel_spec(data, 512, sr, 4096, 2048)
ML2 = loader.get_mel_spec(data2, 128, sr2, 2048, 512)

ML.shape

plt.plot([sum(i) for i in ML])
plt.plot(np.concatenate([[sum(i)]*28 for i in sync.trim_front(label)]), color='green') # this is the midi sound wave (sorta)
plt.imshow(ML2)

ML.shape
len(sync.trim_front(label))
420/14

data.shape
ML.shape

data.size

math.ceil(data.size/512)

# one skips one by one at an interval of 512, and one at 700. They won't sync perfectly
# if the midi_size covers 700 samples in sr, and the hop length of Mel is 512
# we will take 2x 512 slices, which adds up to 1024 samples in total matched to the note
# the math is as the follwing:
# length < ceil(midi_size / hop_length)
# start_position < floor(note_location / hop_length)

librosa.display.specshow(ML.transpose(), sr=sr, hop_length=512, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB');

label = sync.trim_front(label)

sync_in, sync_la = sync.sync_data(ML, label, bpm, 2048)
sync_in.shape

112/14

sync_in2, sync_la2 = sync.sync_data(ML2, label2, bpm2, 512)

np.concatenate(sync_in).shape
plt.plot([sum(i) for i in np.concatenate(sync_in)])
plt.plot(np.concatenate([[sum(i)]*8 for i in sync_la]), color='green') # this is the midi sound

sync_in2.shape
np.array(loader.encode_multihot(sync_la2)).shape

for i in sync_in:
    print(i.shape)

np.array(loader.encode_multihot(sync_la)).shape
np.array(loader.encode_multihot(sync_la2)).shape

sync_in.shape
sync_in2.shape
# >>> (16, 7, 128) <- there are 16 notes, with 7 columns in each, and 128 bins per column
len(sync_la)
# >>> 16

plt.imshow(sync_in[0])

# NOTE: The amount of note is NOT the temporal dimension!!! THE AMOUNT OF COLUMN IS!!!
# we pass in the notes one after another, not in a temporal way!

sync_in[-1].shape

# create model
model = Sequential()

model.add(TimeDistributed(Conv1D(64, 3, activation='relu'), batch_input_shape=(None, None, ML.shape[1], 1))) # min-batch size, window size, mel res, channel
model.add(TimeDistributed(MaxPooling1D(2, strides=1)))

model.add(TimeDistributed(Conv1D(128, 4, activation='relu')))
model.add(TimeDistributed(MaxPooling1D(2, strides=2)))

model.add(TimeDistributed(Conv1D(256, 4, activation='relu')))
model.add(TimeDistributed(MaxPooling1D(2, strides=2)))
#
# # extract features and dropout
model.add(TimeDistributed(Flatten()))
model.add(Dropout(0.5))

# this creates a time distributed input to the LSTM
# input to LSTM
model.add(
    Bidirectional(
        LSTM(256, return_sequences=False, dropout=0.5),
        merge_mode='sum'
    )
)

# classifier with sigmoid activation for multilabel
model.add(Dense(88, activation='sigmoid'))

# compile model
model.compile()

# print out architecture
print(colored('Compilation successful; Architecture summary:', 'green'))
model.summary()

# model.save(os.path.join(absolute_path, 'time_dist_model.h5'))

with tf.GradientTape() as tape: # Start calculating the gradient and applying it
    t1 = np.reshape(sync_in, (sync_in.shape[0], sync_in.shape[1], sync_in.shape[2], 1))
    t2 = tf.reshape(sync_in2, (sync_in2.shape[0], sync_in2.shape[1], sync_in2.shape[2], 1))

    # t1.shape
    # t2.shape

    p1 = model(t1)
    p2 = model(t2)

    p1 = tf.concat([p1], 0)
    print(p1.numpy())

    # sync_la = loader.encode_multihot(sync_la)
    # sync_la2 = loader.encode_multihot(sync_la2)

    # sync_la = tf.convert_to_tensor(sync_la)

    loss = tf.losses.binary_crossentropy(sync_la, p1)

    default_opt = Adam(learning_rate=1e-3)
    grad = tape.gradient(loss, model.trainable_variables)
    print(grad)
    default_opt.apply_gradients(zip(grad, model.trainable_variables))

pc = np.concatenate([p1, p2])

k = loader.encode_multihot(sync_la)
np.array(k).shape

tf.losses.binary_crossentropy(k, model(t1)).numpy().shape
