from termcolor import colored
# TODO: FIX THIS OLD IMPORTED CODE TO WORK WITH THE NEW CODE

print(colored('Loading libraries...', attrs=['bold']))
import os
import math
print(colored('Loading numpy...', attrs=['bold']))
import numpy as np
print(colored('Loading matplotlib...', attrs=['bold']))
import matplotlib.pyplot as plt
from IPython.display import Audio
from numpy.fft import fft, ifft
import time as systemClock
print(colored('Loading tqdm...', attrs=['bold']))
from tqdm import tqdm
print(colored('Loading datasync...', attrs=['bold']))
import sys
sys.path.append('./source/network')
from Dataloader import datasync as sync
from Dataloader import loader
print(colored('Loading keras...', attrs=['bold']))
import keras
print(colored('Loading tensorflow...', attrs=['bold']))
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import keras.backend as K
from keras.optimizers import Adam, SGD, RMSprop
from tensorflow.python.client import device_lib
print(colored('All libraries loaded', attrs=['bold']))
from datetime import datetime
import difflib
import librosa

sample_rate = 44000
mel_res = 512
window_size = 4096
hop_len = 512

absolute_path = os.path.join('/home/lemonorange/catRemixV2')
data_root_path = os.path.join(absolute_path, 'data')
input_path = os.path.join(data_root_path, 'wav')
label_data_path = os.path.join(data_root_path, 'rawMid')
storage_path = os.path.join(absolute_path, 'network')

# Change your training session name here. So for example if you had a session named "cyc1", you put that here.
cycle_path = 'cycA5'

# Select your network preference here. So for example if you want to test the snapshot "snp_1_200.h5", you put that here.
network_path = os.path.join(storage_path, cycle_path, 'snp_1_0.h5')

# Read in training and validation loss files
train_loss_path = os.path.join(storage_path, cycle_path, 'train_loss.txt')
val_loss_path = os.path.join(storage_path, cycle_path, 'val_loss.txt')

train_loss_file = open(train_loss_path)
val_loss_file = open(val_loss_path)

# parse in losses
loss_names = [i.split('=')[0] for i in train_loss_file.readline().split(';')][:-1]
loss_colors = ['red', 'green', 'blue', 'magenta', 'yellow', 'black', 'cyan']

train_loss_file.seek(0)
train_loss = []
for entry in train_loss_file:
    train_loss.append(np.array(
        [i.split('=')[1] for i in entry.split(';')[:-1] if i.split('=')[0]]
    ))
train_loss = np.array(train_loss)

val_loss = []
for entry in val_loss_file:
    val_loss.append(np.array(
        [i.split('=')[1] for i in entry.split(';')[:-1]]
    ))
val_loss = np.array(val_loss)

# Plot all the losses
x = np.arange(train_loss.shape[0])
zero = np.zeros(x.shape[0])

figure, axis = plt.subplots(1,2) # create subplot

for i in range(len(loss_names)): # plot the actual data
    if(True):
        axis[0].plot(x, train_loss[:,i].astype(np.float32), color=loss_colors[i], label=loss_names[i])

axis[0].plot(x, zero, color='black', label='optimal loss')
axis[0].legend()
axis[0].set_title('Training Loss Over Time (400 mini-batches)')
axis[0].set_xlabel('Mini-Batch')
axis[0].set_ylabel('Loss Value')

for i in range(len(loss_names)): # plot the actual data
    if(True):
        axis[1].plot(x, val_loss[:,i].astype(np.float32), color=loss_colors[i], label=loss_names[i])
axis[1].plot(x, zero, color='black', label='optimal loss')
axis[1].legend()
axis[1].set_title('Validation Loss Over Time (400 mini-batches)')
axis[1].set_xlabel('Mini-Batch')
axis[1].set_ylabel('Loss Value')

plt.tight_layout()
plt.show()

# after this part, you should be able to see what your training loss progression is like

# only disable this when running in terminal and you only want to see the graph
# quit()
# ============================================== DATA LOAD IN ==============================================

model = keras.models.load_model(network_path)
all_files = os.listdir(input_path)

file_index = 10
that_file = all_files[file_index]
that_file = "rushe.wav"

that_file

data, sr = loader.parse_input(that_file, input_path, norm=False)
data2, sr = loader.parse_input("dancebg.wav", input_path, norm=False)

Audio(data, rate=sample_rate) # preview the audio

file = that_file
# file = 'testmusic2_0.wav' # extract a testing file

unpaired_input, sr = loader.parse_input(file, input_path) # parse input
unpaired_label, bpm = loader.parse_label(file, label_data_path) # trimming the MIDI and syncing the data
unpaired_label = sync.trim_front(unpaired_label)

ML = loader.get_mel_spec(unpaired_input, mel_res, sr, window_size, hop_len)
input = loader.get_mel_spec(unpaired_input, mel_res, sr, window_size, hop_len)

# input, label = sync.sync_data(ML, unpaired_label, bpm, hop_len) # pair IO + trim
label = np.array(loader.encode_multihot(unpaired_label)) # encode label

chunk_size = int(ML.shape[0] / label.shape[0])-5
chunk_size = 10
chunk_size

input = loader.create_input_matrix(input, chunk_size, chunk_size)

input = np.reshape(input, (input.shape[0], input.shape[1], input.shape[2], 1))
input.shape

# ============================================== PREDICTION ==============================================

g = systemClock.time()
output = model.predict(input)
h = systemClock.time()

np.max(output)
np.min(output)

h-g
# output = model.predict(input*3)
note_to_freq = lambda note : np.float32(440 * 2 ** ((note-69)/12))
note_to_freq_with_offset = lambda note : np.float32(440 * 2 ** (((note+21)-69)/12))

# parse the multi-hot encoding into raw numbers
threshold = 0.5
# threshold = 0.9

note_out = []
for point in output: # read in all data points
    point_note_out = [] # for every one hot encoding, save the significant notes
    for i in range(point.shape[0]):
        if(point[i] > threshold):
            point_note_out.append(i)
    if(len(point_note_out) == 0): point_note_out.append(0)
    note_out.append(np.array(point_note_out))

# get a multi-hot encoding of that
encoded_note_out = loader.encode_multihot(note_out)

sample_rate = 44000
length = np.float32((hop_len * chunk_size)/sample_rate)

t = np.linspace(0, length, int(sample_rate * length))  #  Produces a sample lengthed note

final = []
x = 0
# loop through all note vectors and synthesize them into sine waves
for i in range(len(note_out)):
    # silence is represented by an all zero vector!
    if(np.all(encoded_note_out[i] == np.zeros(88))): # the case that it is silent
        x += 1
        final.append(np.sin(0*t)) # append a silence to the final sequence
        continue

    # if it is not a silence, create the fourier transform
    if (note_out[i][0]<87 and note_out[i][0]>5):
        overall = np.cos(note_to_freq_with_offset(note_out[i][0]) * 2 * np.pi * t)
    else:
        overall = np.sin(1 * 2 * np.pi * t) # take full cycle of 2pi radians and scale them by scaler T
    for freq in note_out[i][1:]:
        if (freq<80 and freq>5):
            overall += np.cos(note_to_freq_with_offset(freq) * 2 * np.pi * t)
        else:
            continue
    final.append(np.hanning(overall.shape[0]) * overall)
    # final.append(overall)

gen = np.concatenate(final)

len(note_out)

# Plot out the waveforms
data.size / 25
6285.16/44000

librosa.display.specshow(ML.transpose(), sr=sr, hop_length=hop_len, x_axis='time', y_axis='log') # this is the original sound wave
plt.plot([sum(i) for i in note_out], color='green') # this is the midi sound wave (sorta)
# plt.plot(loader.normalize(gen[4000000:5000000],3), color='red')
# plt.plot(flattend_input, color='blue', label='raw sound data')


# plt.legend()
# plt.title('MIDI transcription result by BLSTM')

# plt.show()
# this is blueshift by exurb1a
Audio(unpaired_input, rate=sample_rate) # this is the original sounds random I know but bear with me
Audio(gen, rate=sample_rate) # this is transcribed by an AI in less than 200 miliseconds
Audio(gen[:-1] + data2[:gen.shape[0]-1], rate=sample_rate) # this is them combined transcirbed + original

# CAT PITCH SHIFT ALGORITHM
# First, smooth out the waveforms. override notes that last for 0.125ms with the previous note
smoothed_note_out = note_out.copy()
for i in range(len(smoothed_note_out)): # fill in 0s
    if(np.all(smoothed_note_out[i] == 0)): # if this note is silent and there was stuff trailing the current note, extend out the last note
        smoothed_note_out[i] = smoothed_note_out[i-1]

# for i in range(len(smoothed_note_out)-1):
#     if(difflib.SequenceMatcher(None,smoothed_note_out[i],smoothed_note_out[i+1]).ratio()>0.9):
#         smoothed_note_out[i+1] = smoothed_note_out[i]

smoothed_encoded_note_out = loader.encode_multihot(smoothed_note_out)

plt.plot(np.array([sum(i) for i in note_out][:500000]), color='green', label='Raw MIDI by AI')
plt.plot([sum(i) for i in smoothed_note_out][:500000], color='magenta', label='Smoothed Raw MIDI by AI')

smoothed_final = []
smooth_aggression = -1
# loop through all note vectors and synthesize them into sine waves
for i in range(len(smoothed_note_out)-1):
    # silence is represented by an all zero vector!
    if(np.all(smoothed_encoded_note_out[i] == np.zeros(88))): # the case that it is silent
        smoothed_final.append(np.sin(0*t)) # append a silence to the final sequence
        continue

    # if it is not a silence, check if it is too similar, create the fourier transform
    if(np.sum(smoothed_note_out[i] != smoothed_note_out[i+1]) <= smooth_aggression and i != 0):
        smoothed_final.append(smoothed_final[i-1]) # append a silence to the final sequence
        continue

    if (smoothed_note_out[i][0]<87 and smoothed_note_out[i][0]>5):
        overall = np.cos(note_to_freq_with_offset(smoothed_note_out[i][0]) * 2 * np.pi * t)
    else:
        overall = np.sin(1 * 2 * np.pi * t) # take full cycle of 2pi radians and scale them by scaler T
    for freq in smoothed_note_out[i][1:]:
        if (freq<80 and freq>5):
            overall += np.cos(note_to_freq_with_offset(freq) * 2 * np.pi * t)
        else:
            continue
    smoothed_final.append(overall)

smoothed_gen = np.concatenate(smoothed_final)

prank = loader.parse_input(that_file, input_path) # parse input
Audio(loader.normalize(smoothed_gen,2), rate=sample_rate)

offsetted_note_out = [i-39 for i in [1000]+smoothed_note_out]

# generate all necessary wav files for assembly
output_dir_name = os.path.join(absolute_path, 'data', 'dancemonkey')
try:
    os.makedirs(output_dir_name)
    print(colored("Successfully opened output directory.".format(**locals()), 'green'))
except FileExistsError:
    print(colored("Write path {output_dir_name} already exists. Files might be overwritten and corrupted.".format(**locals()), 'yellow'))

# go throught all of the offsetted notes and generate using pitchshifter the correct pitch
template_dir = os.path.join(data_root_path, 'meow.wav')
for i in range(1, len(offsetted_note_out)):
    # check if the current note is the same as the LAST one. If it is, skip it, and if it is not, add the current one.
    if(np.all(offsetted_note_out[i] == offsetted_note_out[i-1])): # if they are equal
        continue # skip it
    # if not, get chord at that point, and generate a new wav chord
    chord = offsetted_note_out[i]
    offset_i = i-1
    for j in range(chord.size): # loop through all notes in chord
        fout_name = '{offset_i}_{j}.wav'.format(**locals())
        fout_dir = os.path.join(output_dir_name, fout_name)
        note_shift = chord[j] # sensitivity
        command = 'pitchshifter -s {template_dir} -o {fout_dir} -p {note_shift}'.format(**locals())
        # execute conversion with Timidity++. The user must have this installed and path configured in order to run this geneartion code successfully
        os.system(command)
print('done')
