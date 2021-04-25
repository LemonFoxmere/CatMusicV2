# This file is meant as a test for parsing WAV files and how to correctly read them
# if you want to see what the parsed files look like or something, run this program in Atom with the Hydrogen plugin.

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavFile
from IPython.display import Audio
from termcolor import colored
from tqdm import tqdm

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

# import paths
absolute_path = os.path.join('/home/lemonorange/catRemixV2')
wav_input_dir_name = os.path.join(absolute_path, 'data', 'wav')
rawMid_input_dir_name = os.path.join(absolute_path, 'data', 'rawMid')
files = os.listdir(wav_input_dir_name)
file = files[1]
file_path = os.path.join(wav_input_dir_name, file)
Fs, data = wavFile.read(file_path)
size = int(Fs * 0.125)

Audio(data[0:size], rate=Fs)

prefix = file.split('_')[0]
new_file = '{prefix}.rawMid'.format(**locals())

fin = open(os.path.join(rawMid_input_dir_name, new_file))
raw_data = []
for i in fin:
    if(i == 'eof'): break
    raw_data.append(list(map(int,i.strip('\n').split())))

plt.plot(normalize(data, -100, 300), color='green', label='Raw Audio')

sum_raw_data = list(map(sum, raw_data))
thing = []
thing_trimmed = []
for i in sum_raw_data:
    for j in range(size):
        thing.append(i)
# trim data
trimmed = False
for i in sum_raw_data:
    for j in range(size):
        if(trimmed == True or i != 0):
            thing_trimmed.append(i)
            trimmed = True

plt.plot(thing, color='red', label='Raw Mids')
plt.plot(thing_trimmed, color='blue', label='Trimmed Raw Mids')
plt.title(label='Synchronization of MID and Audio (WAV normalized and MIDI summed).')
plt.legend()
plt.show()
