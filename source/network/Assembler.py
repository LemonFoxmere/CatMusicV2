from termcolor import colored
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
print(colored('All libraries loaded', attrs=['bold']))
from datetime import datetime
import difflib

chunk_length_seconds = 0.125
sample_rate = 44100
sample_per_chunk = int(sample_rate * chunk_length_seconds)

absolute_path = os.path.join('/home/lemonorange/catRemixV2')
data_root_path = os.path.join(absolute_path, 'data')
input_path = os.path.join(data_root_path, 'wav')
label_data_path = os.path.join(data_root_path, 'rawMid')
storage_path = os.path.join(absolute_path, 'network')

# read_path = os.path.join(data_root_path, 'lol-phoenix')
read_path = os.path.join(data_root_path, 'lol-phoenix')

files = sorted(os.listdir(read_path), key = lambda x : int(x.split('_')[0]))
max_length = chunk_length_seconds * (int(files[-1].split('_')[0])+10)
max_samples = int(max_length * sample_rate)
max_length = max_samples / sample_rate
max_note_index = int(files[-1].split('_')[0])

# start assembly
overall_audio = np.zeros(max_samples).tolist()

def get_note(time_index): # return a list of files for a certain note in a time range
    x = []
    for i in files:
        if(int(i.split('_')[0]) == time_index):
            x.append(i) # if it matches the time frame, add it
    return x

# loop through all files and put them in the correct place
for i in range(max_note_index+1):
    # retrieve the corresponding notes
    note_files = get_note(i)
    if(len(note_files) == 0): continue # if the current note is empty, then continue
    # if it is not empty, read in all notes and fuse them together
    notes = [loader.parse_input(j, read_path, norm=False) for j in note_files]
    fused = sum(notes) # they are garenteed to have the same shape
    # add in the fused
    start_index = i * sample_per_chunk # get start index of where the thing should be placed
    duration = i * fused.size
    overall_audio[start_index : start_index + duration] = fused.tolist()

Audio(overall_audio, rate=sample_rate)
