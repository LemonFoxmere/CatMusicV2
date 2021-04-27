import os
import math
import numpy as np
import scipy.io.wavfile as wavFile
from IPython.display import Audio
from termcolor import colored

class datasync:
    # FOR RAW MIDS ONLY
    @staticmethod
    def trim_front(input):
        trimmed = False
        result = []
        for i in input:
            if(trimmed == True or i[0] != 0):
                result.append(i)
                trimmed = True
        return np.array(result)

    # create a set of training and label data
    # TODO: VERIFY
    @staticmethod
    def sync_data(train, label, chunk_length, chunk_duration=0.125, rate=44000):
        # calculate the amount of wavSamples per chunk
        chunk_size = int(rate * chunk_duration)
        data_set = []
        for i in range(chunk_length-1):
            range_start = i * chunk_size
            range_end = (i+1) * chunk_size
            train_chunk = train[range_start, range_end]
            label_chunk = label[i]
            data_set.append(np.array([label_chunk, train_chunk]))
        return np.array(np.array(data_set))

class loader:
    # Math stuff
    @staticmethod
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

    # parsing IO
    @staticmethod
    def parse_input(input_file_name, input_path, norm=True, norm_lowbound=-1, norm_highbound=1):
        file_path = os.path.join(input_path, input_file_name)
        try:
            Fs, data = wavFile.read(file_path)
        except:
            print(colored('Error: Failed to open {input_file_name} as input! This file either does not exist or has internal errors and cannot be read.'.format(**locals()), 'red'))
            return -1 # if opening failed, return a -1 as error indication
        # check if normalization is wanted or not
        if(norm):
            return loader.normalize(data, norm_lowbound, norm_highbound)
        return data

    @staticmethod
    def parse_label(input_file_name, ground_truth_data_path):
        prefix = input_file_name.split('_')[0]
        label_file_name = '{prefix}.rawMid'.format(**locals())
        file_path = os.path.join(ground_truth_data_path, label_file_name)
        try:
            fin = open(file_path) # try to open label file
        except:
            print(colored('Error: Failed to open {label_file_name} as output! This file probably does not exist.'.format(**locals()), 'red'))
            return -1 # if opening failed, return a -1 as error indication
        # if opening successful, continue with parsing
        raw_data = []
        for i in fin: # add in all data lines except for eof in the file
            if(i == 'eof'): break
            raw_data.append(np.array(list(map(int,i.strip('\n').split()))))
        return np.array(raw_data) # processed data
