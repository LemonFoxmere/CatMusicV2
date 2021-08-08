import os
import sys
import math
import numpy as np
import scipy.io.wavfile as wavFile
from scipy import signal
from IPython.display import Audio
from termcolor import colored
import scipy as sc
import librosa
import librosa.display

class datasync: # FOR SYNCRONIZING DATA ONLY
    # FOR RAW MIDS ONLY
    @staticmethod
    def trim_front(input):
        trimmed = False
        result = []
        for i in input:
            if(trimmed == True or i[0] != 0):
                result.append(i)
                trimmed = True
        return result

    # TODO: create new synchronizer
    # create a set of training and label data
    @staticmethod
    def sync_data(input, label, bpm, hop_length, rate=44000): # input will be a spectrogram, and label will be rawMid
        # calculate note length based on bpm
        sample_length = 1/44000 * (10**6) # this is the sample duration in microseconds
        qn_length = 60/bpm * (10**6) # this is the duration of every quarter note, or 4th note in microseconds
        sn_length = qn_length / 4 #  this is the duration of every semi-quaver, or 16th note in microseconds
        # convert hop_length to microseconds for consistency
        hop_length_Ms = sample_length * hop_length

        # the input length is pretty much garenteed to be moer than the label length because of how it is generated
        synced_inp = []
        synced_lab = []

        # syncronize every chord
        for i in range(len(label)):
            # every chord will be sn_length long
            ex_win_length = math.ceil(sn_length / hop_length_Ms) # this is how much of the window we need to extract from the mel spec
            ex_start_pos = math.ceil((i*sn_length) / hop_length_Ms) # this is the window position in which we need to start extracting
            # add in the syncronized label
            synced_lab.append(label[i])
            # add in the input section
            synced_inp.append(input[ex_start_pos : ex_start_pos + ex_win_length])

        # trim the unusable part so it doesnt create a ragged array
        final_inp = []
        final_lab = []
        for i in range(len(synced_inp)-1):
            if(synced_inp[i].shape[0] == synced_inp[i+1].shape[0]):
                final_inp.append(synced_inp[i])
                final_lab.append(synced_lab[i])
        if(synced_inp[-1].shape[0] == synced_inp[0].shape[0]):
            final_inp.append(synced_inp[-1])
            final_lab.append(synced_lab[-1])

        return np.array(final_inp), final_lab

    # >>>>>>>>>>>>>>>>>>>>>>>>> OLD >>>>>>>>>>>>>>>>>>>>>>>>>
    # @staticmethod
    # def sync_data(train, label, chunk_length, chunk_duration=0.125, rate=44000):
    #     # calculate the amount of wavSamples per chunk
    #     chunk_size = int(rate * chunk_duration)
    #     train_set = []
    #     label_set = []
    #
    #     for i in range(chunk_length-1):
    #         range_start = i * chunk_size
    #         range_end = (i+1) * chunk_size
    #         train_chunk = train[range_start : range_end]
    #         if(len(train_chunk) < chunk_size): # fill in gaps
    #             train_chunk = np.concatenate((train_chunk, np.zeros(chunk_size-len(train_chunk))))
    #         label_chunk = label[i]
    #         train_set.append(train_chunk)
    #         label_set.append(label_chunk)
    #     return train_set, label_set
    # >>>>>>>>>>>>>>>>>>>>>>>>> OLD >>>>>>>>>>>>>>>>>>>>>>>>>

class loader: # FOR EVERYTHING ELSE
    # Math stuff
    @staticmethod
    def normalize(arr, r):
        return arr / (np.max(arr)/r)

    # for padding in ragged array, if any.
    @staticmethod
    def numpy_fillna(data):
        max_len = np.max([len(a) for a in data])
        out = np.asarray([np.pad(a, (0, max_len - len(a)), 'constant', constant_values=0) for a in data])
        return out

    # parsing IO
    @staticmethod
    def parse_input(input_file_name, input_path, norm=True, norm_bound=1):
        input_file_name = str(input_file_name)
        input_path = str(input_path)
        file_path = os.path.join(input_path, input_file_name)
        try:
            data, Fs = librosa.load(file_path, sr=None)
        except:
            print(colored('Error: Failed to open \"{input_file_name}\" as input! This file either does not exist or has internal errors and cannot be read.'.format(**locals()), 'red'))
            return np.array([]) # if opening failed, return a -1 as error indication
        # check if normalization is wanted or not

        if(data.size == 0):
            return np.array([]), Fs

        if(norm):
            return loader.normalize(data, norm_bound), Fs
        return data, Fs

    @staticmethod
    def parse_label(input_file_name, ground_truth_data_path):
        input_file_name = str(input_file_name)
        ground_truth_data_path = str(ground_truth_data_path)
        prefix = input_file_name.split('_')[0]
        label_file_name = '{prefix}.rawMid'.format(**locals())
        file_path = os.path.join(ground_truth_data_path, label_file_name)
        try:
            fin = open(file_path) # try to open label file
        except:
            print(colored('Error: Failed to open \"{label_file_name}\" as output! This file probably does not exist.'.format(**locals()), 'red'))
            return -1 # if opening failed, return a -1 as error indication
        # if opening successful, continue with parsing
        bpm = int(fin.readline().strip()) # Added to read in variation in BPMs
        raw_data = []
        for i in fin: # add in all data lines except for eof in the file
            if(i == 'eof'): break
            raw_data.append(np.array(list(map(int,i.strip('\n').split()))))
        return raw_data, bpm # returns the raw data and a bpm to go along with it for syncing purposes

    @staticmethod
    def encode_multihot(arrs, encoding_size=88):
        # assuming that the given array is in the domain of 0-87, we can directly map each note to a one in a [encoding_size,] sized vector of 0s
        # any chord that starts with 0s will be considered silence due to the feature in rawMidi parser. Check TODO.md for more details
        # first check bound errors
        final_result = []
        for arr in arrs:
            min_arr = min(arr)
            max_arr = max(arr)
            arr = sorted(arr, reverse=True)
            if(max_arr > encoding_size-1 or min_arr < 0):
                raise RuntimeError('Array max and/or min values exceed encoding size. Max: {max_arr}, min: {min_arr}, encoding_size: {encoding_size}'.format(**locals()))
            # start encoding process
            result = np.zeros(88, dtype=np.float32)
            if(arr[0] == 0):
                final_result.append(result)
                continue
            for i in arr:
                result[i] = 1
            final_result.append(result)
        return final_result

    @staticmethod
    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    @staticmethod
    def butter_highpass_filter(data, cutoff, fs, order=5):
        b, a = loader.butter_highpass(cutoff, fs, order=order)
        y = signal.filtfilt(b, a, data)
        return y

    @staticmethod
    def get_mel_spec(y, mel_res, sr, window_size, hop_length, window=sc.signal.windows.hann, transposed=True, high_pass=False, cutoff=100, normalize=True):

        # pass data thru high pass first
        if(high_pass):
            S = librosa.feature.melspectrogram(butter_highpass_filter(y, cutoff, sr), sr=sr, n_fft=window_size, hop_length=hop_length, n_mels=mel_res) # create mel spectrogram w/ hp
        else:
            S = librosa.feature.melspectrogram(y, sr=sr, n_fft=window_size, hop_length=hop_length, n_mels=mel_res) # create mel spectrogram

        S_DB = librosa.power_to_db(S, ref=np.max) # convert amplitube to db

        # normalize DBs
        if(normalize):
            S_DB = loader.normalize(S_DB+80, 1)

        if(transposed):
            return S_DB.transpose()
            # will return a N x N_MELS matrix, with every row representing a vector

        return S_DB
        # will return a N_MELS x N matrix, with every column representing a vector

    @staticmethod
    def create_input_matrix(ML, window_size, hop_length):
        final = []
        for i in range(0, ML.shape[0] - window_size, hop_length):
            final.append(ML[i : i+window_size])
        return np.array(final)
