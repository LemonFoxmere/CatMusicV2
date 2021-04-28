import os
import sys
import math
import numpy as np
from termcolor import colored
from tqdm import tqdm
sys.path.append('./source/network')
from Dataloader import datasync as sync
from Dataloader import loader
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
import keras.backend as K
from keras.optimizers import Adam, SGD, RMSprop, Adagrad
from tensorflow.python.client import device_lib
from matplotlib import pyplot as plt
from functools import reduce

# set memory growth
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

# PATH DEFINITIONS
absolute_path = os.path.join('/home/lemonorange/catRemixV2')
data_root_path = os.path.join(absolute_path, 'data')
input_path = os.path.join(data_root_path, 'wav')
ground_truth_data_path = os.path.join(data_root_path, 'rawMid')
print(colored('\n==================== DEBUG MESSAGES ====================\n', 'grey', 'on_yellow'))
print('Attempting to opening dataset paths...')
if(not os.path.exists(input_path)):
    raise FileNotFoundError('Input path does not exist! Are you sure all of the data are generated?') # check if path exists
if(not os.path.exists(ground_truth_data_path)):
    raise FileNotFoundError('Label path does not exist! Are you sure all of the data are generated?') # check if path exists

# DATA PARTITIONS
training_files = []
testing_files = []
validation_files = []

# DATA PARTITIONING
val_perc=0.001 # partitioning parameters
test_perc=0.005 # partitioning parameters

print('Attempting to partition data...')
all_files = os.listdir(input_path)
data_size = len(all_files)
# partition validation files
validation_files = np.array(all_files[0 : int( data_size * val_perc )])
all_files = all_files[int( data_size * val_perc ) :] # trim original files
# partition test files
testing_files = np.array(all_files[0 : int( data_size * test_perc )])
training_files = np.array(all_files[int( data_size * test_perc ) :])

validation_file_size = validation_files.shape[0]
testing_files_size = testing_files.shape[0]
training_files_size = training_files.shape[0]
# DATA PARTITIONING Finished
print(colored('Data partitioned successfully!', 'green'))
print(colored('  |__ {validation_file_size} validation data points'.format(**locals()), 'green'))
print(colored('  |__ {testing_files_size} testing data points'.format(**locals()), 'green'))
print(colored('  |__ {training_files_size} training data points'.format(**locals()), 'green'))

# DATA details
chunk_length_seconds = 0.125
sample_rate = 44000
sample_per_chunk = int(sample_rate * chunk_length_seconds)

# after file partitioning is done, we can start building the network and its definitions.
# It's not the best idea to do it directly in the main file but i am fucking tired so whatever.

# ==================== NETWORK DEFINITIONS ====================
def make_model():
    m = Sequential()
    m.add(LSTM(500, batch_input_shape=((None, sample_rate, 1)), activation='relu', return_sequences=True))
    m.add(LSTM(500, return_sequences=True, go_backwards=True, activation='relu'))
    m.add(BatchNormalization())
    m.add(LSTM(400, return_sequences=True, activation='relu'))
    m.add(LSTM(400, return_sequences=True, go_backwards=True, activation='relu'))
    m.add(BatchNormalization())
    # m.add(LSTM(400, return_sequences=True, activation='relu'))
    # m.add(LSTM(400, return_sequences=True, go_backwards=True, activation='relu'))
    # m.add(BatchNormalization())
    m.add(LSTM(200, return_sequences=True, activation='relu'))
    m.add(LSTM(200, go_backwards=True, activation='relu'))
    m.add(BatchNormalization())

    m.add(Dense(120, activation='relu'))
    m.add(Dense(88, activation='sigmoid'))

    return m

# ==================== LOSS DEFINITIONS ====================
loss_list = [
    tf.losses.binary_crossentropy,
    tf.losses.categorical_crossentropy,
    tf.losses.mean_squared_error,
    tf.losses.mean_absolute_error,
    tf.nn.sigmoid_cross_entropy_with_logits,
]
default_loss_index = 0

# ==================== TRAINING PROCESS ====================
def train_step(input, ground_truth, model):
    with tf.GradientTape() as tape:
        pass
        # TODO: implement BPTT

# FOR TESTING PURPOSES ONLY. DELETE THIS
test_in = loader.parse_input(training_files[100], input_path)
test_gt = sync.trim_front(loader.parse_label(training_files[100], ground_truth_data_path))
train_set, label_set = sync.sync_data(test_in, test_gt, len(test_gt))
