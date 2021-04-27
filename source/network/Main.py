import os
import sys
import math
import numpy as np
from termcolor import colored
from tqdm import tqdm
import tensorflow as tf
sys.path.append('./source/network')
from Dataloader import datasync
from Dataloader import loader

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

# after file partitioning is done, we can start building the network and its definitions.
#It's not the best idea to do it directly in the main file but i am fucking tired so whatever.

# ==================== NETWORK DEFINITIONS ====================
# TODO: create network archetechture and training methods
