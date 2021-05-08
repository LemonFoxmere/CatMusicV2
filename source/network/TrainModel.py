import os
import sys
import math
import numpy as np
from termcolor import colored
from tqdm import tqdm, trange
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
import matplotlib.pyplot as plt
from functools import reduce
import time as systemclock

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
    m.add(LSTM(500, batch_input_shape=((None, 1, sample_per_chunk)), activation='tanh', return_sequences=True))
    m.add(LSTM(500, return_sequences=True, go_backwards=True, activation='tanh'))
    m.add(BatchNormalization())
    m.add(LSTM(400, return_sequences=True, activation='tanh'))
    m.add(LSTM(400, return_sequences=True, go_backwards=True, activation='tanh'))
    m.add(BatchNormalization())
    # m.add(LSTM(400, return_sequences=True, activation='relu'))
    # m.add(LSTM(400, return_sequences=True, go_backwards=True, activation='relu'))
    # m.add(BatchNormalization())
    m.add(LSTM(200, return_sequences=True, activation='tanh'))
    m.add(LSTM(200, go_backwards=True, activation='tanh'))
    m.add(BatchNormalization())

    m.add(Dense(120, activation='relu'))
    m.add(Dense(90, activation='relu'))
    m.add(Dense(88, activation='sigmoid'))

    return m

# ==================== LOSS & OPTIMIZER DEFINITIONS ====================
loss_list = [
    tf.losses.binary_crossentropy,
    tf.losses.categorical_crossentropy,
    tf.losses.mean_squared_error,
    tf.losses.mean_absolute_error,
    tf.nn.sigmoid_cross_entropy_with_logits,
]
loss_label = [ # how the loss functions will be named when displayed on a graph
    'Binary_CSE',
    'Categorical_CSE',
    'Mean Squared Error',
    'Mean Absolute Error',
    'sigmoid_CSE_with_Logits',
]
default_loss_index = 0

# optimizer selections
default_opt = Adam(lr=1e-3)

# ==================== RECORDING PROCESS ====================
def write_loss(losses, labels, fout):
    i = 0
    write_out = ''
    for loss in losses:
        line = str(np.mean(loss.numpy().tolist()))
        write_out += labels[i]+'='
        write_out += line if line != 'nan' else '-1'
        write_out += ';'
        i+=1
    write_out += '\n'
    fout.write(write_out)

# ==================== TRAINING PROCESS ====================
def train_step(input, ground_truth, model, train_fout=None, val_fout=None): # training input and ground truth should already be synchronized and encoded
    # fout is there for if we want to record the loss data
    # get a validation set
    validation_input_name = np.random.choice(validation_files)
    # parse the validation set
    validation_input = loader.parse_input(validation_input_name, input_path)
    validation_output = sync.trim_front(loader.parse_label(validation_input_name, ground_truth_data_path))
    val_input, val_label = sync.sync_data(validation_input, validation_output, len(validation_output))
    val_label = np.array(loader.encode_multihot(val_label)) # encode to multi-hot

    val_input = np.array(val_input)
    val_input = np.reshape(val_input, (val_input.shape[0], 1, val_input.shape[1])) # reshape to a tensor which the neural net can use (batch_size, 1, samples_per_batch)

    with tf.GradientTape() as tape:
        # generate the predictions
        training_prediction = model(input)
        validation_prediction = model(val_input)

        # it is garenteed the ground truth and prediction will have the same shape
        training_losses = [x(ground_truth, training_prediction) for x in loss_list]
        validation_losses = [x(val_label, validation_prediction) for x in loss_list]
        applicable_loss = training_losses[default_loss_index]
        visible_loss = validation_losses[default_loss_index]

        # store loss
        if(train_fout != None):
            write_loss(training_losses, loss_label, train_fout)
        if(val_fout != None):
            write_loss(validation_losses, loss_label, val_fout)

        # calculate and apply gradient
        grad = tape.gradient(applicable_loss, model.trainable_variables)
        # THIS SHIT DOESNT WORK AND IDK WHY
        default_opt.apply_gradients(zip(grad, model.trainable_variables))

        overall_train_loss = np.mean(applicable_loss)
        overall_val_loss = np.mean(visible_loss)
        # CLI debug messages
        # print(colored('>>> Overall Training Loss: ', 'green') + colored(str(overall_train_loss), 'green', attrs=['bold', 'reverse']))
        # print(colored('>>> Overall Validation Loss: ', 'green') + colored(str(overall_val_loss), 'green', attrs=['bold', 'reverse']))

        return overall_train_loss, overall_val_loss

#  start training process
save_interval = 20 # save an instance every 20 cycles (makeshift early stopping)
epochs = 1
data_cut_off = 100

# training data storages
network_write_path = os.path.join(absolute_path, 'network', 'latest-cycle')
# if it doesn't exist, create one
try:
    os.makedirs(network_write_path) # if it doesn't exist, create on
except FileExistsError:
    pass # if it already exist, keep it that way
loss_tracking = open(os.path.join(network_write_path, 'loss.tex'), 'w')
np.random.shuffle(training_files)

# initialize model and training processes
model = make_model() # create a new model instance
model.compile()

for temp_file in training_files[:data_cut_off]:
    # print loop header
    print(colored('----- Training on {temp_file} -----'.format(**locals()), ))

    # start training process
    unpaired_input = loader.parse_input(temp_file, input_path) # parse input
    unpaired_ground_truth = sync.trim_front(loader.parse_label(training_files[100], ground_truth_data_path)) # parse output
    input, ground_truth = sync.sync_data(unpaired_input, unpaired_ground_truth, len(unpaired_ground_truth)) # pair IO + trim
    ground_truth = np.array(loader.encode_multihot(ground_truth))

    input = np.array(input)
    input = np.reshape(input, (input.shape[0], 1, input.shape[1])) # reshape to a tensor which the neural net can use (batch_size, 1, samples_per_batch)

    overall_train_loss, overall_val_loss = train_step(input, ground_truth, model, fout=loss_tracking)

    # console log all losses
    print(colored('>>> Overall Training Loss: ', 'green') + colored(str(overall_train_loss), 'green', attrs=['bold', 'reverse']))
    print(colored('>>> Overall Validation Loss: ', 'green') + colored(str(overall_val_loss), 'green', attrs=['bold', 'reverse']))

# close loss tracker, assuming program terminated correctly
loss_tracking.close()

# THESE CODE ARE FOR TESTING PURPOSES ONLY.
# test_in = loader.parse_input(training_files[100], input_path)
# test_gt = sync.trim_front(loader.parse_label(training_files[100], ground_truth_data_path))
# train_set, label_set = sync.sync_data(test_in, test_gt, len(test_gt))
#
# validation_input_name = np.random.choice(validation_files)
# # parse the validation set
# validation_input = loader.parse_input(validation_input_name, input_path)
# validation_output = sync.trim_front(loader.parse_label(validation_input_name, ground_truth_data_path))
# val_input, val_label = sync.sync_data(validation_input, validation_output, len(validation_output))
# val_label = loader.encode_multihot(val_label) # encode to multi-hot
#
# training_losses = [x(val_label, [np.zeros(88, dtype=np.float32)]*13) for x in loss_list]
