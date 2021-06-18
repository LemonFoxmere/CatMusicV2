import os
import math
from termcolor import colored
from tqdm import tqdm
import sys
sys.path.append('../network')
from Dataloader import loader

absolute_path = os.path.join('..','..')
# absolute_path = os.path.join('/home/lemonorange/catRemixV2')
input_dir_name = os.path.join(absolute_path, 'data', 'wav')
print('Attempting to open wav file path')
if(not os.path.exists(input_dir_name)): # check if path exists
    raise FileExistsError('Input path not found. You must generate all wav files first with \"MidToWav.py\"!')
# check parse size
parse_size = len(os.listdir(input_dir_name))
if(parse_size == 0):
    raise FileExistsError('No input files detect in \"{input_dir_name}\"! Are you sure you generated the files correctly?'.format(**locals()))
print(colored('Input path opened successfully', 'green'))
# create output directory
output_dir_name = os.path.join(absolute_path, 'data', 'rwav') # rwav stands for raw wav
try:
    os.makedirs(output_dir_name)
    print(colored("Successfully opened output directory. {output_dir_name} was created".format(**locals()), 'green'))
except FileExistsError:
    print(colored("Write path {output_dir_name} already exists. Opening path.".format(**locals()), 'yellow'))
print('Attempting to convert {parse_size} rWav files'.format(**locals()))

files = os.listdir(input_dir_name)
total_size = 0
for file in tqdm(files):
    prefix = file.split('.')[0]
    write_path = os.path.join(output_dir_name, '{prefix}.rwav'.format(**locals()))
    fout = open(write_path, 'w') # open the writing path
    data = loader.parse_input(file, input_dir_name, norm=False)
    fout.write('/'.join([str(i) for i in data.tolist()]))
    fout.close()
    total_size += os.path.getsize(os.path.join( write_path ))

total_size_mb = int(total_size / 10000)/100
files_length = len(files)
print(colored('Program Finished. Synthesized {files_length} rWavs, totalling {total_size_mb}MB.'.format(**locals()), 'green'))
