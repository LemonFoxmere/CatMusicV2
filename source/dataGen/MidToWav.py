import os
import math
from termcolor import colored
from tqdm import tqdm

absolute_path = os.path.join('..','..')
# absolute_path = os.path.join('/home/lemonorange/catRemixV2')
input_dir_name = os.path.join(absolute_path, 'data', 'mid')
print('Attempting to open mid file path')
if(not os.path.exists(input_dir_name)): # check if path exists
    raise FileExistsError('Input path not found. You must generate all mid files first with \"csvToMid.py\"!')
# check parse size
parse_size = len(os.listdir(input_dir_name))
if(parse_size == 0):
    raise FileExistsError('No input files detect in \"{input_dir_name}\"! Are you sure you generated the files correctly?'.format(**locals()))
print(colored('Input path opened successfully', 'green'))

# create output directory
output_dir_name = os.path.join(absolute_path, 'data', 'wav')
try:
    os.makedirs(output_dir_name)
    print(colored("Successfully opened output directory. {output_dir_name} was created".format(**locals()), 'green'))
except FileExistsError:
    print(colored("Write path {output_dir_name} already exists. Opening path.".format(**locals()), 'yellow'))
synth_SR = 44000
print('Attempting to synthesize {parse_size} MID files at sampling rate of {synth_SR}hz'.format(**locals()))

files = os.listdir(input_dir_name)
total_size = 0
for file in tqdm(files):
    prefix = file.split('.')[0]
    write_path = os.path.join(output_dir_name, '{prefix}.wav'.format(**locals()))
    read_path = os.path.join(input_dir_name, file)
    command = 'timidity {read_path} -Ow --output-mono -s {synth_SR} -o {write_path}'.format(**locals())
    # execute conversion with Timidity++. The user must have this installed and path configured in order to run this geneartion code successfully
    os.system(command)
    total_size += os.path.getsize(os.path.join( write_path ))
total_size_mb = int(total_size / 10000)/100
files_length = len(files)
print(colored('Program Finished. Synthesized {files_length} wavs, totalling {total_size_mb}MB.'.format(**locals()), 'green'))
