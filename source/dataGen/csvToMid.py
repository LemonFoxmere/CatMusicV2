import os
import math
from termcolor import colored
from tqdm import tqdm

# absolute_path = os.path.join('..','..')
absolute_path = os.path.join('/home/lemonorange/catRemixV2')
input_dir_name = os.path.join(absolute_path, 'data', 'csvMid')
print('Attempting to open csvMid file path')
if(not os.path.exists(input_dir_name)): # check if path exists
    raise FileExistsError('Input path not found. You must generate all csv files first with \"rawMidToMidcsv.py\"!')
# check parse size
parse_size = len(os.listdir(input_dir_name))
if(parse_size == 0):
    raise FileExistsError('No input files detect in \"{input_dir_name}\"! Are you sure you generated the files correctly?'.format(**locals()))
print(colored('Input path opened successfully', 'green'))

# create output directory
output_dir_name = os.path.join(absolute_path, 'data', 'mid')
try:
    os.makedirs(output_dir_name)
    print(colored("Successfully opened output directory. {output_dir_name} was created".format(**locals()), 'green'))
except FileExistsError:
    print(colored("Write path {output_dir_name} already exists. Opening path.".format(**locals()), 'yellow'))
print('Attempting to convert {parse_size} CSV files into MID files'.format(**locals()))

files = os.listdir(input_dir_name)
for file in tqdm(files):
    prefix = file.split('.')[0]
    write_path = os.path.join(output_dir_name, '{prefix}.mid'.format(**locals()))
    read_path = os.path.join(input_dir_name, file)
    command = 'csvmidi {read_path} {write_path}'.format(**locals())
    # execute conversion with csvmidi. The user must have this installed and path configured in order to run this geneartion code successfully
    os.system(command)

print(colored('Program Finished. All files are converted without any errors', 'green'))
