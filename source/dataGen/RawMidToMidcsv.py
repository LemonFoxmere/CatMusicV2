import os
import math
import numpy as np
from termcolor import colored
from tqdm import tqdm

# piano instruments: 0,1,2,3,4,5
def generateCsvMidHeader(instrument, tempo):
    # I have no idea what these number means and at this point I am just hoping that it will work
    csv_lines = [
        '0,0,Header,1,3,384',
        '1,0,Start_track',
        '1,0,Tempo,{tempo}'.format(**locals()),
        '1,0,Time_signature,4,2,24,8',
        '1,0,End_track',
        '2,0,Start_track',
        '2,0,Program_c,0,{instrument}'.format(**locals())
    ]
    return csv_lines

def chord_to_csvmid(chord, clock_time_stamp, on=True):
    csv = []
    for i in chord:
        i = int(i) + 21 # adjust from 88 piano keys to full 128 range
        csv.append('2,{clock_time_stamp},Note_on_c,0,{i},127'.format(**locals()) if on else '2,{clock_time_stamp},Note_off_c,0,{i},127'.format(**locals()))
    return csv

# REVIEW CODE
def parse_raw_midi(raw_mid, last_chord=None, time_stamp=0):
    clock_time_stamp = seconds_to_clock(time_stamp) # return a midi time stamp
    # process last chord
    if(len(raw_mid) == 0): # base case
        # if the very last tone was a 0, we just need to tack on an End_track. Or else we end off the chord and then tack on the End_track
        return ['2,{clock_time_stamp},End_track'.format(**locals())] if last_chord[0] == 0 else chord_to_csvmid(last_chord, clock_time_stamp, on=False) + ['2,{clock_time_stamp},End_track'.format(**locals())]
    # if it isnt the last one, first check if last_chord is None or not. If it if, pass 1 recursive call to initialize last_chord
    if(str(type(last_chord)) == '<class \'NoneType\'>'):
        # check if the first note is silent or not
        if(raw_mid[0][0] == 0): # if it is, we just increment the time, as there is nothing to play
            return parse_raw_midi(raw_mid[1:], last_chord=raw_mid[0], time_stamp=time_stamp+0.125) # increment time by 0.125s
        # if the start is not silent, start playing the chord, and increment the time
        return chord_to_csvmid(raw_mid[0], clock_time_stamp) + parse_raw_midi(raw_mid[1:], last_chord=raw_mid[0], time_stamp=time_stamp+0.125)
    # if this is not the initialization part, first check if last chord and current chord are consistent
    if(last_chord.shape[0] == raw_mid[0].shape[0]):
        if((last_chord==raw_mid[0]).all()):
            # if the last one is same as the current one, do not end off or start a chord yet, and continue with recursion
            return parse_raw_midi(raw_mid[1:], last_chord=raw_mid[0], time_stamp=time_stamp+0.125) # increment time by 0.125s
    # if the last chord and current chord are not consistent, check if the current chord is silent or not
    if(raw_mid[0][0] == 0): # if it is not consistent and the current is a 0, it must imply that the last chord was concluding
        # if it is silent, just end the last chord and repeat recursion
        return chord_to_csvmid(last_chord, clock_time_stamp, on=False) + parse_raw_midi(raw_mid[1:], last_chord=raw_mid[0], time_stamp=time_stamp+0.125) # increment time by 0.125s
    # if the current chord is not silent, check if the last chord is silent or not.
    if(last_chord[0] == 0): # if this is true, then it must imply that a new chord is starting.
        return chord_to_csvmid(raw_mid[0], clock_time_stamp) + parse_raw_midi(raw_mid[1:], last_chord=raw_mid[0], time_stamp=time_stamp+0.125) # increment time by 0.125s
    # if neither the last one or the current one is a 0, then it must mean that a chord is swtiched. Therefore, end the last one and begin the new one
    return chord_to_csvmid(last_chord, clock_time_stamp, on=False) + chord_to_csvmid(raw_mid[0], clock_time_stamp) + parse_raw_midi(raw_mid[1:], last_chord=raw_mid[0], time_stamp=time_stamp+0.125) # increment time by 0.125s

# check if raw data available
print('Attempting to open rawMid file path')
absolute_path = os.path.join('..','..')
input_dir_name = os.path.join(absolute_path, 'data', 'rawMid')
if(not os.path.exists(input_dir_name)): # check if path exists
    raise FileExistsError('Input path not found. You must generate all rawMid files first with \"genRawMid.py\"!')
parse_size = len(os.listdir(input_dir_name))
if(parse_size == 0):
    raise FileExistsError('No input files detect in \"{input_dir_name}\"! Are you sure you generated the files correctly?'.format(**locals()))
print(colored('Path opened successfully', 'green'))
file_list = os.listdir(input_dir_name)

# create output directory
output_dir_name = os.path.join(absolute_path, 'data', 'csvMid')
try:
    os.makedirs(output_dir_name)
    print(colored("Successfully opened output directory. {output_dir_name} was created".format(**locals()), 'green'))
except FileExistsError:
    print(colored("Write path {output_dir_name} already exists. Opening path.".format(**locals()), 'yellow'))

print('Attempting to parse data from {parse_size} files'.format(**locals()))

# generation meta-data
tempo = 500000
clock_duration = ((tempo/1000000)/24)/16
seconds_to_clock = lambda x : math.floor(x/clock_duration) # calculate how much the clock will fire within a certain duration

for file in tqdm(file_list):
    # begin parsing
    fin = open(os.path.join(input_dir_name, file))
    data = []
    for line in fin: # read in every line into data
        if(line == 'eof'): break
        data.append(np.array(list(map(int, line.split()))))
    control_data = parse_raw_midi(data)

    for i in range(6):
        # generate csv with unique instrument types
        csv = generateCsvMidHeader(i, 500000) # genearte the header
        prefix = file.split('.')[0] # obtain file prefix
        fout = open(os.path.join(output_dir_name, '{prefix}_{i}.csv'.format(**locals())), 'w')
        final_csv = '\n'.join(csv + control_data + ['0,0,End_of_file']) # add control data
        fout.write(final_csv)
        fout.close()
    # finish 1 file iteration

print(colored('Program Finished. All data parsed without any errors', 'green'))
