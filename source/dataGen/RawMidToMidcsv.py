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
# TODO: figure out what the fuck this do again?????
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
bpm_to_tempo = lambda x : int(60/x * (10**6)) # ex: 120 bpm = 0.5 seconds per beat = 0.5 * 10^6 μs per beat
fixed_tempo = bpm_to_tempo(120)
clock_duration = ((fixed_tempo/1000000)/24)/16

for file in tqdm(file_list):
    # begin parsing
    fin = open(os.path.join(input_dir_name, file))
    data = []
    bpm = int(fin.readline().strip()) # Added to read in variation in BPMs
    tempo = bpm_to_tempo(bpm) # SET NEW TEMPO
    seconds_to_clock = lambda x : math.floor(x/clock_duration) # calculate how much the clock will fire within a certain duration
    #TODO: THIS MIGHT BE DEFEATING THE PURPOSE!!! CHECK AUDIO SAMPLE!!!
    # it does fucking defeat the purpose
    # nvm its fixed ignore that

    for line in fin: # read in every line into data
        if(line == 'eof'): break
        data.append(np.array(list(map(int, line.split()))))
    control_data = parse_raw_midi(data) # parse data into control information

    for i in range(7): # 0-7 = all piano titles
        # generate csv with unique instrument types
        csv = generateCsvMidHeader(i, tempo) # genearte the header
        prefix = file.split('.')[0] # obtain file prefix
        fout = open(os.path.join(output_dir_name, '{prefix}_{i}.csv'.format(**locals())), 'w')
        final_csv = '\n'.join(csv + control_data + ['0,0,End_of_file']) # add control data
        fout.write(final_csv)
        fout.close()
    # finish 1 file iteration

print(colored('Program Finished. All data parsed without any errors', 'green'))

# MIDI INSTRUMENTS AND THEIR IDS:
'''
Piano:
1 Acoustic Grand Piano
2 Bright Acoustic Piano
3 Electric Grand Piano
4 Honky-tonk Piano
5 Electric Piano 1
6 Electric Piano 2
7 Harpsichord
8 Clavinet

Chromatic Percussion:
9 Celesta
10 Glockenspiel
11 Music Box
12 Vibraphone
13 Marimba
14 Xylophone
15 Tubular Bells
16 Dulcimer

Organ:
17 Drawbar Organ
18 Percussive Organ
19 Rock Organ
20 Church Organ
21 Reed Organ
22 Accordion
23 Harmonica
24 Tango Accordion

Guitar:
25 Acoustic Guitar (nylon)
26 Acoustic Guitar (steel)
27 Electric Guitar (jazz)
28 Electric Guitar (clean)
29 Electric Guitar (muted)
30 Overdriven Guitar
31 Distortion Guitar
32 Guitar harmonics

Bass:
33 Acoustic Bass
34 Electric Bass (finger)
35 Electric Bass (pick)
36 Fretless Bass
37 Slap Bass 1
38 Slap Bass 2
39 Synth Bass 1
40 Synth Bass 2

Strings:
41 Violin
42 Viola
43 Cello
44 Contrabass
45 Tremolo Strings
46 Pizzicato Strings
47 Orchestral Harp
48 Timpani

Strings (continued):
49 String Ensemble 1
50 String Ensemble 2
51 Synth Strings 1
52 Synth Strings 2
53 Choir Aahs
54 Voice Oohs
55 Synth Voice
56 Orchestra Hit

Brass:
57 Trumpet
58 Trombone
59 Tuba
60 Muted Trumpet
61 French Horn
62 Brass Section
63 Synth Brass 1
64 Synth Brass 2

Reed:
65 Soprano Sax
66 Alto Sax
67 Tenor Sax
68 Baritone Sax
69 Oboe
70 English Horn
71 Bassoon
72 Clarinet

Pipe:
73 Piccolo
74 Flute
75 Recorder
76 Pan Flute
77 Blown Bottle
78 Shakuhachi
79 Whistle
80 Ocarina

Synth Lead:
81 Lead 1 (square)
82 Lead 2 (sawtooth)
83 Lead 3 (calliope)
84 Lead 4 (chiff)
85 Lead 5 (charang)
86 Lead 6 (voice)
87 Lead 7 (fifths)
88 Lead 8 (bass + lead)

Synth Pad:
89 Pad 1 (new age)
90 Pad 2 (warm)
91 Pad 3 (polysynth)
92 Pad 4 (choir)
93 Pad 5 (bowed)
94 Pad 6 (metallic)
95 Pad 7 (halo)
96 Pad 8 (sweep)

Synth Effects:
97 FX 1 (rain)
98 FX 2 (soundtrack)
99 FX 3 (crystal)
100 FX 4 (atmosphere)
101 FX 5 (brightness)
102 FX 6 (goblins)
103 FX 7 (echoes)
104 FX 8 (sci-fi)

Ethnic:
105 Sitar
106 Banjo
107 Shamisen
108 Koto
109 Kalimba
110 Bag pipe
111 Fiddle
112 Shanai

Percussive:
113 Tinkle Bell
114 Agogo
115 Steel Drums
116 Woodblock
117 Taiko Drum
118 Melodic Tom
119 Synth Drum

Sound effects:
120 Reverse Cymbal
121 Guitar Fret Noise
122 Breath Noise
123 Seashore
124 Bird Tweet
125 Telephone Ring
126 Helicopter
127 Applause
128 Gunshot
'''
