import os
import math
import numpy as np

def generate_chord(note_count):
    chord = []
    for i in range(note_count):
        new_note = math.floor(np.random.normal(39,15,1)) # generate a new unique note
        if(new_note < 0): new_note = 0
        elif(new_note > 87): new_note = 87
        while(new_note in chord): # check for repetitions
            new_note = math.floor(np.random.normal(39,15,1))
            if(new_note < 0): new_note = 0
            elif(new_note > 87): new_note = 87
        chord.append(new_note) # add note to chord
    return chord

# genearte a sample recursively
def generate_sample(chord_count, rep_epsilon, rep_epsilon_decay, gap_epsilon, gap_epsilon_decay, last_chord=None):
    if(chord_count == 0):
        # recursive base case
        return []
    # if not base case, determine whether this chord should be a gap or not with gap_epsilon
    gap_rand = np.random.rand()
    if(gap_rand <= gap_epsilon):
        # execute epsilon decay and store current note as 2x 125ms gap
        return [np.array([0]), np.array([0])] + generate_sample(chord_count, rep_epsilon, rep_epsilon_decay, gap_epsilon-gap_epsilon_decay, gap_epsilon_decay)
    # if gap does not generate, check is last_chord is None, so as requiring to generate a new starting chord
    if(last_chord == None):
        # generate new chord
        note_count = math.floor(np.random.normal(3.5, 2, 1))
        if(note_count < 1): note_count = 1
        elif(note_count > 8): note_count = 8
        chord = generate_chord(note_count)
        # after chord generation is complete, generate another one recursively, while increasing the chance for a 250ms gap (epsilon growth)
        return [np.array(chord)] + generate_sample(chord_count-1, rep_epsilon, rep_epsilon_decay, gap_epsilon+gap_epsilon_decay*0.5, gap_epsilon_decay, last_chord=chord)
    else: # there was a chord before hand, and we need to determine whether or not to repeat the chord
        rep_rand = np.random.rand()
        if(rep_rand <= rep_epsilon): # if decided to repeat
            # execute epsilon decay and store previous note, while performing gap epsilon growth
            return [np.array(last_chord)] + generate_sample(chord_count-1, rep_epsilon-rep_epsilon_decay, rep_epsilon_decay, gap_epsilon+gap_epsilon_decay/4, gap_epsilon_decay, last_chord=last_chord)
        else: # if decided to note repeat
            # generate new chord, and increse repetition probability
            note_count = math.floor(np.random.normal(3.5, 2, 1))
            if(note_count < 1): note_count = 1
            elif(note_count > 8): note_count = 8
            chord = generate_chord(note_count)
            return [np.array(chord)] + generate_sample(chord_count-1, rep_epsilon+rep_epsilon_decay, rep_epsilon_decay, gap_epsilon+gap_epsilon_decay/2, gap_epsilon_decay, last_chord=chord)

for i in generate_sample(10, 0.65, 0.1, 0.4, 0.1):
    print(i.tolist())

# TODO: create file output geneartion
