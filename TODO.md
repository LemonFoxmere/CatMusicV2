# Hello! I was not expecting you here. This is where I keep my dev logs and stuff.
---

## **Current TODOs**
<sub>NS = Not Started; IP = In Progress; I = Issues; C = Completed; CNT = Completed but needs tweak; RN = Revision Needed; IRP = In Research Progress</sub>
| TASK | COMPLETION |
| :-: | :-: |
Create RawMid Generation Algorithm | C
Create RawMid to MIDI Parser | C
Create MIDI Synthesizing Algorithm | C
Create MIDI to WAV converter | C
Test Synchronizer | C
Create WAV Parser for Neural Net | C
Create Data Synchronizer | C
Create Training Dataset Matching Algorithm | C
Create Multi-hot Encoder | C
Create Data Partitioning System | C
Create Neural Network Architecture | RN
Create BPTT Algorithm | C
Create Data Feed in algorithm | RN
Create Early Stopping System | IP
Create Loss Tracker | C
Create [Accuracy Tracker*](https://stats.stackexchange.com/questions/12702/what-are-the-measure-for-accuracy-of-multilabel-data#168952) | NS
Create Training Process | C
Create Batch Generation | IP
Create Batch Testing Environment | NS
Implement Sample expansion if necessary | NS
Test Training Effectiveness | NS
Test Down Sampling effect | NS
Test Data Efficiency on Training | NS

last updated: 5-7-2021*

## Dev Notes:
* Trim all data to 88 notes instead of the traditional 128
* Continue experimenting using BLSTM and LSTM, and comparing their effectiveness
* Normalize input in range [-1, 1]
* Make absolute sure that the rawMidis are trimmed before Training
    * Timidity++ Synthesizer does not take into account of the initial 0s
    * Thus if we use data that does not start immediatly, there WILL be synchronization problems, which lead to the neural network under performing.
    * Check [FIGURE_1.png*](https://lemonorangewastaken.github.io/CatMusicV2/references/graphs/Figure_1.png) and [FIGURE_2.png*](https://lemonorangewastaken.github.io/CatMusicV2/references/graphs/Figure_2.png) for more details
* Down sampling the WAV input may result in better abstraction and classification; Need tests.
* When MID ends, any reverb/sustain that Timitdity++ will be cut off, as it does not contribute to training as much as it should; Need tests.
* As of writing, 99.4% will be training data, 0.5% testing data, 0.1% validation data
* Things like early stopping and loss + accuracy tracking must also be implemented
* IMPORTANT: All generated MIDs are ranged from 0-87 (length 88), while the generation bumps that number up to 21-98 (length 88) within the domain of 0-127 (length 128) as per MIDI standards.
    * BUG
    * RawMid generation algorithm only produced 87 usable notes, with the note `0` being used to indicate silence.
    * However due to development during 3AM again, the CSV generation checks if the first element of the chord is `0` to see if it represents silence or note
    * This means that any chord starting with `0` will be considered silent, even if it is not. The chance of it occuring is not big but it is there
    * Therefore, during the generation of the multi-hot labeling, precaution must be taken to see if the chord starts with `0` or not
        * If it does, discard the chord and return a blank vector, keep otherwise. THE CSV GENERATOR ONLY LOOKS AT THE FIRST NOTE IN EACH CHORD FOR DETERMINATION
        * SILENCE MUST BE REPRESENTED WITH A BLANK VECTOR, AS THE NOTE 0 IN MOST CASES WILL STILL REPRESENT A NOTE INSTEAD OF A SILENCE.
    * stupid design choice I know, but I've gone so far into this without checking I might as well just go forward with this. The training set will not be severely affected from this, but it will act as a slight annoyance.

---
## Overall Plan (V2):
#### Data Generation (Training)
* Generate multiple raw midis (RAWMID) chords with a bunch of normal distributions and probability stuff.
    * This is the output data. Should be labeled by unique sequential numbers
    * Output naming convention: `outputName.rawmid`
    * Ex: `48.rawmid`
* Use generated RAWMID and parse them into midcsvs
* Use parsed midcsvs and convert them to midi
* Using the converted midi, synthesize piano / instrumental notes based on the midis, producing a WAV file (input data)
    * In case of multiple synthesizes (multiple piano/instrument types)
        * Output naming convention: `outputName_{instrumentID}.wav`
        * Ex: `48_2.wav`
        * This way we can just split the data among the `_` and get its corresponding
* In case of 88 notes total, with 8 keys being the max, there are around 70 billion combination of chords possible
    * If there are 5 chord in each sample set, there will be a total of 350 billion types of possible combination
    * With the use of normal distributions and other probability reduction, we can get the common ones
#### Neural Network Architecture
* This model (as of time of writing) will be consisting of 4 Bi-LSTM layers, and 3 Dense layers
    * Each of the layers will be RELU activation, except for the last Dense layer, which will have SIGMOID instead.
    * As this can be treated as a Multi-Label classification problem, but only adding a temporal dimension, the relatively small but accurate data set should result in optimal classification.
    * The network will be trained with standard BPTT, binary_CSE for the loss, as well as Adam for the optimizer. (All as of time of writing)

#### Sound Data Makeups
* Standard WAV (synthesized by timidity) files are sampled at 44100hz
* Assuming that the standard music tempo is 120bpm
    * A quarter-note will be 1/4 of a whole note, which corresponds to 500ms
    * With the same logic, 1/8 notes will be 250ms, and 1/16 notes will be only 125ms
    * Assuming that our smallest MIDI note sampling rate is 125ms
        * If generating 5 notes at that duration, assuming no pauses, will result in a 625ms duration sample.
        * For the synthesized wav that means 27562.5 samples.
        * This is problematic, as half a sample will not make sense.
    * SOLUTION: Decrease WAV sampling rate to 44000hz
        * While we do lose a little bit of data in the process, it can be evenly spitted to even 31.25ms
* THEREFORE
    * At 120BPM, each MIDI sample will last 125ms, which is 1/16 of a note
    * Because the WAV SR is a nice split-able number, we can scale the repetition for each note up or down at will
* Gaps within the music are also necessary. Each gap should be at least 250ms, which is the equivalence of a silent 1/8 note, or 11000 WAV samples
* This should result in a perfect synchronization between the WAV and RAWMIDI data

#### Data Batching and Training
* As discussed earlier, the shortest possible note duration is 125ms, or 5500 samples
    * Therefore, it is unnecessary to group data together smaller than that section
    * TODO: Test how well margin overlapping will work within training, and how classifications are effected.

#### Synchronization of rawMid and WAV
* Absolute time is based on 1/16th of a MIDI clock. Calculation are as follow:
    * clock_duration (seconds) = ((tempo/1,000,000)/24)/16
    * clock_passage_duration (seconds) = clock_period * clock_repetition
* Make absolute sure that the rawMidis are trimmed before Training
    * Timidity++ Synthesizer does not take into account of the initial 0s
    * Thus if we use data that does not start immediatly, there WILL be synchronization problems, which lead to the neural network under performing.
    * Check [FIGURE_1.png*](https://lemonorangewastaken.github.io/CatMusicV2/references/graphs/Figure_1.png) for more details
* REMINDER
    * Reverb may also be needed for consideration as sample offsets
        * So instead of taking in 5500 samples (125ms) per batch, we will take in for example 6500 samples (147.39ms)

## End of Overall Plan
---
### Why Did I Expect This to Work?
During an over-fitting test, I discovered that, even though purposely over-fitted, the over-fitted model was able to correctly classify some other data sample that it has never seen before. Its ability to classify unseen data to such an extent shows first, its capability for accurate classification, and second, a more discrete classification problem rather than a synthesization problem.

### Journals:
#### Research Journal 3-29 (V1):
Experiments are done on the effectiveness of loss functions on large onehot-encoded datasets
On average, the binary_crossentropy loss function performed poorly on most occasions, but can be used as an accurate representation of the deviation of the output to the ground truth. The softmax_cross_entropy_with_logits, as well as categorical_crossentropy performed similarly, with the categorical_crossentropy having a noticable 50% decrease in loss in smaller output to ground truth deviations. A new loss function was then created in the attempt to combine the benefits of both loss functions. The new function is depicted as follow:

Assume that the output vector generated by CATCSE is x, and BINCSE is y, and a constant C which will act as a scaler vector
new_loss = ((x*c)*y)
Optimal scaler has not been tested yet, although optimal scaler seems to be close to the constant E.

This function attempts to scale down the CATCSE values as the deviation gets smaller, which in a way, the BINCSE is controlling the overall loss at smaller deviations. However, at larger deviations, under normal circumstances, BINCE approaches close to 0.6-0.7. And after the application of the constant scaler, acts as a new scaler vector of 1.2-1.4 for the CATCSE loss, which can more accuratly represent the overall deviation fron the ground truth.

#### Research Journal 4-10 (V1):
Stacked LSTM models are created to improve efficiency, and a basic sense of rhythm and composition is detected. The suspecting failure point is the poor matching algorithm chosen for the generated output set and ground truth data set. The custom loss is abandoned for its inefficiency in classifying larger sequential multi-label datasets, and instead binary_crossentropy is used. Softmax as the output activation is also abandoned in replace of sigmoid, which is not based on a probabilistic model, thus being more efficient in multi-label classification.


---
For all references on my research and stuff, look in the `references` folder.\
This is a continuation of CatMusicV1. You can check out the deprecated repo [here](https://github.com/LemonOrangeWasTaken/CatMusic).\
*Copyright (c) 2021 Lemon Orange*
