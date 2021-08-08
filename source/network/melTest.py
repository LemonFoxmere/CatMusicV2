import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('./source/network')
from Dataloader import datasync as sync
from Dataloader import loader
import math
from scipy import signal

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

absolute_path = '/home/lemonorange/catRemixV2'
input_data_path = os.path.join(absolute_path, 'data/wav')

y.shape

y, sr = librosa.load(os.path.join(input_data_path, '0_0.wav'), sr=None)

print(y.shape, sr)

# display audio waveform
plt.title('Original waveform (sr=44000)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.plot(y)
plt.show()

# create fft representation
n_fft = 2048 # from what I understand this is the sample window size
librosa.display.waveplot(y[:2048], sr=sr) # display the timeframe
D = np.abs(librosa.stft(y[:n_fft], n_fft=n_fft, hop_length=n_fft+1))
plt.plot(D);

np.max(D)

high_pass = lambda x : -0.964**(x-(math.log(1, (1/0.964)))-37)+1 if x >= 37 else 0

# WITHOUT HIGHPASS
hop_length = 512
D = np.abs(librosa.stft(y, n_fft=n_fft,  hop_length=hop_length)) # in default this uses a hann window.
DB = librosa.amplitude_to_db(D, ref=np.max) # convert amp to db
librosa.display.specshow(loader.normalize(DB+80, 1)-1, sr=sr, x_axis='time', y_axis='log'); # show frequency on log scale
plt.colorbar(format='%+2.0f dB');

# hop length is how much the time window should move every process
# WITH HIGHPASS
hop_length = 512
D = np.abs(librosa.stft(butter_highpass_filter(y, 120, sr), n_fft=n_fft,  hop_length=hop_length)) # in default this uses a hann window.
DB = librosa.amplitude_to_db(D, ref=np.max) # convert amp to db
librosa.display.specshow(loader.normalize(DB+80, 1), sr=sr, x_axis='time', y_axis='log'); # show frequency on log scale
plt.colorbar(format='%+2.0f dB');

DB = loader.normalize(DB+80, 1)

# display mel scale
n_mels = 128 # this is the bin resolution
mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
# create mel scale

plt.figure(figsize=(15, 4));

# display the different mel filers
plt.subplot(1, 3, 1);
librosa.display.specshow(mel, sr=sr, hop_length=hop_length, x_axis='linear');
plt.ylabel('Mel filter');
plt.colorbar();
plt.title('1. Our filter bank for converting from Hz to mels.');

plt.subplot(1, 3, 2);
mel_10 = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=10)
librosa.display.specshow(mel_10, sr=sr, hop_length=hop_length, x_axis='linear');
plt.ylabel('Mel filter');
plt.colorbar();
plt.title('2. Easier to see what is happening with only 10 mels.');

plt.subplot(1, 3, 3);
idxs_to_plot = [0, 9, 49, 99, 127]
for i in idxs_to_plot:
    plt.plot(mel[i]);
plt.legend(labels=[f'{i+1}' for i in idxs_to_plot]);
plt.title('3. Plotting some triangular filters separately.');

plt.tight_layout();

DB.shape

plt.figure(figsize=(15, 4));

plt.subplot(1,2,1)
plt.imshow(DB[:n_mels]);
plt.title('BEFORE APPLYING MEL');
plt.subplot(1,2,2)
plt.imshow(mel.dot(DB));
plt.title('AFTER APPLYING MEL');

n_mels = 128
S = librosa.feature.melspectrogram(butter_highpass_filter(y, 120, sr), sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels) # create mel spectrogram
S_DB = librosa.power_to_db(S, ref=np.max) # convert amplitube to db
librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')

S_DB = loader.normalize(S_DB+80, 1)

librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')

plt.colorbar(format='%+2.0f dB');

librosa.display.specshow(G, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel');
plt.colorbar(format='%+2.0f dB');

S_DB.transpose().shape

fig = plt.figure(figsize=(6, 3), dpi=150)
plt.title('Transposed MEL spectrogram sample (512 hop length; 2048 sampling res; han windowing)')
plt.xlabel('MEL scale (512 bins)')
plt.ylabel('Time')
plt.imshow(S_DB.transpose())
plt.colorbar(format='%+2.0f dB');
plt.show()
