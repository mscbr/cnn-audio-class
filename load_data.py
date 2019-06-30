from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
#pip install python_speach_features
from python_speech_features import mfcc, logfbank
import librosa

def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            if i>8:
                return
            else:
                #print(x,y,i)
                axes[x,y].set_title(list(signals.keys())[i])
                axes[x,y].plot(list(signals.values())[i])
                axes[x,y].get_xaxis().set_visible(False)
                axes[x,y].get_yaxis().set_visible(False)
                #print(i)
                #print(list(signals.keys())[i])
                i += 1

#plot_signals(signals)
#plt.show()


def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            if i>8:
                return
            else:
                data = list(fft.values())[i]
                Y, freq = data[0], data[1]
                axes[x,y].set_title(list(fft.keys())[i])
                axes[x,y].plot(freq, Y)
                axes[x,y].get_xaxis().set_visible(False)
                axes[x,y].get_yaxis().set_visible(False)
                i += 1

def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            if i>8:
                return
            else:
                axes[x,y].set_title(list(fbank.keys())[i])
                axes[x,y].imshow(list(fbank.values())[i],
                        cmap='hot', interpolation='nearest')
                axes[x,y].get_xaxis().set_visible(False)
                axes[x,y].get_yaxis().set_visible(False)
                i += 1

def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            if i>8:
                return
            else:
                axes[x,y].set_title(list(mfccs.keys())[i])
                axes[x,y].imshow(list(mfccs.values())[i],
                        cmap='hot', interpolation='nearest')
                axes[x,y].get_xaxis().set_visible(False)
                axes[x,y].get_yaxis().set_visible(False)
                i += 1
import os
import pandas as pd
import numpy as np

def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return (Y, freq)

data_dir = os.path.expanduser('~/Documents/neuronsw/ML task/tram_demo/dataset')


def create_df(dir):
    df_output = pd.DataFrame(columns=['path','label','filename', 'length'])
    i=0
    for folder in os.listdir(dir):
        if not folder.startswith('.'):
            for subfolder in os.listdir(data_dir+"/"+folder):
                if not subfolder.startswith('.'):
                    for filename in os.listdir(data_dir+"/"+folder+'/'+subfolder):
                        print(i)
                        rate, signal = wavfile.read(data_dir+"/"+folder+'/'+subfolder+'/'+filename)
                        print(rate);
                        df_output.loc[i] = [data_dir+"/"+folder+'/'+subfolder+'/'+filename,
                                     folder+'/'+subfolder,
                                     filename,
                                     signal.shape[0]/rate]
                        i+=1
    return df_output
                        

df = create_df(data_dir)

classes = list(np.unique(df['label']))
class_dist = df.groupby(['label'])['length'].mean()

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%', shadow=False, startangle=90)
ax.axis('equal')
plt.show()
#df.reset_index(inplace=True)
signals = {}
fft = {}
fbank = {}
mfccs = {}

for c in classes:
    wav_file = df[df.label == c].iloc[0,0]
    signal, rate = librosa.load(wav_file, sr=22050)
    signals[c] = signal
    fft[c] = calc_fft(signal, rate)
    
    bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1024).T
    fbank[c] = bank
    mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1024).T
    mfccs[c] = mel
    
plot_signals(signals)
plt.show()
plot_fft(fft)
plt.show()
plot_fbank(fbank)
plt.show()
plot_mfccs(mfccs)
plt.show()

def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean>threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


i = 0   
for f in tqdm(df['path']):
    signal, rate = librosa.load(f, sr=16000)
    #mask = envelope(signal, rate, 0.0005)
    wavfile.write(filename='tram_demo/downsampled/'+df.iloc[i,1]+'/'+df.iloc[i,2], rate=rate, data=signal)
    i+=1
    