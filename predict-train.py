# predicting trams audio signals
import pickle
import os
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import load_model
import pandas as pd
from sklearn.metrics import accuracy_score

def build_predictions(audio_dir):
    y_true = []
    y_pred = []
    fn_prob = {} #for softmax probs
    
    print('Extracting features from audio')
    
    for folder in os.listdir(audio_dir):
        if not folder.startswith('.'):
            for subfolder in os.listdir(audio_dir+"/"+folder):
                if not subfolder.startswith('.'):
                    for filename in tqdm(os.listdir(audio_dir+"/"+folder+'/'+subfolder)):
                        if not filename.startswith('.'):
                            
                            rate, wav = wavfile.read(os.path.join(audio_dir, folder, subfolder, filename))
                            label = fn2class[filename]
                            c = classes.index(label)
                            y_prob = []
                            for i in range(0, wav.shape[0]-config.step, config.step):
                                sample = wav[i:i+config.step]
                                x = mfcc(sample, rate, numcep=config.nfeat,
                                         nfilt=config.nfilt, nfft=config.nfft)
                                x = (x - config.min)/(config.max - config.min)
                                
                                if config.mode == 'conv':
                                    x = x.reshape(1, x.shape[0], x.shape[1], 1)
                                elif config.mode == 'time':
                                    x = np.expand_dims(x, axis=0)
                                
                                y_hat = model.predict(x)
                                y_prob.append(y_hat)
                                y_pred.append(np.argmax(y_hat)) #for the max val in y_hat, taking the index
                                y_true.append(c)
            
                            fn_prob[filename] = np.mean(y_prob, axis=0).flatten()
    return y_true, y_pred, fn_prob
    
    '''for fn in tqdm(os.listdir(audio_dir)):
        rate, wav = wavfile.read(os.path.join(audio_dir, fn))
        label = fn2class[fn]
        c = classes.index(label)
        y_prob = []
        
        for i in range(0, wav.shape[0]-config.stop, config.step): #stop????
            sample = wav[i:i+config.step]
            x = mfcc(sample. rate, numcep=config.nfeat,
                     nfilt=config.nfilt, nfft=config.nfft)
            x = (x - config.min)/(config.max - config.min)
            
            if config.mode == 'conv':
                x = x.reshape(1, x.shape[0], x.shape[1], 1)
            elif config.mode == 'time':
                x = np.expand_dims(x, axis=0)
            
            y_hat = model.predict(x)
            y_prob.append(y_hat)
            y_pred.append(np.argmax(y_hat)) #for the max val in y_hat, taking the index
            y_true.apppend(c)
            
        fn_prob[fn] = np.mean(y_prob, axis=0).flatten()
    return y_true, y_pred, fn_prob

print(os.path.join('tram_demo','downsampled','accelerating'))'''


def create_df(dir):
    df_output = pd.DataFrame(columns=['path','label','filename', 'length'])
    i=0
    for folder in os.listdir(dir):
        if not folder.startswith('.'):
            for subfolder in os.listdir(data_dir+"/"+folder):
                if not subfolder.startswith('.'):
                    for filename in tqdm(os.listdir(data_dir+"/"+folder+'/'+subfolder)):
                        if not filename.startswith('.'):
                            #print(i)
                            #print(data_dir+"/"+folder+'/'+subfolder+'/'+filename)
                            rate, signal = wavfile.read(data_dir+"/"+folder+'/'+subfolder+'/'+filename)
                            #print(rate);
                            df_output.loc[i] = [data_dir+"/"+folder+'/'+subfolder+'/'+filename,
                                         folder+'/'+subfolder,
                                         filename,
                                         signal.shape[0]/rate]
                            i+=1
    return df_output
data_dir = os.path.expanduser('~/Documents/neuronsw/ML task/tram_demo/downsampled')
df = create_df(data_dir)
classes = list(np.unique(df.label))
fn2class = dict(zip(df.filename, df.label))
p_path = os.path.join('pickles', 'conv.p')

with open(p_path, 'rb') as handle:
    config = pickle.load(handle)
    
model = load_model(config.model_path)

y_true, y_pred, fn_prob = build_predictions('tram_demo/downsampled')
acc_score = accuracy_score(y_true=y_true, y_pred=y_pred)

y_probs = []
for i, row in df.iterrows():
    y_prob = fn_prob[row.filename]
    y_probs.append(y_prob)
    for c, p in zip(classes, y_prob):
        df.at[i, c] = p
        
y_pred = [classes[np.argmax(y)] for y in y_probs]
df['y_pred'] = y_pred
df.to_csv('predictions.csv', index=False)

