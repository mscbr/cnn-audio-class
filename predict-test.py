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
import librosa

def build_predictions(file):
    y_true = []
    y_pred = []
    fn_prob = {} #for softmax probs
    y_prob = []
    print('Extracting features from audio')
    wav, rate = librosa.load(file, sr=16000)
    #rate, wav = wavfile.read(file)
    
    for i in range(0, wav.shape[0]-config.step, config.step*10): #'x10' for fewer samplepoints 
        sample = wav[i:i+config.step]
        x = mfcc(sample, rate, numcep=config.nfeat,
                 nfilt=config.nfilt, nfft=config.nfft)
        
        #normalization
        x = (x - config.min)/(config.max - config.min)
        #reshaping for CNN purposes + grayscale layer
        x = x.reshape(1, x.shape[0], x.shape[1], 1)
        
        y_hat = model.predict(x)
        y_prob.append(y_hat)
        y_pred.append(np.argmax(y_hat)) #for the max val in y_hat, taking the index
        #y_true.apppend(c)
        fn_prob[i] = np.mean(y_prob, axis=0).flatten()
    #fn_prob[fn] = np.mean(y_prob, axis=0).flatten()
    return y_true, y_pred, fn_prob
    
    
    #it will be only one audiofile - do not need to iterate over files
    '''for fn in tqdm(os.listdir(audio_dir)):
        rate, wav = wavfile.read(os.path.join(audio_dir, fn))
        #label = fn2class[fn]
        #c = classes.index(label)
        y_prob = []
        
        for i in range(0, wav.shape[0]-config.step, config.step*10): #'x10' for fewer samplepoints 
            sample = wav[i:i+config.step]
            x = mfcc(sample, rate, numcep=config.nfeat,
                     nfilt=config.nfilt, nfft=config.nfft)
            
            #normalization
            x = (x - config.min)/(config.max - config.min)
            #reshaping for CNN purposes + grayscale layer
            x = x.reshape(1, x.shape[0], x.shape[1], 1)
            
            y_hat = model.predict(x)
            y_prob.append(y_hat)
            y_pred.append(np.argmax(y_hat)) #for the max val in y_hat, taking the index
            #y_true.apppend(c)
            
        fn_prob[fn] = np.mean(y_prob, axis=0).flatten()
    return y_true, y_pred, fn_prob'''

#TEST VALUES
rate, wav = wavfile.read(os.path.join('tram_demo', 'test_files', 'tram-2018-11-30-15-30-17.wav'))
print(rate)
print(config.step)
print(wav.shape[0]/(config.step*10))
'''#EXPRESSION TEST
label = fn2class['tram-2018-11-17-14-20-54_63.40_66.80.mp4.wav']
c = classes.index(label)'''

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

classes = ['accelerating/1_New', 'accelerating/2_CKD_Long', 'accelerating/3_CKD_Short', 'accelerating/4_Old', 
           'braking/1_New', 'braking/2_CKD_Long', 'braking/3_CKD_Short', 'braking/4_Old', 'negative/checked']
with open(p_path, 'rb') as handle:
    config = pickle.load(handle)
    
model = load_model(config.model_path)

y_true, y_pred, fn_prob = build_predictions('tram_demo/test_files/tram-2018-11-30-15-30-17.wav')
#acc_score = accuracy_score(y_true=y_true, y_pred=y_pred)

y_probs = []
for i, row in df.iterrows():
    y_prob = fn_prob[row.filename]
    y_probs.append(y_prob)
    for c, p in zip(classes, y_prob):
        df.at[i, c] = p
        
y_pred = [classes[np.argmax(y)] for y in y_probs]
df['y_pred'] = y_pred
df.to_csv('predictions.csv', index=False)

