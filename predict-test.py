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
    
    for i in tqdm(range(0, wav.shape[0]-config.step, config.step)): #'x10' for fewer samplepoints 
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
    


p_path = os.path.join('pickles', 'conv.p')

classes = ['accelerating_1_New', 'accelerating_2_CKD_Long', 'accelerating_3_CKD_Short', 'accelerating_4_Old', 
           'braking_1_New', 'braking_2_CKD_Long', 'braking_3_CKD_Short', 'braking_4_Old', 'negative']
with open(p_path, 'rb') as handle:
    config = pickle.load(handle)
    
model = load_model('models/cnn_model1.model')

y_true, y_pred, fn_prob = build_predictions('tram_demo/test_files/tram-2018-12-07-15-32-08.wav')
#acc_score = accuracy_score(y_true=y_true, y_pred=y_pred)

df = pd.DataFrame.from_dict(fn_prob, orient='index')


df = pd.read_csv('df_structure.csv')
seconds_offset = []
for key in fn_prob:
    seconds_offset.append(key)
df["seconds_offset"] = seconds_offset
df.set_index('seconds_offset', inplace=True)
y_probs = []
for key in fn_prob:
    y_prob = fn_prob[key]
    y_probs.append(y_prob)
    for c, p in zip(classes, y_prob):
       df.at[key, c] = p

      
y_pred = [classes[np.argmax(y)] for y in y_probs]
df['y_pred'] = y_pred
#df.to_csv('predictions.csv', index=False)

