import os
import librosa
import asyncio
import numpy as np
import pandas as pd
import scipy.stats as stats
import time
import nest_asyncio
from speechpy import feature

async def extract_MFCCs(count, file):
    async with sem:
        file_path = os.path.join(current_path,"CETUC", "Full", file)
        
        audio_data, sample_rate = librosa.load(file_path)
        
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate)
        
        mfccs_mean = list(np.mean(mfccs.T, axis= 0))
        if file[0] == 'F':
            gender = 0
        if file[0] == 'M': 
            gender = 1
        
        sample_features = mfccs_mean
        sample_features.insert(0,str(file))
        sample_features.append(gender)
        print(sample_features)
        await asyncio.wait(1)
        #print(f"\r{count}/{len(audio_files)}",end='')
        features_list.append(sample_features)
        
        
        return 


# #Calculo de tempo de disparo
start_time = time.time()

#inicio do Loop
loop = asyncio.get_event_loop()

#Controle de requisições por vez
sem = asyncio.Semaphore(600)

#Array de tasks
sents = []

nest_asyncio.apply()

#Coleta as recomendações para envio
gender_list = []
file_list = []
features_list = []


current_path = os.getcwd()
file_path = os.path.join(current_path,"CETUC", "Full")
audio_files = os.listdir(file_path)

for k, file in enumerate(audio_files[:5]):
    sent = asyncio.ensure_future(extract_MFCCs(count=k+1, file=file))
    
    sents.append(sent)
 
done, _ = loop.run_until_complete(asyncio.wait(sents))


dataframe_features = pd.DataFrame(features_list, columns = ['FileName',
                                                            'MFCC_1',  'MFCC_2',  'MFCC_3',  'MFCC_4',  'MFCC_5',
                                                            'MFCC_6',  'MFCC_7',  'MFCC_8',  'MFCC_9',  'MFCC_10',
                                                            'MFCC_11',  'MFCC_12',  'MFCC_13',  'MFCC_14',  'MFCC_15',
                                                            'MFCC_16',  'MFCC_17',  'MFCC_18',  'MFCC_19',  'MFCC_20',
                                                            'Gender'])

dataframe_features.to_csv('data/CETUC_MFCCs_data.csv', index=False)