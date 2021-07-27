import os
import librosa
import asyncio
import numpy as np
import pandas as pd
import scipy.stats as stats
import time
import nest_asyncio
from speechpy import feature

async def LMFE(count, file):
    async with sem:
        file_path = os.path.join(current_path,"CETUC", "Full", file)
        if file[0] == 'F':
            gender = 0
        if file[0] == 'M': 
            gender = 1
        audio_data, sample_rate = librosa.load(file_path, sr=16000)
        
        logenergy = feature.lmfe(audio_data, sampling_frequency=sample_rate,
                                    frame_length=0.020, frame_stride=0.01,
                                    num_filters=40, fft_length=512,
                                    low_frequency=0, high_frequency=None)

        features_list.append([file, logenergy, gender])
                                    
        print(f"\r{count}/{len(audio_files)}", end='')

        return


# #Calculo de tempo de disparo
start_time = time.time()

#inicio do Loop
loop = asyncio.get_event_loop()

#Controle de requisições por vez
sem = asyncio.Semaphore(3000)

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

for k, file in enumerate(audio_files[:50000]):
    sent = asyncio.ensure_future(LMFE(count=k+1, file=file))
    #sent = asyncio.create_task(LMFE(count=k+1, file=audio_files))
    
    sents.append(sent)
 
done, _ = loop.run_until_complete(asyncio.wait(sents))


dataframe_features = pd.DataFrame(features_list, columns = ['FileName', 'LMFE', 'Gender'])
dataframe_features.to_csv('data/CETUC_LMFE_data.csv', index=False)





# #Calculo de tempo de disparo
start_time = time.time()

#inicio do Loop
loop = asyncio.get_event_loop()

#Controle de requisições por vez
sem = asyncio.Semaphore(3000)

#Array de tasks
sents = []

nest_asyncio.apply()

#Coleta as recomendações para envio
gender_list = []
file_list = []
features_list = []

for k, file in enumerate(audio_files[50000:]):
    sent = asyncio.ensure_future(LMFE(count=k+1, file=file))
    #sent = asyncio.create_task(LMFE(count=k+1, file=audio_files))
    
    sents.append(sent)
 
done, _ = loop.run_until_complete(asyncio.wait(sents))


dataframe_features = pd.DataFrame(features_list, columns = ['FileName', 'LMFE', 'Gender'])
dataframe_features.to_csv('data/CETUC_LMFE_data_2.csv', index=False)