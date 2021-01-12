#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv1D, MaxPooling1D, BatchNormalization, GlobalAveragePooling2D
from keras.utils import to_categorical
from keras.models import model_from_json
import pandas as pd
import sys
import os
import itertools
import csv
from keras import backend as K
import gc

# In[2]:


def load_y_data(inputPath='Personality.xlsx'):
    #return nparray: list['Neuroticism (t score)','Extraversion',
    #     'Openness','Agreeableness','Conscienciousness']*'num of people'
    print('[INFO] loading Personality Data, Path: '+ inputPath)
    personality_in1 = pd.read_excel(inputPath)
    personality = personality_in1.drop([0])
    list_personality = personality[['Neuroticism (t score)','Extraversion (t score)',             'Openness (t score)','Agreeableness (t score)','Conscienciousness (t score)','Sex']].values.tolist()
    nparray_personality = np.array(list_personality)
    return nparray_personality

def load_x_data(inputPath):
    print('[INFO] loading Acc Data, Path: '+ inputPath)
    x_data = []
    for u in range(2,53):
        acc = pd.read_csv(os.path.join(inputPath,f'{u}','ACC.csv'))
        #acc[0:1]はHzを表しているので除く
        x_data += [acc[1:]]
    #返り値：list、中がDataFrame
    return x_data

def split_data(data, size):
    return_list = []
    block_num = len(data) // size
    for k in range(0,block_num):
        return_list += [np.array(data[k*size:(k+1)*size])]
    #返り値：nparray
    return np.array(return_list)

def mk_data(x, y, size = 10000, number = 3, lock = 1):
    #x,y -> load_x_data,load_y_data
    xlist = []
    ylist = []
    for k in range(0,len(x)):
        if len(x[k]) < size * number:
            print('ValueError: x[k] must be more than size*number')
            break
        splited = split_data(x[k], size)
        index = list(itertools.permutations(range(splited.shape[0]), number))
        random.shuffle(index)
        upper = lock
        #print(upper)
        for l in index[:upper]:
            onedata = splited[list(l)]
            xlist += [onedata.reshape(size*number,3)]
            ylist += [y[k]]
    re_x = np.array(xlist)
    re_y = np.array(ylist)
    return re_x, re_y

def data_shuffle(x_b, y_b):
    x_list = []
    index = np.random.permutation(len(x_b))
#    index =[29,43,7,24,26,40,12,47,20,16,49,14,41,15,48,10,19,42,13,1,6,8,4,25,23,35,34,2,21,0,31,46,32,30,18,17,45,36,37,50,11,3,38,9,5,39,44,33,22,28,27]
    for i in index:
        x_list += [x_b[i]]
    y_b = y_b[index]
    return x_list, y_b, index


# In[3]:


#var
counter = 300
fivefactor = 0
#0:Nt, 1:Et, 2:Ot, 3:At, 4:Ct
split_use_num = 5
epoch=300

#finalvar
covlist = []
traindata_size = 100000

split = int(traindata_size/split_use_num)
if traindata_size%split_use_num != 0:
    print('split_use_num was individabl')

#preprocess
personality = load_y_data('../Personality_v2.xlsx')
#print(personality)

directoryname = {0:'Nt', 1:'Et', 2:'Ot', 3:'At', 4:'Ct'}
# t-scoreを０−１に落とす、平均０.５
max_values = (np.abs(personality[:, fivefactor] - 50)).max()
y_train_before = (personality[:, fivefactor] - 50)/(2*max_values) + 0.5

x_train_before = load_x_data('../E4wristband')
#print(y_train_before.shape)

indexlist = np.empty((counter, y_train_before.shape[0]))


# In[ ]:


shuffle_index = np.empty(y_train_before.shape[0])
boader=30
x_train=np.empty((boader,traindata_size,3))
y_train=np.empty(boader)
x_test=np.empty(((y_train_before.shape[0]-boader),traindata_size,3))
y_test=np.empty(y_train_before.shape[0]-boader)
for repeat in range(counter):
    x_all, y_all, shuffle_index = data_shuffle(x_train_before[:], y_train_before[:])

    
    x_train,y_train = mk_data(x_all[:boader], y_all[:boader], split, split_use_num, 30)
    x_test, y_test = mk_data(x_all[boader:], y_all[boader:], traindata_size, 1, 1)

    model = Sequential()
    model.add(Conv1D(filters = 32, kernel_size = 8, padding="same", input_shape = (x_train.shape[1], 3), activation = 'relu', name='conv1d_0'))
    model.add(Conv1D(32, 4, padding="same", activation='relu', name='conv1d_1'))
    model.add(MaxPooling1D(3, name='max_pooling1d_0'))
    model.add(Conv1D(64, 2, padding="same", activation='relu', name='conv1d_2'))
    model.add(Conv1D(64, 2, padding="same", activation='relu', name='conv1d_3'))
    model.add(MaxPooling1D(3, name='max_pooling1d_1'))
    model.add(Conv1D(128, 2, padding="same", activation='relu', name='conv1d_4'))
    model.add(Conv1D(128, 2, padding="same", activation='relu', name='conv1d_5'))
    model.add(MaxPooling1D(3, name='max_pooling1d_2'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(100, activation='sigmoid', name='dense_0'))
    model.add(Dense(1, activation='sigmoid', name='dense_1'))
    model.compile(optimizer='sgd', loss='mse')

    model.summary()
    learning = model.fit(x_train, y_train, epochs=epoch, validation_data = (x_test,y_test))

    print('finish '+str(repeat)+'times learning')
    result = model.predict(x_test).flatten()
    cov = np.corrcoef(result, y_test)
    print('共分散 : '+ str(cov[0][1]))
    covlist += [cov[0][1]]
    
    indexlist[repeat] = shuffle_index[:]
    
    if repeat==0:
        jsonfilename=directoryname[fivefactor]+'_cnn_model.json'
        json_string = model.to_json()
        open(os.path.join('../data',directoryname[fivefactor],jsonfilename), 'w').write(json_string)
    
    hdf5filename=str(repeat)+'cnn_model_weights.hdf5'
    model.save_weights(os.path.join('../data',directoryname[fivefactor],hdf5filename))
    
    del model
    K.clear_session()
    #del x_test
    #del y_test
    #del x_train
    #del y_train
    del x_all
    del y_all
    gc.collect()

# In[17]:


print(covlist)
print(indexlist)
print('save convlist and indexlist')
#print(type(indexlist))


# In[ ]:


with open('../data/'+ directoryname[fivefactor]+'/cov.csv', 'w') as j:
    writer = csv.writer(j)
    writer.writerows(covlist)
np.savetxt('..data/'+directoryname[fivefactor]+'/indexlist.csv', indexlist, delimiter=",", fmt='%d')

