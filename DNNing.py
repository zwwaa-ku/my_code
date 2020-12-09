#!/usr/bin/env python
# coding: utf-8

# In[85]:


import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv1D, MaxPooling1D, BatchNormalization, GlobalAveragePooling2D
from keras.utils import to_categorical
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import itertools
import csv

# In[8]:


def load_y_data(inputPath='Personality.xlsx'):
    #return nparray: list['Neuroticism (t score)','Extraversion',
    #     'Openness','Agreeableness','Conscienciousness']*'num of people'
    print('[INFO] loading Personality Data, Path: '+ inputPath)
    personality_in1 = pd.read_excel(inputPath)
    personality = personality_in1.drop([0])
    list_personality = personality[['Neuroticism (t score)','Extraversion (t score)',             'Openness (t score)','Agreeableness (t score)','Conscienciousness (t score)']].values.tolist()
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
        upper = int(lock * len(index))
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
    for i in index:
        x_list += [x_b[i]]
    y_b = y_b[index]
    return x_list, y_b, index


# In[87]:


covlist = []
error = []
learning_num = 200
learning_epochs = 400

personality = load_y_data('Personality.xlsx')

# t-scoreを０−１に落とす、平均０.５
max_values = (np.abs(personality[:, 0] - 50)).max()
y_train_before = (personality[:, 0] - 50)/(2*max_values) + 0.5

x_train_before = load_x_data('E4wristband')

print(y_train_before.shape)


# In[7]:


#user 51,44,42,41,34,28,9の削除がしたい
#del x_train_before[49]
#del x_train_before[42]
#del x_train_before[40]
#del x_train_before[39]
#del x_train_before[32]
#del x_train_before[26]
#del x_train_before[7]
#y_train_before = np.delete(y_train_before, [49,42,40,39,32,26,7], 0)


# In[77]:


for learning_count in range(learning_num):
    x_train_before, y_train_before, shuffle_index = data_shuffle(x_train_before, y_train_before)


	# In[78]:



	# print(len(x_train_before))

    x_train,y_train = mk_data(x_train_before[:30], y_train_before[:30], 20000, 5, 0.05)
    x_test, y_test = mk_data(x_train_before[30:], y_train_before[30:], 100000, 1, 1)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)


    # In[79]:


    model = Sequential()
    model.add(Conv1D(filters = 32, kernel_size = 8, input_shape = (x_train.shape[1], 3), activation = 'relu'))
    model.add(Conv1D(32, 4, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(64, 2, activation='relu'))
    model.add(Conv1D(64, 2, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 2, activation='relu'))
    model.add(Conv1D(128, 2, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Flatten())
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='sgd', loss='mse')

    model.summary()
    learning = model.fit(x_train, y_train, epochs=learning_epochs, validation_data = (x_test,y_test))
	#Conv1Dを２層にする、
	#Conv filters ->上位に行くにつれてふやす
	#CNN stride


	# In[88]:


    result = model.predict(x_test).flatten()
    plt.scatter(result[:3], y_test[:3], c ='red')
    plt.scatter(result[3:6], y_test[3:6], c='yellow')
    plt.scatter(result[6:9], y_test[6:9], c='blue')
    plt.scatter(result[9:12], y_test[9:12], c='green')
    plt.scatter(result[12:15], y_test[12:15], c='black')
    plt.scatter(result[15:18], y_test[15:18], c='orange')
    plt.scatter(result[18:], y_test[18:], c='grey')

    plt.xlabel('predict')
    plt.ylabel('y_test')
    plt.show()
    cov = np.corrcoef(result, y_test)
    print(cov[0][1])
    covlist += [[cov[0][1]]]


	# In[81]:


	#print(shuffle_index[30:])
	#print(result)
	#print(result-y_test)


	# In[89]:



    for k in range(21):
        error += [[shuffle_index[30+k], result[k],result[k]-y_test[k]]]

	#print(error)
    print('試行回数' + str(len(error)/21))


	# In[90]:


def ave_error(error, total = 51):
    count = np.zeros(total)
    sum_e = np.zeros(total)
    for [num, value, er] in error:
        count[num] += 1
        sum_e[num] += abs(er)
    for k in range(total):
        if count[k] != 0:
            count[k] = sum_e[k] / count[k]
            
    return count,sum_e

ave, sum_e = ave_error(error)
print(ave)
storage = error[:]
#print(y_train_before)


# In[26]:



#tresult = model.predict(x_train).flatten()
#plt.scatter(tresult, y_train)
#plt.show()
#print(np.corrcoef(tresult, y_train))


# In[28]:


#plt.plot(range(1, 25+1), learning.history['loss'], label="training")
#plt.plot(range(1, 25+1), learning.history['val_loss'], label="validation")
#plt.xlabel('Epochs')
#plt.ylabel('loss')
#plt.legend()
#plt.show()


# In[ ]:


np.savetxt('data/average.csv', ave, delimiter=",")
print(type(covlist[0]))    
with open('data/cov.csv', 'w') as j:
    writer = csv.writer(j)
    writer.writerows(covlist)

