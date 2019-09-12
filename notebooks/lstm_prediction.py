#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import dateutil
from datetime import datetime
import matplotlib.dates as md
import scipy as sp
from scipy import signal
import scipy.signal as signal
import re
from datetime import timedelta 
import string
import pytz
from matplotlib.ticker import FormatStrFormatter
import os
from pandas.plotting import lag_plot
from matplotlib.patches import Rectangle
from scipy import fftpack
from scipy import fft, arange

from PIL import Image
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import preprocessing
from sklearn.preprocessing import normalize
import tensorflow as tf
from tensorflow import keras
import sys
import uuid
from skimage import transform
import datetime

from tensorflow.keras.layers import GlobalAveragePooling2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Input, Dropout, Conv2D, BatchNormalization, MaxPool2D, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam,  RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import MaxPooling1D, Conv1D, GlobalAveragePooling1D, Reshape

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
plt.style.use('default')

#%%

data_dir = './data/raw/'
fname = os.path.join(data_dir, 'neom-data.csv')

df = pd.read_csv(fname, header = 0, error_bad_lines=False)
#df = df.rename(columns={'Unnamed: 0': 'date'})
df.columns = df.columns.str.replace('Unnamed: 0','date')
#df.set_index('date',inplace=True)

#%%
df

#%%
# values = df.values
# # specify columns to plot
# groups = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# i = 1
# # plot each column
# plt.figure(figsize=(30,20))
# for group in groups:
# 	plt.subplot(len(groups), 1, i)
# 	plt.plot(values[:, group])
# 	plt.title(df.columns[group], y=0.5, loc='right')
# 	i += 1
# plt.show()

#%%
df.head(1)
#%%
df.shape
#%%


float_data = df.drop(df.columns[0], axis=1)

float_data = float_data.as_matrix()


#%%
float_data
#%%

mean = float_data[:48000].mean(axis=0)
float_data -= mean
std = float_data[:48000].std(axis=0)
float_data /= std

#%%

#test on temperature only
float_data = float_data["t2(C)"]
#%%
def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets

#%%
#we have 1 value each hour
#loopback = 240 mean our observation will go back 10 days
lookback = 240
step = 1 # our observations will be sampled at one data point per hour.
delay = 24 # i.e. our targets will be 24 hours in the future.
batch_size = 128

train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=48000,
                      shuffle=True,
                      step=step, 
                      batch_size=batch_size)
val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=48001,
                    max_index=90000,
                    step=step,
                    batch_size=batch_size)
test_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=90001,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

# This is how many steps to draw from `val_gen`
# in order to see the whole validation set:
val_steps = (90000 - 48001 - lookback) // batch_size

# This is how many steps to draw from `test_gen`
# in order to see the whole test set:
test_steps = (len(float_data) - 90001 - lookback) // batch_size

#%%
def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))
    return np.mean(batch_maes)
    
naive_method = evaluate_naive_method()

#%%
naive_method_celcius = naive_method * std[1]

print(naive_method_celcius)

#%%
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(32,
                     dropout=0.2,
                     recurrent_dropout=0.2,
                     input_shape=(None, float_data.shape[-1])))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=40,
                              validation_data=val_gen,
                              validation_steps=100)

model.save("lstm32_500epoch.h5")
                           

#%%
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

#%%
