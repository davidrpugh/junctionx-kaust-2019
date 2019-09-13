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
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y %H:%M')
df['Day'] = df["date"].dt.dayofyear
df['Hour'] = df["date"].dt.hour
#df.set_index('date',inplace=True)

#%%
df

#%%
oneYear = df[(df["date"]>= pd.to_datetime('01/01/2016')) ]
oneYear = oneYear[(oneYear["date"]< pd.to_datetime('01/01/2017'))]
#%%
oneYear["date"].tail(2)



#%%
values = oneYear.values
# specify columns to plot
groups = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
i = 1
# plot each column
plt.figure(figsize=(30,20))
for group in groups:
	plt.subplot(len(groups), 1, i)
	plt.plot(values[:, group])
	plt.title(df.columns[group], y=0.5, loc='right')
	i += 1
plt.show()

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

#model.save("lstm32_40epoch.h5")
                           

#%%
#  ['date',
#  'mslp(hPa)',
#  't2(C)',
#  'td2(C)',
#  'wind_speed(m/s)',
#  'wind_dir(Deg)',
#  'rh(%)',
#  'GHI(W/m2)',
#  'SWDIR(W/m2)',
#  'SWDNI(W/m2)',
#  'SWDIF(W/m2)',
#  'rain(mm)',
#  'AOD']

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
oneYearTestSize = oneYear.shape[0]

oneYear_no_nrj_X = oneYear[[
 'mslp(hPa)',
 't2(C)',
 'td2(C)',
 'wind_speed(m/s)',
 'wind_dir(Deg)',
 'rh(%)',
 'rain(mm)',
 'AOD']]

oneYear_no_nrj_Y = oneYear['SWDNI(W/m2)']

float_data_X = oneYear_no_nrj_X.as_matrix()
mean_X = float_data_X.mean(axis=0)
float_data_X -= mean_X
std_X = float_data_X.std(axis=0)
float_data_X /= std_X

float_data_Y = oneYear_no_nrj_Y.as_matrix()
mean_Y = float_data_Y.mean(axis=0)
float_data_Y -= mean_Y
std_Y = float_data_Y.std(axis=0)
float_data_Y /= std_Y


#%%

X = df[[
 'mslp(hPa)',
 't2(C)',
 'td2(C)',
 'wind_speed(m/s)',
 'wind_dir(Deg)',
 'rh(%)',
 'rain(mm)',
 'AOD']]


min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)


Y = df['SWDNI(W/m2)']

X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)

X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)





#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(32, activation='relu', input_shape=(8,)),
    Dense(32, activation='relu'),
    Dense(1),
])
model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['mae'])

#%%
hist = model.fit(X_train, Y_train,
          batch_size=32, epochs=100,
          validation_data=(X_val, Y_val))

#%%
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

#%%
#model.save("SWDNI_dense_100epoch.h5")


#%%
train_names = [
 'SWDNI(W/m2)',
 'mslp(hPa)',
 't2(C)',
 'td2(C)',
 'wind_speed(m/s)',
 'wind_dir(Deg)',
 'rh(%)',
 'rain(mm)',
 'AOD',
 'Day',
 'Hour']
 

target_names = ['SWDNI(W/m2)']

shift_days = 1
shift_steps = shift_days * 24  # Number of hours.

#%%
df_targets = df[target_names].shift(-shift_steps)


#%%
x_data = df[train_names].values[0:-shift_steps]


#%%
y_data = df_targets.values[:-shift_steps]

#%%
num_data = len(x_data)
train_split = 0.9
num_train = int(train_split * num_data)
num_test = num_data - num_train
x_train = x_data[0:num_train]
x_test = x_data[num_train:]
y_train = y_data[0:num_train]
y_test = y_data[num_train:]
num_x_signals = x_data.shape[1]
num_y_signals = y_data.shape[1]
x_scaler = preprocessing.MinMaxScaler()
x_train_scaled = x_scaler.fit_transform(x_train)
x_test_scaled = x_scaler.transform(x_test)
y_scaler = preprocessing.MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)
#%%
num_y_signals
#%%
def batch_generator(batch_size, sequence_length):
    """
    Generator function for creating random batches of training-data.
    """

    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(num_train - sequence_length)
            
            # Copy the sequences of data starting at this index.
            x_batch[i] = x_train_scaled[idx:idx+sequence_length]
            y_batch[i] = y_train_scaled[idx:idx+sequence_length]
        
        yield (x_batch, y_batch)

#%%
batch_size = 100
sequence_length = 24 * 7 * 8
generator = batch_generator(batch_size=batch_size,
                            sequence_length=sequence_length)
x_batch, y_batch = next(generator)



#%%
validation_data = (np.expand_dims(x_test_scaled, axis=0),
                   np.expand_dims(y_test_scaled, axis=0))

#%%
warmup_steps = 50
def loss_mse_warmup(y_true, y_pred):
    """
    Calculate the Mean Squared Error between y_true and y_pred,
    but ignore the beginning "warmup" part of the sequences.
    
    y_true is the desired output.
    y_pred is the model's output.
    """

    # The shape of both input tensors are:
    # [batch_size, sequence_length, num_y_signals].

    # Ignore the "warmup" parts of the sequences
    # by taking slices of the tensors.
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]

    # These sliced tensors both have this shape:
    # [batch_size, sequence_length - warmup_steps, num_y_signals]

    # Calculate the MSE loss for each value in these tensors.
    # This outputs a 3-rank tensor of the same shape.
    loss = tf.losses.mean_squared_error(y_true_slice, y_pred_slice)

    # Keras may reduce this across the first axis (the batch)
    # but the semantics are unclear, so to be sure we use
    # the loss across the entire tensor, we reduce it to a
    # single scalar with the mean function.
    loss_mean = tf.reduce_mean(loss)

    return loss_mean

#%%



from tensorflow.python.keras.initializers import RandomUniform
from tensorflow.keras import layers
from tensorflow.keras.layers import GRU,Embedding, LSTM

init = RandomUniform(minval=-0.05, maxval=0.05)

model = tf.keras.Sequential()

# model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5, return_sequences=True, input_shape=(None, float_data.shape[-1])))
# model.add(layers.GRU(64, activation='relu', dropout=0.1,
# recurrent_dropout=0.5))

model.add(Embedding((None, num_x_signals), 32)) 
model.add(LSTM(32))


# model.add(layers.GRU(64,
#                     return_sequences=True,
#                      input_shape=(None, num_x_signals)))

#model.add(layers.GRU(64, activation='relu', dropout=0.1, recurrent_dropout=0.5))

model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(lr=1e-3), loss='mae')

#%%



history = model.fit_generator(generator,
                              steps_per_epoch=30,
                              epochs=10,
                              validation_data=validation_data)


                              
# model = Sequential()
# model.add(GRU(units=512,
#               return_sequences=True,
#               input_shape=(None, num_x_signals,)))
# model.add(Dense(num_y_signals, activation='sigmoid'))


#%%
result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
                        y=np.expand_dims(y_test_scaled, axis=0))

#%%
def plot_comparison(start_idx, length=100, train=True):
    """
    Plot the predicted and true output-signals.
    
    :param start_idx: Start-index for the time-series.
    :param length: Sequence-length to process and plot.
    :param train: Boolean whether to use training- or test-set.
    """
    
    if train:
        # Use training-data.
        x = x_train_scaled
        y_true = y_train
    else:
        # Use test-data.
        x = x_test_scaled
        y_true = y_test
    
    # End-index for the sequences.
    end_idx = start_idx + length
    
    # Select the sequences from the given start-index and
    # of the given length.
    x = x[start_idx:end_idx]
    y_true = y_true[start_idx:end_idx]
    
    # Input-signals for the model.
    x = np.expand_dims(x, axis=0)

    # Use the model to predict the output-signals.
    y_pred = modelLSTM.predict(x)
    
    # The output of the model is between 0 and 1.
    # Do an inverse map to get it back to the scale
    # of the original data-set.
    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])
    
    # For each output-signal.
    for signal in range(len(target_names)):
        # Get the output-signal predicted by the model.
        signal_pred = y_pred_rescaled[:, signal]
        
        # Get the true output-signal from the data-set.
        signal_true = y_true[:, signal]

        # Make the plotting-canvas bigger.
        plt.figure(figsize=(15,5))
        
        # Plot and compare the two signals.
        plt.plot(signal_true, label='true')
        plt.plot(signal_pred, label='pred')
        
        # Plot grey box for warmup-period.
        p = plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)
        
        # Plot labels etc.
        plt.ylabel(target_names[signal])
        plt.legend()
        plt.show()

#%%
plot_comparison(start_idx=0, length=48, train=True)


#%%
x_train_scaled[:24]

#%%
y_train_scaled[:24]

#%%
modelLSTM = tf.keras.models.Sequential()
modelLSTM.add(tf.keras.layers.LSTM(32, return_sequences=True, input_shape=(None, num_x_signals)))
modelLSTM.add(Dropout(0.25))
modelLSTM.add(Dense(1, activation='softmax'))
modelLSTM.compile(loss='mae', optimizer='adam', metrics=['accuracy'])

#%%


history = modelLSTM.fit_generator(generator,
                              steps_per_epoch=10,
                              epochs=5,
                              validation_data=validation_data)

#%%
history.history

#%%
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

#%%
