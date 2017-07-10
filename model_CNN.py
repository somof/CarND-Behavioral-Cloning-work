import os
import argparse
import pickle
from PIL import Image
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='Remote Driving')
parser.add_argument('-i', '--csvfile', type=str, help='input csv file')
parser.add_argument('-d', '--imgdir', type=str, help='input data dir')
args = parser.parse_args()
# parser.print_help()
# print(args)

print('csvfile  : ', args.csvfile)
print('imagedir : ', args.imgdir)

import matplotlib.pyplot as plt
import pandas as pd
# center,left,right,steering,throttle,brake,speed
df = pd.read_csv(args.csvfile, header=0)

images = []
measurements = []
for i, dat in df.iterrows():
    path = os.path.join(args.imgdir, dat[0])
    # if i % 1000 == 0:
    #     plt.imshow(Image.open(os.path.join(args.imgdir, dat[0].strip())).crop((0, 70, 320, 120)))
    #     plt.imshow(Image.open(os.path.join(args.imgdir, dat[1].strip())).crop((0, 70, 320, 120)))
    #     plt.imshow(Image.open(os.path.join(args.imgdir, dat[2].strip())).crop((0, 70, 320, 120)))
    #     plt.show()

    steering = float(dat[3])

    # center camera
    images.append(np.asarray(Image.open(os.path.join(args.imgdir, dat[0].strip()))))
    measurements.append(steering)
    # left camera
    images.append(np.asarray(Image.open(os.path.join(args.imgdir, dat[1].strip()))))
    measurements.append(steering + 0.2)
    # right camera
    images.append(np.asarray(Image.open(os.path.join(args.imgdir, dat[2].strip()))))
    measurements.append(steering - 0.2)

X_train = np.array(images)
y_train = np.array(measurements)
print(X_train.dtype)
print(X_train.shape)


np.random.seed(1337) # for reproducibility
    
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda
from keras.layers import Cropping2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 20), (0, 0))))

model.add(Convolution2D(20, 5, 5, subsample=(1, 1), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(50, 5, 5, subsample=(1, 1), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=20, verbose=2)
model.save('model_LeNet.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('LeNet model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.xlim(0, 20)
plt.xticks(np.arange(0, 20, 1))
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('fig/LossMetrics_LeNet.png')
# plt.show()

"""
"""
