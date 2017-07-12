import os
import csv
import argparse
from PIL import Image
import numpy as np
import sklearn

np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda
from keras.layers import Cropping2D

parser = argparse.ArgumentParser(description='Remote Driving')
parser.add_argument('-i', '--csvfile', type=str, help='input csv file')
parser.add_argument('-d', '--imgdir', type=str, help='input data dir')
args = parser.parse_args()
# parser.print_help()
# print(args)

print('csvfile  : ', args.csvfile)
print('imagedir : ', args.imgdir)

# center,left,right,steering,throttle,brake,speed
samples = []
with open(args.csvfile) as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)
    for line in reader:
        samples.append(line)

# shuffle before splitting dataset
sklearn.utils.shuffle(samples, replace=False)


# split dataset for test and validation
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print('train data size     : ', len(train_samples))
print('validation data size: ', len(validation_samples))


def generator(samples, batch_size=32, steer_offset=0.3):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples, replace=False)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for dat in batch_samples:
                steering = float(dat[3])
                # center camera
                images.append(np.asarray(Image.open(os.path.join(args.imgdir, dat[0].strip()))))
                angles.append(steering)
                # left camera
                images.append(np.asarray(Image.open(os.path.join(args.imgdir, dat[1].strip()))))
                angles.append(steering + steer_offset)
                # right camera
                images.append(np.asarray(Image.open(os.path.join(args.imgdir, dat[2].strip()))))
                angles.append(steering - steer_offset)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=100)
validation_generator = generator(validation_samples, batch_size=100)


# the final model based on NVIDIA model (PolotNet)
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 20), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Dropout(0.25))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.25))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(20))
model.add(Dropout(0.5))
model.add(Dense(1))


# training parameters
EPOCHS = 60
EPOCHS = 200
model.compile(loss='mse', optimizer='adam')
history = model.fit_generator(train_generator,
                              samples_per_epoch=len(train_samples),
                              validation_data=validation_generator,
                              nb_val_samples=len(validation_samples),
                              nb_epoch=EPOCHS,
                              verbose=2)

# save the model into a file
model.save('model.h5')


# save a training curve
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('the final model all mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.xlim(0, EPOCHS)
plt.xticks(np.arange(0, EPOCHS, 4))
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('fig/LossMetrics.png')
# plt.show()

"""
Output: steering

Convolutional feature map
64@2x33 

Convolutional feature map
64@4x35 

Convolutional feature map
48@6x37 

Convolutional feature map
36@15x77

Convolutional feature map
24@33x158
"""
