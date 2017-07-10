import os
import argparse
from PIL import Image
import numpy as np
import sklearn

np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
# from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda
from keras.layers import Cropping2D

import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(description='Remote Driving')
parser.add_argument('-i', '--csvfile', type=str, help='input csv file')
parser.add_argument('-d', '--imgdir', type=str, help='input data dir')
args = parser.parse_args()
# parser.print_help()
# print(args)

print('csvfile  : ', args.csvfile)
print('imagedir : ', args.imgdir)

# center,left,right,steering,throttle,brake,speed
df = pd.read_csv(args.csvfile, skiprows=1)

# 学習用と検証用を別個に作る

sklearn.utils.shuffle(df, replace=False)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(df, test_size=0.2)
print('train data size     : ', len(train_samples))
print('validation data size: ', len(validation_samples))



# images = []
# measurements = []
# for i, dat in df.iterrows():
#     path = os.path.join(args.imgdir, dat[0])
#     # if i % 1000 == 0:
#     #     plt.imshow(Image.open(os.path.join(args.imgdir, dat[0].strip())).crop((0, 70, 320, 120)))
#     #     plt.imshow(Image.open(os.path.join(args.imgdir, dat[1].strip())).crop((0, 70, 320, 120)))
#     #     plt.imshow(Image.open(os.path.join(args.imgdir, dat[2].strip())).crop((0, 70, 320, 120)))
#     #     plt.show()
#     #     plt.imshow(im.crop((0, 70, 320, 120)))
#     #     plt.show()

#     steering = float(dat[3])

#     # center camera
#     images.append(np.asarray(Image.open(os.path.join(args.imgdir, dat[0].strip()))))
#     measurements.append(steering)
#     # left camera
#     images.append(np.asarray(Image.open(os.path.join(args.imgdir, dat[1].strip()))))
#     measurements.append(steering + 0.2)
#     # right camera
#     images.append(np.asarray(Image.open(os.path.join(args.imgdir, dat[2].strip()))))
#     measurements.append(steering - 0.2)

# X_train = np.array(images)
# y_train = np.array(measurements)
# print(X_train.dtype, X_train.shape)


def generator(samples, batch_size=32, steer_offset=0.2):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        # shuffle(samples)
        sklearn.utils.shuffle(samples, replace=False)
        for offset in range(0, num_samples, batch_size):
            # batch_samples = samples[offset:offset+batch_size]
            batch_samples = samples.ix[:, offset:offset+batch_size]

            images = []
            angles = []
            # for batch_sample in batch_samples:
            for i, dat in batch_samples.iterrows():
                steering = float(dat[3])
                # center camera
                images.append(np.asarray(Image.open(os.path.join(args.imgdir, dat[0].strip()))))
                angles.append(steering)
                # left camera
                images.append(np.asarray(Image.open(os.path.join(args.imgdir, dat[1].strip()))))
                angles.append(steering + steer_offset
                # right camera
                images.append(np.asarray(Image.open(os.path.join(args.imgdir, dat[2].strip()))))
                angles.append(steering - steer_offset)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 20), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#  history = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5, verbose=2)
history = model.fit_generator(train_generator,
                              samples_per_epoch=len(train_samples),
                              validation_data=validation_generator,
                              nb_val_samples=len(validation_samples),
                              nb_epoch=5,
                              verbose=2)

model.save('model.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('the final model all mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.xlim(0, 10)
plt.xticks(np.arange(0, 10, 1))
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('fig/LossMetrics.png')
# plt.show()

"""
"""
