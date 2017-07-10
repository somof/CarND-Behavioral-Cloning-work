# Is this considered overfitting?


Alex,

Yes, the left and right images are also used in the validation set, is this a mistake? This is the way I'm implementing dropout per your comments:

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3), output_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Convolution2D(6, 5, 5, activation = "relu"))
model.add(Dropout(0.5))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation = "relu"))
model.add(Dropout(0.5))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 3, 3, activation = "relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 3, 3, activation = "relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 2)
I'm unsure on where exactly to add the dropouts, but the car does not even go past the first curve with this pipeline.



Alex_CuiForum Mentor2d
Hi @gabymosqueradiaz,

You model arch looks good to me but I do have a few comments:

1. Did you crop out the un-needed region in the image (sky, etc)?
2. You can add some dropout layers after flatten layers.
3. It seems you models works fine with most of the track but bad at one position, right? Do you have a plot of histogram of your steering angle for the training set? Should be something like this: https://discussions.udacity.com/t/angle-correction-doesnt-work/251286/2






Alex,

Yes, I cropped the 70 upper pixels and the 25 lower pixels of the image.
I changed my architecture to this (with the dropouts after the dense layers):

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3), output_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Convolution2D(6, 5, 5, activation = "relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation = "relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 3, 3, activation = "relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 3, 3, activation = "relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dropout(0.8))
model.add(Dense(84))
model.add(Dropout(0.8))
model.add(Dense(1))

And I still get these results, which to me clearly show overfitting:

Train on 21993 samples, validate on 5499 samples
Epoch 1/10
21993/21993 [==============================] - 313s - loss: 0.0494 - val_loss: 0.1146
Epoch 2/10
21993/21993 [==============================] - 313s - loss: 0.0367 - val_loss: 0.0893
Epoch 3/10
21993/21993 [==============================] - 313s - loss: 0.0347 - val_loss: 0.0860
Epoch 4/10
21993/21993 [==============================] - 313s - loss: 0.0333 - val_loss: 0.0872
Epoch 5/10
21993/21993 [==============================] - 313s - loss: 0.0330 - val_loss: 0.0957
Epoch 6/10
21993/21993 [==============================] - 314s - loss: 0.0324 - val_loss: 0.0914
Epoch 7/10
21993/21993 [==============================] - 313s - loss: 0.0322 - val_loss: 0.0873
Epoch 8/10
21993/21993 [==============================] - 313s - loss: 0.0317 - val_loss: 0.0880
Epoch 9/10
21993/21993 [==============================] - 312s - loss: 0.0315 - val_loss: 0.1002
Epoch 10/10
21993/21993 [==============================] - 311s - loss: 0.0314 - val_loss: 0.0869
With that neural network, my car still does not get past the dirt road portion after the bridge. What am I doing wrong?

This is all my code is:

import os
import csv
import cv2
import numpy as np


samples = []
with open('/home/carnd/P3-Gaby/data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
images = []
measurements = []
for line in samples:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('\\')[-1]
        current_path = '/home/carnd/P3-Gaby/data/data/IMG/' + filename
        image = cv2.imread(current_path)
                
        if image is None:
            print("Image path incorrect: ", current_path)
            continue  # skip adding these rows in the for loop
        
        images.append(image)
        
    correction = 0.3
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement+correction)
    measurements.append(measurement-correction)
        
        
X_train = np.array(images)
y_train = np.array(measurements)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D, MaxPooling2D, Dropout

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3), output_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Convolution2D(6, 5, 5, activation = "relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation = "relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 3, 3, activation = "relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 3, 3, activation = "relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dropout(0.8))
model.add(Dense(84))
model.add(Dropout(0.8))
model.add(Dense(1))


model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 10)

model.save('model_dropout.h5')
exit()
Can't figure out what I'm doing wrong or how I could improve the model or the overfitting.







Alex_CuiForum Mentor1d  gabymosqueradiaz
Hi @gabymosqueradiaz,

Can you try @sagarjaju375 suggestion?

One way to reduce over-fitting is to collect more data or make model simpler. If that suggestion doesn't work, you can try to make your model simpler and try again.




