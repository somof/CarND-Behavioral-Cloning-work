import time
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.misc import imread
from alexnet import AlexNet

sign_names = pd.read_csv('signnames.csv')
nb_classes = 43

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))

# Returns the second final layer of the AlexNet model,
# this allows us to redo the last layer for the specifically for 
# traffic signs model.
fc7 = AlexNet(resized, feature_extract=True)
# TODO: Define a new fully connected layer followed by a softmax activation to
#   classify the traffic signs. Assign the result of the softmax activation to
#   `probs` below.
# HINT: Look at the final layer definition in alexnet.py to get an idea of what this
#   should look like.
# shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
# probs = ...

shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
probs = tf.nn.softmax(logits)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Answer
answer = ['construction', 'stop']

# Read Images
im1 = imread("construction.jpg").astype(np.float32)
im1 = im1 - np.mean(im1)

im2 = imread("stop.jpg").astype(np.float32)
im2 = im2 - np.mean(im2)

# Run Inference
t = time.time()
output = sess.run(probs, feed_dict={x: [im1, im2]})

# Print Output
print()
for index in range(output.shape[0]):
    inference = np.argsort(output)[index, :]

    print("Image {0:2d}: {1:s}".format(index, answer[index]))
    for i in range(5):
        print(" {1:4.1f}%: {0:24s}".format(sign_names.ix[inference[-1 - i]][1], 100.0 * output[index, inference[-1 - i]]))

    print()

print("Time: %.3f seconds" % (time.time() - t))

"""
Here's how I did it:

# Returns the second final layer of the AlexNet model,
# this allows us to redo the last layer specifically for 
# traffic signs model.
fc7 = AlexNet(resized, feature_extract=True)
shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
probs = tf.nn.softmax(logits)
"""
