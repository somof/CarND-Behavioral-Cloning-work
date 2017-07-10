"""
The traffic signs are 32x32 so you
have to resize them to be 227x227 before
passing them to AlexNet.
"""
import time
import tensorflow as tf
import numpy as np
from scipy.misc import imread
from caffe_classes import class_names
from alexnet import AlexNet


# placeholders
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))

assert resized is not Ellipsis, "resized needs to modify the placeholder image size to (227,227)"
probs = AlexNet(resized)

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
        print(" {1:4.1f}%: {0:24s}".format(class_names[inference[-1 - i]], 100.0 * output[index, inference[-1 - i]]))

    print()

print("Time: %.3f seconds" % (time.time() - t))
