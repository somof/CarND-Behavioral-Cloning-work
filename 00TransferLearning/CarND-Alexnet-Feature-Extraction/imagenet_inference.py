# NOTE: You don't need to edit this code.
import time
import tensorflow as tf
import numpy as np
from scipy.misc import imread
from caffe_classes import class_names
from alexnet import AlexNet


# placeholders
x = tf.placeholder(tf.float32, (None, 227, 227, 3))

# By keeping `feature_extract` set to `False`
# we indicate to keep the 1000 class final layer
# originally used to train on ImageNet.
probs = AlexNet(x, feature_extract=False)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Answer
answer = ['poodle', 'weasel']

# Read Images
im1 = (imread("poodle.png")[:, :, :3]).astype(np.float32)
im1 = im1 - np.mean(im1)

im2 = (imread("weasel.png")[:, :, :3]).astype(np.float32)
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
