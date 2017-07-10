Keras

# Overview

Keras makes coding deep neural networks simpler. 
To demonstrate just how easy it is, 
you're going to build a simple fully-connected network in a few dozen lines of code.

We’ll be connecting the concepts that 
you’ve learned in the previous lessons to the methods that Keras provides.

The network you will build is similar to Keras’s sample network 
that builds out a convolutional neural network for MNIST. 
However for the network you will build you're going to use a small subset of the German 
Traffic Sign Recognition Benchmark dataset that you've used previously.

The general idea for this example is that you'll first load the data, 
then define the network, and then finally train the network.



# Neural Networks in Keras

Here are some core concepts you need to know for working with Keras.

## Sequential Model

    from keras.models import Sequential
    
    #Create the Sequential model
    model = Sequential()

The keras.models.Sequential class is a wrapper for the neural network model. 
It provides common functions like fit(), evaluate(), and compile(). 
We'll cover these functions as we get to them. 
Let's start looking at the layers of the model.

## Layers

A Keras layer is just like a neural network layer. 
There are fully connected layers, max pool layers, and activation layers. 
You can add a layer to the model using the model's add() function. 
For example, a simple model would look like this:

    from keras.models import Sequential
    from keras.layers.core import Dense, Activation, Flatten
    
    #Create the Sequential model
    model = Sequential()
    
    #1st Layer - Add a flatten layer
    model.add(Flatten(input_shape=(32, 32, 3)))
    
    #2nd Layer - Add a fully connected layer
    model.add(Dense(100))
    
    #3rd Layer - Add a ReLU activation layer
    model.add(Activation('relu'))
    
    #4th Layer - Add a fully connected layer
    model.add(Dense(60))
    
    #5th Layer - Add a ReLU activation layer
    model.add(Activation('relu'))

Keras will automatically infer the shape of all layers after the first layer. 
This means you only have to set the input dimensions for the first layer.

The first layer from above, model.add(Flatten(input_shape=(32, 32, 3))), 
sets the input dimension to (32, 32, 3) and output dimension to (3072=32 x 32 x 3). 
The second layer takes in the output of the first layer and sets the output dimensions to (100). 
This chain of passing output to the next layer continues until the last layer, 
which is the output of the model.


## Quiz

In this quiz you will build a multi-layer feedforward neural network to classify 
traffic sign images using Keras.

1. Set the first layer to a Flatten() layer with the input_shape set to (32, 32, 3).
2. Set the second layer to a Dense() layer with an output width of 128.
3. Use a ReLU activation function after the second layer.
4. Set the output layer width to 5, because for this data set there are only 5 classes.
5. Use a softmax activation function after the output layer.
6. Train the model for 3 epochs. You should be able to get over 50% training accuracy.

To get started, review the Keras documentation about models and layers. 
The Keras example of a Multi-Layer Perceptron network is similar to what you need to do here. 
Use that as a guide, but keep in mind that there are a number of differences.

## Data Download
The data set used in these quizzes can be downloaded here.

https://d17h27t6h515a5.cloudfront.net/topher/2017/March/58dbf6d5_small-traffic-set/small-traffic-set.zip

    python network_solution.py


# Convolutions

1. Build from the previous network.
2. Add a convolutional layer with 32 filters, a 3x3 kernel, and valid padding 
   before the flatten layer.
3. Add a ReLU activation after the convolutional layer.
4. Train for 3 epochs again, should be able to get over 50% accuracy.

https://keras.io/layers/convolutional/#convolution2d

 - Hint 1: The Keras example of a convolutional neural network for MNIST 
   would be a good example to review.
https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

 - Hint 2: You can set the padding type by passing in a border_mode= argument 
   to the Convolution2D() layer.


    # Load pickled data
    import pickle
    import numpy as np
    import tensorflow as tf
    tf.python.control_flow_ops = tf
    
    with open('small_train_traffic.p', mode='rb') as f:
        data = pickle.load(f)
    
    X_train, y_train = data['features'], data['labels']
    
    # Initial Setup for Keras
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation, Flatten
    from keras.layers.convolutional import Convolution2D
    
    # Build Convolutional Neural Network in Keras Here
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 3)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    
    # Preprocess data
    X_normalized = np.array(X_train / 255.0 - 0.5 )
    
    from sklearn.preprocessing import LabelBinarizer
    label_binarizer = LabelBinarizer()
    y_one_hot = label_binarizer.fit_transform(y_train)
    
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    history = model.fit(X_normalized, y_one_hot, nb_epoch=3, validation_split=0.2)
    

# Pooling

1. Build from the previous network
2. Add a 2x2 max pooling layer immediately following your convolutional layer.
3. Train for 3 epochs again. You should be able to get over 50% training accuracy.

https://keras.io/layers/pooling/#maxpooling2d

    # Load pickled data
    import pickle
    import numpy as np
    import tensorflow as tf
    tf.python.control_flow_ops = tf
    
    with open('small_train_traffic.p', mode='rb') as f:
        data = pickle.load(f)
    
    X_train, y_train = data['features'], data['labels']
    
    # Initial Setup for Keras
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation, Flatten
    from keras.layers.convolutional import Convolution2D
    from keras.layers.pooling import MaxPooling2D
    
    # Build Convolutional Neural Network in Keras Here
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    
    # Preprocess data
    X_normalized = np.array(X_train / 255.0 - 0.5 )
    
    from sklearn.preprocessing import LabelBinarizer
    label_binarizer = LabelBinarizer()
    y_one_hot = label_binarizer.fit_transform(y_train)
    
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    history = model.fit(X_normalized, y_one_hot, nb_epoch=3, validation_split=0.2)


# Dropout

1. Build from the previous network.
2. Add a dropout layer after the pooling layer. Set the dropout rate to 50%.

    # Load pickled data
    import pickle
    import numpy as np
    import tensorflow as tf
    tf.python.control_flow_ops = tf
    
    with open('small_train_traffic.p', mode='rb') as f:
        data = pickle.load(f)
    
    X_train, y_train = data['features'], data['labels']
    
    # Initial Setup for Keras
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation, Flatten, Dropout
    from keras.layers.convolutional import Convolution2D
    from keras.layers.pooling import MaxPooling2D
    
    # Build Convolutional Pooling Neural Network with Dropout in Keras Here
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    
    # preprocess data
    X_normalized = np.array(X_train / 255.0 - 0.5 )
    
    from sklearn.preprocessing import LabelBinarizer
    label_binarizer = LabelBinarizer()
    y_one_hot = label_binarizer.fit_transform(y_train)
    
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    history = model.fit(X_normalized, y_one_hot, nb_epoch=3, validation_split=0.2)


# Test

Once you've picked out your best model, it's time to test it!

1. Try to get the highest validation accuracy possible. Feel free to use 
   all the previous concepts and train for as many epochs as needed.
2. Select your best model and train it one more time.
3. Use the test data and the Keras evaluate() method to see how well the model does.

https://keras.io/models/model/#evaluate

    # Load pickled data
    import pickle
    import numpy as np
    import tensorflow as tf
    tf.python.control_flow_ops = tf
    
    with open('small_train_traffic.p', mode='rb') as f:
        data = pickle.load(f)
    
    X_train, y_train = data['features'], data['labels']
    
    # Initial Setup for Keras
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation, Flatten, Dropout
    from keras.layers.convolutional import Convolution2D
    from keras.layers.pooling import MaxPooling2D
    
    # TODO: Build the Final Test Neural Network in Keras Here
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    
    # preprocess data
    X_normalized = np.array(X_train / 255.0 - 0.5 )
    
    from sklearn.preprocessing import LabelBinarizer
    label_binarizer = LabelBinarizer()
    y_one_hot = label_binarizer.fit_transform(y_train)
    
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    history = model.fit(X_normalized, y_one_hot, nb_epoch=10, validation_split=0.2)
    
    with open('small_test_traffic.p', 'rb') as f:
        data_test = pickle.load(f)
    
    X_test = data_test['features']
    y_test = data_test['labels']
    
    # preprocess data
    X_normalized_test = np.array(X_test / 255.0 - 0.5 )
    y_one_hot_test = label_binarizer.fit_transform(y_test)
    
    print("Testing")
    
    metrics = model.evaluate(X_normalized_test, y_one_hot_test)
    for metric_i in range(len(model.metrics_names)):
        metric_name = model.metrics_names[metric_i]
        metric_value = metrics[metric_i]
        print('{}: {}'.format(metric_name, metric_value))


