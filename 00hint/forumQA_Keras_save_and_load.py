"""
I'm trying to progressively train my model on data that I collect. 
I started by training my model on the Udacity supplied sample data and saved the weights. 
However, I'm not sure whether I'm correctly reloading the weights when passing the new data set for training. 
I collected a small amount of data compared to the Udacity dataset and my model performed significantly worse making me 
think I was training from scratch instead of starting with the pre-loaded weights. 
Here's my code with some pseudo code for brevity:
"""


def main():
    # Read in data and perform basic scaling and augmentation to balance data set
    # Define data generators
    train_generator = generator(paths_train, samples_train, BATCH_SIZE, args.training_data)
    valid_generator = generator(paths_val, samples_val, BATCH_SIZE, args.training_data, 1)

    #Setup the model architecture

    #do cropping
    #sample convolution layer below
    model.add(Convolution2D(24, 5, 5, init = 'normal', activation = 'elu', border_mode = 'valid', subsample = (2,2), 
                            W_regularizer=l2(0.001), bias=True))
    ...
    model.add(Flatten())
    ...
    #Input shape: (None, 50), Output shape: (None, 10)
    model.add(Dense(10))
    #Input shape: (None, 10), Output shape: (None, 1)
    model.add(Dense(1))
    
    model.compile(optimizer = 'adam', loss = 'mse')
    
    if args.weights:
        print("Loading weights:", args.weights)
        model.load_weights(args.weights)
    
    training_history = model.fit_generator(generator = train_generator, samples_per_epoch = len(samples_train), 
                                nb_epoch = args.epochs, verbose = 1, 
                                validation_data = valid_generator, nb_val_samples = len(samples_val))
    
    model.save_weights(args.file + '_weights.h5')
    model.save(args.file + '_model.h5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autonomous Driving Model Training')
    parser.add_argument(
            '-td',
            '--training_data',
            type=str,
            help='Relative path to training data e.g.".\Folder\"'
    )
    parser.add_argument(
            '-e',
            '--epochs',
            type=int,
            default=5,
            help='# of epochs to train the model'
    )
    parser.add_argument(
            '-w',
            '--weights',
            type=str,
            default=' ',
            help='Trained weights file (*.h5) to initialize parameters'
    )
    parser.add_argument(
            '-f',
            '--file',
            type=str,
            help='filename to save weights and model'
    )
    
    
    args = parser.parse_args()
    
    #Call main
    main()

"""
At runtime, I pass in the last saved weights file and the path to the new training data set. 
Is there a reliable way to check if my weights are being loaded properly into my model? 
Also, I am not sure if the init kwarg in my Convolution2D layers is overwriting my preloaded weights.
"""


# subodh.malgonde Forum Mentor
"""
You can refer to the code in drive.py to load the model. It uses something similar to this:

    from keras.models import load_model
    model = load_model('model.h5')

This assumes that you saved your model by doing:

    model.save('model.h5')

>>raza.shahzad:
  model.save_weights(args.file + '_weights.h5')
  model.save(args.file + '_model.h5')

You need not save the weights separately.
The h5 file contains information about both the weights and the architecture.

>>raza.shahzad:
>>Also, I am not sure if the init kwarg in my Convolution2D layers is overwriting my preloaded weights.

You need not make the model architecture at all if you are using the .h5 file. You should do something like:

    if args.h5_filename:
        model = load_model('model.h5')
    else:
        model = Sequential()
        ....
        # add layers

Let us know if this answers your question.
"""

    
