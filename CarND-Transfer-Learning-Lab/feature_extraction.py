import pickle
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
# TODO: import Keras layers you need here
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.datasets import cifar10

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")
flags.DEFINE_integer('epochs', 50, "The number of epochs.")
flags.DEFINE_integer('batch_size', 256, "The batch size.")


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)
    #(X_train, y_train), (X_val, y_val) = cifar10.load_data()
    y_train = y_train.reshape(-1)
    y_val = y_val.reshape(-1)
    
    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.3, random_state = 42, stratify = y_train)
    
    print('Training Feature Shape:', X_train.shape, 'Training Label Shape:', y_train.shape)
    print('Validation Feature Shape:', X_val.shape, 'Validation Label Shape:',y_val.shape)

    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic
    nb_classes = len(np.unique(y_train))
    input_shape = X_train.shape[1:]
    inp = Input(shape=input_shape)
    x = Flatten()(inp)
    x = Dense(nb_classes, activation='softmax')(x)
    model = Model(inp, x)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # TODO: train your model here
    model.fit(X_train, y_train, FLAGS.batch_size, FLAGS.epochs, validation_data=(X_val, y_val), shuffle=True)


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
