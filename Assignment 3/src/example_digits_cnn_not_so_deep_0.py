# example_digits_cnn_not_so_deep.py

import os
import numpy as np
import scipy
import scipy.ndimage
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import InputLayer, Flatten, Dense, Softmax
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy

def read_hep2_dataset():
    matfilename = os.path.join('databases','hep_proper_mask.mat')
    mat_content = scipy.io.loadmat(matfilename)
    X1_masks = np.transpose(mat_content['Y1'], (3, 0, 1, 2))
    X1_masks = X1_masks.astype(bool)
    X2_masks = np.transpose(mat_content['Y2'], (3, 0, 1, 2))
    X2_masks = X2_masks.astype(bool)
    matfilename = os.path.join('databases','hep_proper_v2.mat')
    mat_content = scipy.io.loadmat(matfilename, verify_compressed_data_integrity=False)
    X1 = np.transpose(mat_content['X1'], (3, 0, 1, 2))
    X2 = np.transpose(mat_content['X2'], (3, 0, 1, 2))
    Y1 = np.asarray(mat_content['Y1'][0]) - 1
    Y1 = Y1.astype(int)
    Y2 = np.asarray(mat_content['Y2'][0]) - 1
    Y2 = Y2.astype(int)

    return X1_masks, X2_masks, X1, X2, Y1, Y2

def read_mnist_dataset(n_train=5000, n_test=5000, rotate=True):
    # We use the first 5000 images and apply rotations to create a dataset similar to the one provided in the MATLAB version

    (train_X, train_y), (test_X, test_y) = datasets.mnist.load_data()
    train_X = train_X[..., None] / 255.0  # Normalize and add channel dimension
    test_X = test_X[..., None] / 255.0    # Normalize and add channel dimension
    
    train_X = train_X[:n_train]
    train_y = train_y[:n_train]
    test_X = test_X[:n_test]
    test_y = test_y[:n_test]

    if rotate:
        for i, (x) in enumerate(train_X):
            train_X[i] = scipy.ndimage.rotate(x, np.random.uniform(-45,45), reshape=False, mode='nearest')
        for i, (x) in enumerate(test_X):
            test_X[i] = scipy.ndimage.rotate(x, np.random.uniform(-45,45), reshape=False, mode='nearest')

    return train_X, test_X, train_y, test_y

def get_model1(input_shape = (28, 28, 1), nbr_classes = 10, lr = 1.0):

    model = models.Sequential([
        InputLayer(input_shape=input_shape),  # Specify input size
        Flatten(),                            # Flatten the 2D image to a 1D vector
        Dense(nbr_classes),                            # Fully connected layer, from prod(input size) nodes to 10 nodes
        Softmax()                             # Convert to probabilities
    ])

    # a different way to write the same thing
    #model = models.Sequential()
    #model.add(InputLayer(input_shape=input_shape))
    #model.add(Flatten())
    #model.add(Dense(10))
    #model.add(Softmax())

    model.compile(optimizer=SGD(learning_rate=lr), 
              loss=SparseCategoricalCrossentropy(), 
              metrics=['accuracy'])
    
    return model

    '''
    # Some other layers that might be useful
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    '''

if __name__ == "__main__":
    
    # load the mnist dataset
    train_X, test_X, train_y, test_y = read_mnist_dataset()
    
    # load the hep2 dataset
    #train_X_masks, test_X_masks, train_X, test_X, train_y, test_y = read_hep2_dataset()
    
    # Select the deep learning network architecture
    input_shape = train_X.shape[1:]
    nbr_classes = len(np.unique(train_y))
    mini_batch_size = 512
    max_epochs = 30
    learning_rate = 1.0
    
    model = get_model1(input_shape, nbr_classes, learning_rate)

    # train the model
    history = model.fit(
        train_X, train_y,
        batch_size=mini_batch_size,
        epochs=max_epochs,
        validation_data=(test_X, test_y),
        verbose=1
    )

    # evaluate the model om the training data
    train_loss, train_accuracy = model.evaluate(train_X, train_y, verbose=0)
    print(f'\nThe accuracy on the training set: {train_accuracy}')

    # then evaluate the model on the test data
    test_loss, test_accuracy = model.evaluate(test_X, test_y, verbose=0)
    print(f'The accuracy on the test set: {test_accuracy}')