# example_hep_ml_v0.py

import scipy.io 
import os.path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def read_hep2_dataset():

    # Get the folder where THIS script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Build the path to Assignment 3/data/databases
    data_dir = os.path.join(base_dir, "..", "data", "databases")

    # Make absolute paths
    mask_path = os.path.join(data_dir, "hep_proper_mask.mat")
    v2_path   = os.path.join(data_dir, "hep_proper_v2.mat")

    # Load mask file
    mat_content = scipy.io.loadmat(mask_path)
    X1_masks = np.transpose(mat_content['Y1'], (3, 0, 1, 2)).astype(bool)
    X2_masks = np.transpose(mat_content['Y2'], (3, 0, 1, 2)).astype(bool)

    # Load v2 file
    mat_content = scipy.io.loadmat(v2_path, verify_compressed_data_integrity=False)
    X1 = np.transpose(mat_content['X1'], (3, 0, 1, 2))
    X2 = np.transpose(mat_content['X2'], (3, 0, 1, 2))

    Y1 = (np.asarray(mat_content['Y1'][0]) - 1).astype(int)
    Y2 = (np.asarray(mat_content['Y2'][0]) - 1).astype(int)

    return X1_masks, X2_masks, X1, X2, Y1, Y2


from skimage.measure import shannon_entropy

from skimage.measure import shannon_entropy
from skimage.feature import greycomatrix, greycoprops
import numpy as np

def get_features(image, mask):
    pixels = image[mask]  # extract only cell pixels

    F = []
    f_name = []

    # ------------------------------
    # 1. Intensity-based features
    # ------------------------------
    F.append(np.mean(pixels))
    f_name.append('mean intensity')

    F.append(np.std(pixels))
    f_name.append('std intensity')

    F.append(np.median(pixels))
    f_name.append('median intensity')

    F.append(np.min(pixels))
    f_name.append('min intensity')

    F.append(np.max(pixels))
    f_name.append('max intensity')

    # ------------------------------
    # 2. Texture features
    # ------------------------------
    # Entropy
    F.append(shannon_entropy(pixels))
    f_name.append('entropy')

    # GLCM texture (contrast & homogeneity)
    # Use scaled image for GLCM
    img_uint8 = (image * 255).astype('uint8')

    glcm = greycomatrix(img_uint8,
                        distances=[1],
                        angles=[0],
                        symmetric=True,
                        normed=True)

    contrast = greycoprops(glcm, 'contrast')[0, 0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]

    F.append(contrast)
    f_name.append('glcm contrast')

    F.append(homogeneity)
    f_name.append('glcm homogeneity')

    # ------------------------------
    # 3. Shape features
    # ------------------------------
    area = np.sum(mask)
    F.append(area)
    f_name.append('area')

    # Shape compactness = area / perimeter^2
    # Simple perimeter estimate: count boundary pixels
    perimeter = np.sum(mask ^ np.pad(mask, ((1, 1), (1, 1)), mode='constant')[1:-1, 1:-1])

    # Avoid division by zero
    compactness = area / (perimeter**2 + 1e-6)
    F.append(compactness)
    f_name.append('compactness')

    # ------------------------------
    # Done
    # ------------------------------
    assert len(F) == len(f_name)
    return F, f_name



def get_all_features(X1,X2):
    # iterate over all training images and get features
    nr_training_images = X1.shape[0]
    train_features = []
    for i in range(nr_training_images):
        fv, f_name = get_features(X1[i,:,:,0],X1_masks[i,:,:,0])
        train_features.append(fv)
    train_features = np.asarray(train_features)

    # iterate over all test images and get features 
    nr_test_images = X2.shape[0]
    test_features = []
    for i in range(nr_test_images):
        fv,f_name = get_features(X2[i,:,:,0],X2_masks[i,:,:,0])
        test_features.append(fv)
    test_features = np.asarray(test_features)

    return train_features, test_features, f_name

def plot_features(features,Y1,f_name,features_to_plot):
    f_name = [f_name[idx] for idx in features_to_plot]
    features = features[:,features_to_plot]
    feature1 = features[:,0]
    feature2 = features[:,1] 
    colors = ['r', 'g', 'b', 'r', 'g', 'b']
    markers = ['o', 'o', 'o', '+', '+', '+']

    # Create a scatter plot
    plt.figure(1)
    for label, color, marker in zip(range(6), colors, markers):
        plt.scatter(
            feature1[Y1 == label],  # X-values for the current class
            feature2[Y1 == label],  # Y-values for the current class
            c=color,
            label=f'Class {label}',
            marker=marker
        )

    # Add labels and legend
    plt.xlabel(f_name[0])
    plt.ylabel(f_name[1])
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # read data. X1 is training data, X2 is test data. Y1 and Y2 the corresponding labels
    X1_masks, X2_masks, X1, X2, Y1, Y2 = read_hep2_dataset()
    
    # get the features for all train and testa data. f_name is naem of features
    train_features, test_features, f_name = get_all_features(X1,X2)

    # Visualize the data. Plot two of the features against eachother
    features_to_plot = [0,1]
    plot_features(train_features,Y1,f_name,features_to_plot)
    
    ###########################################################
    ########## Train machine learning models to data ##########
    ###########################################################

    ############### Descision tree ###############
    print('\nFitting a decision tree model')
    model1 = DecisionTreeClassifier()
    model1.fit(train_features, Y1)

    # Test the classifier on the training set
    Y_result1 = model1.predict(train_features)
    accuracy1 = accuracy_score(Y1, Y_result1)
    print(f"The accuracy on the training set: {accuracy1}")

    # Test the classifier on the test set
    Y_result2 = model1.predict(test_features)  
    accuracy2 = accuracy_score(Y2, Y_result2)
    print(f"The accuracy on the test set: {accuracy2}")

    ############### Random forest ###############
    print('\nFitting a random forest model')
    model2 = RandomForestClassifier(n_estimators=50, oob_score=True)
    model2.fit(train_features, Y1)

    # Test the classifier on the training set
    Y_result1 = model2.predict(train_features)
    accuracy1 = accuracy_score(Y1, Y_result1)
    print(f"The accuracy on the training set: {accuracy1}")

    # Test the classifier on the test set
    Y_result2 = model2.predict(test_features)
    accuracy2 = accuracy_score(Y2, Y_result2)
    print(f"The accuracy on the test set: {accuracy2}")

    ############### Support vector machine ###############
    print('\nFitting a support vector machine model')
    model3 = SVC()
    model3.fit(train_features, Y1)

    # Test the classifier on the training set
    Y_result1 = model3.predict(train_features)
    accuracy1 = accuracy_score(Y1, Y_result1)
    print(f"The accuracy on the training set: {accuracy1}")

    # Test the classifier on the test set
    Y_result2 = model3.predict(test_features)
    accuracy2 = accuracy_score(Y2, Y_result2)
    print(f"The accuracy on the test set: {accuracy2}")

    ############### kNN ###############
    print('\nFitting a k-nearest neighbour model')
    model4 = KNeighborsClassifier()
    model4.fit(train_features, Y1)

    # Test the classifier on the training set
    Y_result1 = model4.predict(train_features)
    accuracy1 = accuracy_score(Y1, Y_result1)
    print(f"The accuracy on the training set: {accuracy1}")

    # Test the classifier on the test set
    Y_result2 = model4.predict(test_features)
    accuracy2 = accuracy_score(Y2, Y_result2)
    print(f"The accuracy on the test set: {accuracy2}")

    print('')