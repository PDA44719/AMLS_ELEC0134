from math import atan
import numpy as np
from sklearn.svm import SVC
from facial_regions_detector import extract_features_labels
from sklearn.metrics import accuracy_score
import os

class TaskB1Evaluator:
    """A class that is used to run and evaluate the proposed task B2 solution."""


    def __init__(self):
        """Create an instance of the class."""
        self.training_directory = "./Datasets/cartoon_set"
        self.testing_directory = "./Datasets/cartoon_set_test"
        self.img_dir_training = os.path.join(self.training_directory,'img')
        self.img_dir_testing = os.path.join(self.testing_directory,'img')

    def __create_dataset(self, basedir, imgdir):
        """
        Create a couple of numpy arrays containing the image landmarks and labels from
        a specific directory. The images that the detector is not able to extract features
        from, are not included.

        Arguments
        ---------
        basedir : str
            Directory containing the labels.csv file. It could either be the training or testing
            directory.
        img_dir : str
            Directory that contains the images whose features are to be extracted.
        
        Returns
        -------
        dataset : numpy.ndarray
            Array containing the 68 landmarks from the images analyzed by the landmark
            detector.
        labels : numpy.ndarray
            Array containing the face shape labels of the images that he detector was able to
            identify.

        """
        dataset, labels, _ = extract_features_labels(basedir, imgdir, 'labels.csv')
        return dataset, labels

        
    def __extract_angle_features(self, feature_number, side, dataset):
        """
        Using the formulas specified in the report, extract the angle between the some of the
        face ladmarks extracted.

        Arguments
        ---------
        feature : str
            Directory containing the labels.csv file. It could either be the training or testing
            directory.
        side : str
            The side of the face where the landmarks are located. It can either be right or left
            side. This argument is needed because the formulas change depending on the whether
            the angle of right or left side landmarks wants to be calculated.
        dataset : numpy.ndarray
            The array containing the extracted face landmarks of the different images in a
            directory.
        
        Returns
        -------
        list
            The list of angles/features that have been calculated for all the images whose 
            landmarks were extracted.

        """
        if side == "left":
            features = [atan(abs(element[feature_number-4][0] - element[8][0])/abs(element[feature_number-4][1] - element[8][1])) for element in dataset]
        else:  # Right side
            features = [atan(abs(element[feature_number-3][0] - element[8][0])/abs(element[feature_number-3][1] - element[8][1])) for element in dataset]
        return features

    def __extract_features(self, dataset):
        """
        Based on the formulas specified in the report, calculate all the features that will be used 
        for the training process (and for testing as well).
        
        Arguments
        ---------
        dataset : numpy.ndarray
            Array containing the 68 landmarks from the images analyzed by the landmark
            detector.
        
        Returns
        -------
        list
            A list of lists containing the features of all the images in the datasets. These
            features will then be used for training/testing the model.
        
        """
        features = []
        # Append feature 1, feature 2 and feature 3 (distance ratios specified in the report)
        features.append([(element[8][1]-element[19][1])/np.linalg.norm(element[0] - element[16]) for element in dataset])
        features.append([(np.linalg.norm(element[4] - element[12]))/(np.linalg.norm(element[0] - element[16])) for element in dataset])
        features.append([(np.linalg.norm(element[8] - element[57]))/(np.linalg.norm(element[4] - element[12])) for element in dataset])

        # Append features 4-19 (angles specified in the report)
        features.extend(self.__extract_angle_features(i, "left", dataset) for i in range(4, 12))
        features.extend(self.__extract_angle_features(i, "right", dataset) for i in range(12, 20))

        return features

    def __create_features_array(self, features):
        """
        Insert the different features into a numpy array.
        
        Arguments
        ---------
        features : list
            A list containing the 19 features that are to be inserted into the array.
        
        Returns
        -------
        numpy.ndarray
            The array containing the features.

        """
        images_features = np.ones([len(features[0]), len(features)])  # Initialize the array

        # For every image whose features were extracted, insert those features into the array
        for i in range(len(features[0])):
            # For image i, insert each of its 19 features into row i of the array
            images_features[i] = [features[j][i] for j in range(len(features))]

        return images_features
    
    def __fit_and_predict(self, train_features, train_labels, test_features):
        """
        Fit the SVM-RBF classifier using the train features and labels, and make predictions on 
        the testing set.

        Argument
        --------
        train_features : numpy.ndarray
            Array containing the 19 features of the different images in the train dataset.
        train_labels : numpy.ndarray
            Label indicating the face shape of the avatars.
        test_features : numpy.ndarray
            Array containing the 19 features of the different images in the test dataset.

        Returns
        -------
        numpy.ndarray
            An array containing the predictions made by the SVM model.

        """
        svm_rbf = SVC(kernel = 'rbf', probability=True, C=6, gamma=0.6)
        svm_rbf.fit(train_features, train_labels)
        y_pred = svm_rbf.predict(test_features)
        return y_pred

    def run_task(self):
        """Train the model and evaluate it on the testing set."""
        # Extract the train landmarks, labels and calculate the features from those landmarks
        train_dataset, train_labels = self.__create_dataset(self.training_directory, self.img_dir_training)
        train_features_list = self.__extract_features(train_dataset)
        train_features_arr = self.__create_features_array(train_features_list)

        # Extract the test landmarks, labels and calculate the features from those landmarks
        test_dataset, test_labels = self.__create_dataset(self.testing_directory, self.img_dir_testing)
        test_features_list = self.__extract_features(test_dataset)
        test_features_arr = self.__create_features_array(test_features_list)

        # Evaluate the performance on the testing set
        predictions = self.__fit_and_predict(train_features_arr, train_labels, test_features_arr)
        print(f'The accuracy obtained on the testing set is: {accuracy_score(test_labels, predictions)}')


    

            
