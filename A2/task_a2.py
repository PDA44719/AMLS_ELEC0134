import pandas as pd
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.callbacks import EarlyStopping
from facial_regions_detector import run_dlib_shape

class TaskA2Evaluator:
    """A class that is used to run and evaluate the proposed task A2 solution."""


    def __init__(self):
        """Create an instance of the class."""
        self.training_directory = "./Datasets/celeba"
        self.testing_directory = "./Datasets/celeba_test"

    def __create_dataframe(self, directory):
        """
        Create a dataframe containing the information available in the labels.csv file.

        Arguments
        ---------
        directory : str
            Directory containing the labels.csv file. It could either be the training or testing
            directory.
        
        Returns
        -------
        pandas.core.frame.DataFrame
            The dataframe that was created.

        """
        df = pd.read_csv(f'{directory}/labels.csv', sep='\t')
        df.drop(df.columns[0], axis=1, inplace=True)  # Drop the first column, which will not be used
        return df
    
    def __extract_mouth_landmarks(self, directory):
        """
        Go through every image on the directory, extract the landmarks associated with the mouth and
        the image labels. If the detector is not able to identify a face in an image, that image
        will not be considered.
        
        Arguments
        ---------
        directory : str
            Directory from which the image landmarks and labels will be extracted. It could either be
            the training or testing directory.
            
        Returns
        -------
        all_mouth_landmarks : list
            A list containing the mouth landmarks associated with the images (only the ones where
            a face was identified by the detector).
        labels : list
            The smiling/not smiling labels of the images (1 for smiling, 0 otherwise).
        
        """
        df = self.__create_dataframe(directory)

        # Initialize lists
        all_mouth_landmarks = []
        labels = []

        for index, row in df.iterrows():  # Go through every image on the dataframe
            print(f'Working on image {index+1} out of {len(df)}')
            image = Image.open(f'{directory}/img/{row["img_name"]}')
            image_array = np.array(image)
            features,_ = run_dlib_shape(image_array)  # Extract the 68 face landmarks

            # Only append the features and labels if the model could identify a face in the image
            if features is not None:  
                all_mouth_landmarks.append(features[48:])
                labels.append(1 if row['smiling'] == 1 else 0)
        
        return all_mouth_landmarks, labels

    def __compute_m_matrix(self, landmarks):
        """
        Compute the M matrix, which contains the distances between the different
        mouth landmarks in an image.

        Arguments
        ---------
        landmarks : list
            A list containing the x and y coordinate values of every mouth landmark
            in an image.
        
        Returns
        -------
        numpy.ndarray
            A numpy array containing the Euclidean distances between every mouth landmark
            in an image.

        """
        distances_matrix = np.ndarray((20, 20))
        for i in range(len(landmarks)):
            for j in range(len(landmarks)):
                distances_matrix[i][j] = np.linalg.norm(landmarks[i] - landmarks[j])
        return distances_matrix

    def __create_dataset(self, all_mouth_landmarks):
        """
        Generate the dataset which will be used for training/testing. It contains the 
        M matrices of all the images that the detector was able to extract features from.

        Arguments
        ---------
        all_mout_landmarks : list
            A list containing the all the face landmarks extracted when analyzing every 
            image in a directory.
        
        Returns
        -------
        numpy.ndarray
            A numpy array containing the M matrices of the images that will be used for
            training/testing.
        
        """
        dataset = np.ndarray((len(all_mouth_landmarks), 190))
        for i in range(len(all_mouth_landmarks)):
            print(f'Completing image {i+1} out of {len(all_mouth_landmarks)}')
            distances_matrix = self.__compute_m_matrix(all_mouth_landmarks[i])  # Obtain upper triangle
            D = distances_matrix[np.triu_indices(20, 1)]
            D_l1 = sum(D)
            F = D / D_l1
            dataset[i] = F
        
        return dataset

    def __create_mlp(self):
        """Define the Multi-Layer Perceptron (MLP) to be used for the task."""
        classifier = tf.keras.models.Sequential()
        classifier.add(tf.keras.layers.Dense(units = 100, activation = 'relu', input_shape = (190,)))
        classifier.add(tf.keras.layers.Dropout(0.2))  # Drop some of the layer outputs, increase robustness
        classifier.add(tf.keras.layers.Dense(units = 50, activation = 'relu'))
        classifier.add(tf.keras.layers.Dropout(0.2))  # Drop some of the layer outputs, increase robustness
        classifier.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))
        return classifier

    def run_task(self):
        """Train the model and evaluate it on the testing set."""
        # Extract training features and labels
        mouth_landmarks_train, train_labels = self.__extract_mouth_landmarks(self.training_directory)
        train_features = self.__create_dataset(mouth_landmarks_train)

        # Create model and train on the training set
        model = self.__create_mlp()
        model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        early_stopping = EarlyStopping(patience=10)  # Use early stopping to avoid overfitting
        epochs_hist = model.fit(train_features, np.array(train_labels), epochs = 500,
                                validation_split=0.2,  # Validation set used to prevent overfitting
                                callbacks=[early_stopping])  

        # Extract testing features and labels, and evaluate the model on them
        mouth_landmarks_test, test_labels = self.__extract_mouth_landmarks(self.testing_directory)
        test_features = self.__create_dataset(mouth_landmarks_test)
        print(model.evaluate(test_features, np.array(test_labels)))
