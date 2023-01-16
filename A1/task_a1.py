import pandas as pd
import numpy as np
import cv2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from PIL import Image

class TaskA1Evaluator:
    """A class that is used to run and evaluate the proposed task A1 solution."""


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

    def __create_dataset_from_images(self, directory):
        """
        Generate a couple of numpy arrays: one containing the normalized pixel values of
        every image in the dataset, the other containing the gender labels for each image.
        The images will be resized to 28x28 (from 218x178).

        Arguments
        ---------
        directory : str
            Directory from which the images and labels will be extracted. It could either be the
            training or testing directory.
        
        Returns
        -------
        images : numpy.ndarray
            Array containing the normalized pixel values of the resized images.
        labels : numpy.ndarray
            Numpy array containing the gender labels for every image (1 for male, 0 for female).

        """
        df = self.__create_dataframe(directory)

        # Initialize the arrays
        images = np.ndarray((len(df), 28, 28))
        labels = np.ndarray((len(df),))

        for index, row in df.iterrows():  # Go through every image on the dataframe
            print(f'Working on image {index+1} out of {len(df)}')
            image = Image.open(f'{directory}/img/{row["img_name"]}')
            resized_image = image.resize((28, 28))
            resized_image_array = np.array(resized_image)
            image_grayscale = cv2.cvtColor(resized_image_array, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
            image_grayscale = image_grayscale/255  # Normalize the images

            # Insert values into the arrays
            images[index] = image_grayscale
            labels[index] = 1 if row['gender'] == 1 else 0
        
        return images, labels


    def __create_cnn_model(self):
        """Define the Convolutional Neural Network (CNN) to be used for the task."""
        model = models.Sequential()
        
        # Add the convolutional and max pooling layers to extract features from the images
        model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))

        # Flatten the features and use a couple of dense layers for the classification
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation = 'relu'))
        model.add(layers.Dense(1, activation = 'sigmoid'))

        return model

    def run_task(self):
        """Train the model and evaluate it on the testing set."""
        train_images, train_labels = self.__create_dataset_from_images(self.training_directory)
        model = self.__create_cnn_model()
        early_stopping = EarlyStopping(patience=10)  # Define early stopping to prevent overfitting

        # Compile and fit the model to the training set
        model.compile(loss = 'binary_crossentropy', metrics = ['accuracy'], optimizer = 'Adam')
        model.fit(x=train_images.reshape(train_images.shape[0], *(28, 28, 1)), y=train_labels, batch_size = 512, epochs = 100,
                  validation_split=0.2, callbacks=[early_stopping])  # validation is also used to avoid overfitting

        # Evaluate the model on the testing set
        test_images, test_labels = self.__create_dataset_from_images(self.testing_directory)
        pred_test_labels = model.predict(test_images.reshape(test_images.shape[0], *(28, 28, 1)))
        pred_test_labels = (pred_test_labels > 0.5)
        print(f'The accuracy obtained on the testing dataset is: {accuracy_score(test_labels, pred_test_labels)}')
        



    

