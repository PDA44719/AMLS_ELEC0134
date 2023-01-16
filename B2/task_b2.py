import pandas as pd
import numpy as np
from PIL import Image
from facial_regions_detector import run_dlib_shape
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score


class TaskB2Evaluator:
    """A class that is used to run and evaluate the proposed task B2 solution."""


    def __init__(self):
        """Create an instance of the class."""
        self.training_directory = "./Datasets/cartoon_set"
        self.testing_directory = "./Datasets/cartoon_set_test"

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

    def __resize_image(self, img_array):
        """
        Resize an image from its original size to 5x5. With the resized image, less computation
        is required.

        Arguments
        ---------
        img_array : numpy.ndarry
            An array containing the RBG values of the cropped image containing the eye.

        Returns
        -------
        numpy.ndarray
            The resized 5x5 RBG values of the cropped image.

        """
        pil_image = Image.fromarray(img_array)
        resized_image = pil_image.resize((5, 5))
        return np.array(resized_image)   

    def __crop_and_resize_image(self, directory, image_file):
        """
        Crop the original avatar image (to obtain the right eye of the avatar),
        resize it to a 5x5 size, flatten it and normalize it.
        
        Arguments
        ---------
        directory : str
            Directory containing the image to be cropped. The image could be from either 
            the training or testing directory.
        image_file : str
            The name of the image to be cropped and resized.
        
        Returns
        -------
        numpy.ndarray
            The array containing the 75 normalized pixel values (5x5x3 images) from
            the cropped image. Those values correspond to the RBG values of the right
            eye of the avatar.

        """
        # Get the 68 landmarks from the image
        image = Image.open(f'{directory}/img/{image_file}')
        image_array = np.array(image)
        features, _ = run_dlib_shape(image_array)

        # Obtain the horizontal and vertical coordinates of 4 right eye landmarks
        eye_features = [features[37], features[38], features[40], features[41]]
        horizontal_coordinates = [element[0] for element in eye_features]
        vertical_coordinates = [element[1] for element in eye_features]

        # From those coordinates, get the max and min values to crop the image
        eyes_crop = [(min(horizontal_coordinates), max(horizontal_coordinates)),
                     (min(vertical_coordinates), max(vertical_coordinates))]
        image_array = image_array[eyes_crop[1][0]:eyes_crop[1][1], eyes_crop[0][0]:eyes_crop[0][1], :]

        resized_image_arr = self.__resize_image(image_array)  # Resize the cropped image

        # Flatten and normalize the RBG values
        return resized_image_arr[:, :, 0:3].reshape((1, 75))/255

    def __extract_eyes_and_labels(self, df, directory):
        """
        From all the images, crop their right eyes (to be used for classifying the eye color)
        and get their labels. If the dlib detector is not able to extract any features, the
        image will be ignored.
        
        Arguments
        ---------
        df : pandas.core.frame.DataFrame
            The dataframe containing the information about the images and eye color labels.
        directory : str
            The directory containing the images whose eyes and labels are going to be extracted.
        
        Returns
        -------
        eyes_list : list
            A list containing the RBG values of the right eye of the avatars.
        labels_list : list
            A list containing the labels of the images that the detector was able to identify.
        
        """
        eyes_list = []
        labels_list = []
        for index, row in df.iterrows():  # go through every image on the dataframe
            print(f'Iteration number: {index+1} out of {len(df)}')
            try:  # Try to crop and resize the image
                eye_array = self.__crop_and_resize_image(directory, row['file_name'])
                eyes_list.append(eye_array)
                labels_list.append(row['eye_color'])

            # If the detector could not extract any features from the image, go to the next
            except:
                continue

        return eyes_list, labels_list

    def __convert_lists_to_arrays(self, features_list, labels_list):
        """
        Convert the extracted features and labels lists into a numpy array.
        
        Arguments
        ---------
        features_list : list
            A list containing the normalized RBG values of the avatars' eyes.
        labels_list : list
            A list containing the eye color labels for the cropped eye images.
        
        Returns
        -------
        features_array : numpy.ndarray
            The features converted into a numpy array.
        labels_array : numpy.ndarray
            The labels converted into a numpy array.
        
        """
        features_array = np.array(features_list).reshape((len(features_list), 75))
        labels_array = np.array(labels_list)
        return features_array, labels_array

    def __fit_and_predict(self, train_features, train_labels, test_features):
        """
        Fit the bagging classifier using the train features and labels, and make predictions on 
        the testing set.

        Argument
        --------
        train_features : numpy.ndarray
            Normalized RGB values of the avatars' eyes. They were extracted from the train set.
        train_labels : numpy.ndarray
            Label indicating the eye color of the avatars.
        test_features : numpy.ndarray
            Normalized RGB values of the avatars' eyes. They were extracted from the test set.

        Returns
        -------
        numpy.ndarray
            An array containing the predictions made by the bagging model

        """
        bagmodel = BaggingClassifier(n_estimators=50,max_samples=0.8, max_features=30,random_state=2)
        bagmodel.fit(train_features, train_labels) 
        predictions = bagmodel.predict(test_features)
        return predictions

    def run_task(self):
        """Train the model and evaluate it on the testing set."""
        # Extract the eye values and labels for the training set
        train_df = self.__create_dataframe(self.training_directory)
        train_eyes, train_labels = self.__extract_eyes_and_labels(train_df, self.training_directory)
        train_features, train_labels = self.__convert_lists_to_arrays(train_eyes, train_labels)

        # Extract the eye values and labels for the testing set
        test_df = self.__create_dataframe(self.testing_directory)
        test_eyes, test_labels = self.__extract_eyes_and_labels(test_df, self.testing_directory)
        test_features, test_labels = self.__convert_lists_to_arrays(test_eyes, test_labels)

        # Evaluate the performance on the testing set
        predictions = self.__fit_and_predict(train_features, train_labels, test_features)
        print(f'The accuracy obtained on the teting set is: {accuracy_score(test_labels, predictions)}')
    
