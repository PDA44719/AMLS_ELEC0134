import os
import numpy as np
import tensorflow as tf
import cv2
import dlib
import re
import pandas as pd

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image

def generate_dict_from_df(df, keys_col, values_col):
    """Generate a dictionary containing a key and a value both extracted from df."""
    zip_ = zip(df[keys_col], df[values_col])
    l = list(zip_)
    d = dict(l)
    return d

def extract_features_labels(basedir, images_dir, labels_filename):
    """
    This funtion extracts the landmarks features for all images in a specific directory.
    It also extract the face shape label for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        fs_labels:          an array containing the face shape label for each image in which a face was detected
        image_files:        a list containing the images in which a face was detected
    """
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None
    labels_df = pd.read_csv(os.path.join(basedir, labels_filename), sep='\t')

    # Create a dictionary from which the labels will be obtained
    fs_labels = generate_dict_from_df(labels_df, 'file_name', 'face_shape')
    if os.path.isdir(images_dir):
        all_features = []
        all_fs_labels = []
        image_files = []
        i = 1  # Counter used to know the iteration number
        for img_path in image_paths:
            print(f'Iteration {i} out of {len(image_paths)}')

            # Extract only the file name from the path
            new_file_name = re.search('img\\\\(.*)', img_path).group(1)

            # load image
            img = tf.keras.utils.img_to_array(
                tf.keras.utils.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            features, _ = run_dlib_shape(img)
            if features is not None:
                all_features.append(features)
                all_fs_labels.append(fs_labels[new_file_name])
                image_files.append(new_file_name)
            i += 1

    landmark_features = np.array(all_features)
    fs_labels = np.array(all_fs_labels)
    return landmark_features, fs_labels, image_files

