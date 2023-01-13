#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np


# In[18]:


import tensorflow as tf
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
import dlib
import re


# In[19]:


import cv2


# In[20]:


import pydot


# In[21]:


import graphviz


# # Define a few variables

# In[22]:


basedir = './dataset_AMLS_22-23/celeba'
images_dir = os.path.join(basedir,'img')
labels_filename = 'labels.csv'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# # Define a set of functions to be used to extract the features 

# In[23]:


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


# In[24]:


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


# In[25]:


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


# In[26]:


# This function was added by myself
def generate_dict_from_df(df, keys_col, values_col):
    zip_ = zip(df[keys_col], df[values_col])
    l = list(zip_)
    d = dict(l)
    return d


# In[27]:


def extract_features_labels():
    """
    This funtion extracts the landmarks features for all images in the folder 'dataset/celeba'.
    It also extract the gender label for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        gender_labels:      an array containing the gender label (male=0 and female=1) for each image in
                            which a face was detected
    """
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None
    #labels_file = open(os.path.join(basedir, labels_filename), 'r')
    #lines = labels_file.readlines()
    labels_df = pd.read_csv(os.path.join(basedir, labels_filename), sep='\t')
    gender_labels = generate_dict_from_df(labels_df, 'img_name', 'gender')
    #smiling_labels = generate_dict_from_df(labels_df, 'file_name', 'smiling')
    #gender_labels = {line.split(',')[0] : int(line.split(',')[6]) for line in lines[2:]}
    if os.path.isdir(images_dir):
        all_features = []
        all_gender_labels = []
        #all_smiling_labels = []
        image_files = []
        i = 1
        for img_path in image_paths:
            print(f'Iteration {i} out of {len(image_paths)}')
            #file_name= img_path.split('.')[1].split('/')[-1]
            #file_name = str(img_path).replace('\\\\', 'mmmm')
            new_file_name = re.search('img\\\\(.*)', img_path).group(1)
            #new_img_path = re.search('\\(.*).png', img_path)
            #print(file_name)
            #print(new_file_name)

            # load image
            img = tf.keras.utils.img_to_array(
                tf.keras.utils.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            features, _ = run_dlib_shape(img)
            if features is not None:
                all_features.append(features)
                all_gender_labels.append(gender_labels[new_file_name])
                #all_smiling_labels.append(smiling_labels[new_file_name])
                image_files.append(new_file_name)
            i += 1

    landmark_features = np.array(all_features)
    gender_labels = (np.array(all_gender_labels) + 1)/2 # simply converts the -1 into 0, so male=0 and female=1
    #smiling_labels = (np.array(all_smiling_labels) + 1)/2
    return landmark_features, gender_labels, image_files


# # Obtain the features and create the train, test, validate split

# In[23]:


X, y_gender, image_files = extract_features_labels()


# In[25]:


X.shape


# # Try to resize Image to make it simpler for CNN

# In[28]:


from PIL import Image


# In[14]:


image = Image.open('./dataset_AMLS_22-23/celeba/img/0.jpg')


# In[15]:


image_array = np.array(image)


# In[16]:


image_array.shape


# In[28]:


plt.imshow(image)


# In[33]:


resized_image = image.resize((50, 50))


# In[34]:


plt.imshow(resized_image)


# In[36]:


resized_image_array = np.array(resized_image)


# In[39]:


resized_image_array.shape


# In[40]:


image_grayscale = cv2.cvtColor(resized_image_array, cv2.COLOR_RGB2GRAY )


# In[41]:


plt.imshow(image_grayscale)


# In[44]:


image_grayscale


# # Try to create a dataset with all the 28x28 images

# In[29]:


df = pd.read_csv('./dataset_AMLS_22-23/celeba/labels.csv', sep='\t')
df.drop(df.columns[0], axis=1, inplace=True)
df


# In[196]:


test_df = pd.read_csv('./dataset_AMLS_22-23_test/celeba_test/labels.csv', sep='\t')
test_df.drop(test_df.columns[0], axis=1, inplace=True)
test_df


# In[12]:


row = int(df[df['img_name'] == '2.jpg']['gender'])
row


# In[13]:


image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]


# In[197]:


images = np.ndarray((5000, 28, 28))


# In[198]:


gender_labels = []


# In[199]:


for i in range(5000):
    print(f'Iteration {i} out of 5000')
    image = Image.open(f'./dataset_AMLS_22-23/celeba/img/{i}.jpg')
    resized_image = image.resize((28, 28))
    resized_image_array = np.array(resized_image)
    image_grayscale = cv2.cvtColor(resized_image_array, cv2.COLOR_RGB2GRAY)
    image_grayscale = image_grayscale/255
    images[i] = image_grayscale
    gender_labels.append(int((int(df[df['img_name'] == f'{i}.jpg']['gender']) + 1)/2))


# In[200]:


gender_labels


# In[201]:


images.shape


# In[202]:


images[4000:].shape


# # Split the Dataset into Train, Test and Validation Sets

# In[65]:


train_imgs = images[:4000]; test_imgs = images[4000:4500]; val_imgs = images[4500:];


# In[203]:


labels = np.array(gender_labels)


# In[204]:


labels


# In[68]:


train_labels = labels[:4000]; test_labels = labels[4000:4500]; val_labels = labels[4500:];


# In[56]:


val_labels.shape


# # Let's try to create a convolutional neural network

# In[220]:


from tensorflow.keras import datasets, layers, models
cnn = models.Sequential()

cnn.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
cnn.add(layers.MaxPooling2D(2, 2))

cnn.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
cnn.add(layers.MaxPooling2D(2, 2))

cnn.add(layers.Conv2D(64, (3, 3), activation = 'relu'))

cnn.add(layers.Flatten())

cnn.add(layers.Dense(64, activation = 'relu'))
cnn.add(layers.Dense(1, activation = 'sigmoid')) # 10 as there are 10 different items of clothing that can be found in the pictures
cnn.summary()


# In[221]:


cnn.compile(loss = 'binary_crossentropy', metrics = ['accuracy'], optimizer = 'Adam')


# In[222]:


labels = np.array(gender_labels)


# In[223]:


labels


# In[224]:


print(type(cnn))


# In[225]:


from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=10)


# In[226]:


epochs = 50

history = cnn.fit(x=images.reshape(images.shape[0], *(28, 28, 1)), y=labels, batch_size = 512, epochs = epochs,
                  validation_split=0.2, callbacks=[early_stopping])


# In[294]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(20, 7))

# summarize history for accuracy
ax[0].plot(history.history['accuracy'])
ax[0].plot(history.history['val_accuracy'])
#ax[0].set_title('odel accuracy')
ax[0].set_ylabel('Accuracy', fontsize=25)
ax[0].set_xlabel('Epoch', fontsize=25)
ax[0].legend(['Train', 'Validation'], loc='lower right', fontsize=25)
ax[0].grid()

for label in (ax[0].get_xticklabels() + ax[0].get_yticklabels()):
	label.set_fontsize(19)


# summarize history for loss
ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
#ax[1].set_title('model loss')
ax[1].set_ylabel('Loss', fontsize=16)
ax[1].set_xlabel('Epoch', fontsize=16)
ax[1].legend(['Train', 'Validation'], loc='upper right', fontsize=16)
ax[1].grid()

for label in (ax[1].get_xticklabels() + ax[1].get_yticklabels()):
	label.set_fontsize(14)

    
plt.show()


# In[228]:


test_labels = []
test_images = np.ndarray((1000, 28, 28))
for i in range(1000):
    print(f'Iteration {i+1} out of 2500')
    image = Image.open(f'./dataset_AMLS_22-23_test/celeba_test/img/{i}.jpg')
    resized_image = image.resize((28, 28))
    resized_image_array = np.array(resized_image)
    image_grayscale = cv2.cvtColor(resized_image_array, cv2.COLOR_RGB2GRAY)
    image_grayscale = image_grayscale/255
    test_images[i] = image_grayscale
    test_labels.append(int((int(test_df[test_df['img_name'] == f'{i}.jpg']['gender']) + 1)/2))


# # Evaluate the performance of the model on the test set 

# In[229]:


pred_test_labels = cnn.predict(test_images.reshape(test_images.shape[0], *(28, 28, 1)))


# In[230]:


pred_test_labels = (pred_test_labels > 0.5)


# In[231]:


pred_test_labels


# In[232]:


from sklearn.metrics import accuracy_score


# In[233]:


acc = accuracy_score(test_labels, pred_test_labels)


# In[234]:


acc


# In[171]:


test_labels


# In[262]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels, pred_test_labels)
plt.figure(figsize = (6, 3))
plt.xlabel('True Labels')
ax = sns.heatmap(cm, annot = True)
ax.set_xlabel('True Labels', fontsize=16)
ax.set_ylabel('Predicted Labels', fontsize=16)
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(14)


# In[237]:


from sklearn.metrics import classification_report


# In[239]:


print(classification_report(test_labels, pred_test_labels))


# In[47]:


sns.countplot(df['gender'], label='Count')


# In[ ]:




