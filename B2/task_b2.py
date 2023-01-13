#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np


# In[2]:


import tensorflow as tf
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random


# In[3]:


import cv2
import os
import dlib
import re


# In[4]:


basedir = './dataset_AMLS_22-23/cartoon_set'
images_dir = os.path.join(basedir,'img')
labels_filename = 'labels.csv'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# In[5]:


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


# In[6]:


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


# In[7]:


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


# In[8]:


# This function was added by myself
def generate_dict_from_df(df, keys_col, values_col):
    zip_ = zip(df[keys_col], df[values_col])
    l = list(zip_)
    d = dict(l)
    return d


# In[9]:


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
    fs_labels = generate_dict_from_df(labels_df, 'file_name', 'face_shape')
    #smiling_labels = generate_dict_from_df(labels_df, 'file_name', 'smiling')
    #gender_labels = {line.split(',')[0] : int(line.split(',')[6]) for line in lines[2:]}
    if os.path.isdir(images_dir):
        all_features = []
        all_fs_labels = []
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
                all_fs_labels.append(fs_labels[new_file_name])
                #all_smiling_labels.append(smiling_labels[new_file_name])
                image_files.append(new_file_name)
            i += 1

    landmark_features = np.array(all_features)
    fs_labels = (np.array(all_fs_labels) + 1)/2 # simply converts the -1 into 0, so male=0 and female=1
    #smiling_labels = (np.array(all_smiling_labels) + 1)/2
    return landmark_features, fs_labels, image_files


# In[10]:


cartoon_df = pd.read_csv('dataset_AMLS_22-23/cartoon_set/labels.csv', sep='\t')


# In[11]:


cartoon_df.drop(cartoon_df.columns[0], axis=1, inplace=True)


# In[95]:


cartoon_df.iloc[5]


# In[13]:


cartoon_df.iloc[:,0].values


# In[14]:


image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]


# In[15]:


image_paths


# In[16]:


len(image_paths)


# In[13]:


get_ipython().run_line_magic('store', '-r')


# In[18]:


X


# In[19]:


img_array = tf.keras.utils.img_to_array(
                    tf.keras.utils.load_img(f'dataset_AMLS_22-23/cartoon_set/img/2.png',
                                   target_size=None,
                                   interpolation='bicubic'))


# In[20]:


img_array.shape


# In[21]:


image_cv2_array = cv2.imread("dataset_AMLS_22-23/cartoon_set/img/2.png")


# In[22]:


image_cv2_array.shape


# In[23]:


features, _ = run_dlib_shape(img_array)


# In[24]:


features


# In[25]:


right_eye = features[36:42]


# In[26]:


right_eye.shape


# In[27]:


right_eye[0]


# In[28]:


right_eye


# In[29]:


right_eye_side1 = features[36]


# In[30]:


right_eye_side1


# In[31]:


right_eye_side2 = features[39]


# In[32]:


right_eye_side2


# In[33]:


right_eye_side1[0]


# In[34]:


image_rgb = cv2.cvtColor(image_cv2_array, cv2.COLOR_BGR2RGB)


# In[35]:


plt.imshow(image_rgb[255:271, 190:222, :].reshape((10, 10)))


# In[ ]:


image_rgb[int(right_eye_side1[0]), right_eye_side1[1]+1, :]


# In[36]:


image_rgb.shape


# In[37]:


plt.imshow(image_rgb)


# In[38]:


image_rgb[190:222, 255:271, :].shape


# In[39]:


img_array2 = tf.keras.utils.img_to_array(
                    tf.keras.utils.load_img(f'dataset_AMLS_22-23/cartoon_set/img/4.png',
                                   target_size=None,
                                   interpolation='bicubic'))


# In[40]:


image_cv2_array2 = cv2.imread("dataset_AMLS_22-23/cartoon_set/img/4.png")


# In[41]:


features2, _ = run_dlib_shape(img_array2)


# In[42]:


right_eye = features2[36:42]


# In[43]:


right_eye


# In[44]:


image_rgb2 = cv2.cvtColor(image_cv2_array2, cv2.COLOR_BGR2RGB)


# In[45]:


plt.imshow(image_rgb2[250:277, 190:227, :])


# In[46]:


img_array3 = tf.keras.utils.img_to_array(
                    tf.keras.utils.load_img(f'dataset_AMLS_22-23/cartoon_set/img/5.png',
                                   target_size=None,
                                   interpolation='bicubic'))


# In[47]:


image_cv2_array3 = cv2.imread("dataset_AMLS_22-23/cartoon_set/img/5.png")


# In[48]:


features3, _ = run_dlib_shape(img_array3)


# In[49]:


right_eye = features3[36:42]


# In[50]:


right_eye


# In[51]:


image_rgb3 = cv2.cvtColor(image_cv2_array3, cv2.COLOR_BGR2RGB)


# In[52]:


plt.imshow(image_rgb3[250:277, 190:227, :])


# In[53]:


plt.imshow(image_rgb3[250:277, 191:229, :])


# In[54]:


right_eye


# In[55]:


elements_left_column = [element for element in right_eye[:,0]]
elements_left_column


# In[56]:


max_n_min_l = (max(elements_left_column), min(elements_left_column))
max_n_min_l


# In[57]:


X.shape


# # Function to extract eye from an image

# In[14]:


from PIL import Image


# In[15]:


def get_eye(image_file, iteration):
    img_array = cv2.imread(f'dataset_AMLS_22-23/cartoon_set/img/{image_file}')
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    right_eye = img_array[250:277, 190:227, :]
    print(f'Completed eye {iteration}')
    return right_eye.reshape((1, 2997))


# In[16]:


image_files


# In[17]:


from matplotlib.pyplot import figure


# In[18]:


image = Image.open(f'dataset_AMLS_22-23/cartoon_set/img/1019.png')
image_array = np.array(image)
features,_ = run_dlib_shape(image_array)
eye_features = [features[37], features[38], features[40], features[41]]
horizontal_pixels = [element[0] for element in eye_features]
vertical_pixels = [element[1] for element in eye_features] 
iris_crop = [(min(horizontal_pixels), max(horizontal_pixels)), (min(vertical_pixels), max(vertical_pixels))]
figure(figsize=(8, 6), dpi=80)
plt.plot(eye_features[0][0], eye_features[0][1], marker = 'o', color='red')
plt.plot(eye_features[1][0], eye_features[1][1], marker = 'o', color='red')
plt.plot(eye_features[2][0], eye_features[2][1], marker = 'o', color='red')
plt.plot(eye_features[3][0], eye_features[3][1], marker = 'o', color='red')
plt.imshow(image_array)
#image_array[210:221, 254:268, :]
#plt.imshow(image_array[iris_crop[0][0]:iris_crop[0][1], iris_crop[1][0]:iris_crop[1][1], :])


# In[118]:


plt.imshow(image_array[255:270, 201:216])


# In[74]:


iris_crop


# In[15]:


def resize_image(img_array):
    pil_image = Image.fromarray(img_array)
    resized_image = pil_image.resize((5, 5))
    return np.array(resized_image)


# In[16]:


def get_eye_2(image_file):
    image = Image.open(f'dataset_AMLS_22-23/cartoon_set/img/{image_file}')
    image_array = np.array(image)
    features,_ = run_dlib_shape(image_array)
    eye_features = [features[37], features[38], features[40], features[41]]
    horizontal_pixels = [element[0] for element in eye_features]
    vertical_pixels = [element[1] for element in eye_features] 
    iris_crop = [(min(horizontal_pixels), max(horizontal_pixels)), (min(vertical_pixels), max(vertical_pixels))]
    image_array = image_array[iris_crop[1][0]:iris_crop[1][1], iris_crop[0][0]:iris_crop[0][1], :]
    resized_image_arr = resize_image(image_array)
    """
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    right_eye = img_array[250:277, 190:227, :]
    print(f'Completed eye {iteration}')
    return right_eye.reshape((1, 2997))
    """
    return resized_image_arr[:, :, 0:3].reshape((1, 75))/255


# In[ ]:





# In[16]:


image = Image.open(f'dataset_AMLS_22-23/cartoon_set/img/1031.png')


# In[17]:


image.mode


# In[18]:


im_arr = np.array(image)
im_arr.shape


# In[26]:


image = get_eye_2('4.png')


# In[27]:


plt.imshow(image.reshape((5, 5, 3)))


# In[28]:


plt.imshow(Image.open(f'dataset_AMLS_22-23/cartoon_set/img/4.png'))


# In[20]:


image


# In[21]:


image.shape


# In[22]:


plt.imshow(image.reshape((5, 5, 3)))


# In[125]:


pil_image = Image.fromarray(image)


# In[54]:


type(image)


# In[126]:


plt.imshow(image)


# In[128]:


resized_image = pil_image.resize((10, 10))


# In[129]:


plt.imshow(resized_image)


# In[16]:


cartoon_df


# In[16]:


lol = 4


# In[17]:


int(cartoon_df[cartoon_df['file_name'] == f'{lol}.png']['eye_color'])


# # Let's create a dataset comprised of the iris images

# In[ ]:


iris_list = []
labels_list = []
for i in range(10000):
    print(f'Iteration number: {i} out of 10000')
    try:
        iris_array = get_eye_2(f'{i}.png')
        iris_list.append(iris_array)
        labels_list.append(int(cartoon_df[cartoon_df['file_name'] == f'{i}.png']['eye_color']))
    except:
        continue


# In[22]:


get_ipython().run_line_magic('store', 'iris_list')


# In[23]:


get_ipython().run_line_magic('store', 'labels_list')


# In[29]:


get_ipython().run_line_magic('store', '-r')


# In[19]:


iris_list[1].shape


# In[30]:


len(labels_list)


# In[17]:


dataset = np.ndarray((8194, 75))


# In[18]:


dataset


# In[60]:


for i in range(len(iris_list)):
    print(f'{i} out of 8194')
    dataset[i] = iris_list[i]


# In[61]:


dataset


# In[62]:


from sklearn.decomposition import PCA


# In[63]:


pca = PCA(n_components=75)


# In[64]:


pca.fit(dataset)


# In[65]:


s_pca=pca.transform(dataset)


# In[66]:


s_pca


# In[67]:


features = s_pca[:, :5]
features


# In[68]:


len(features)


# In[69]:


features[:, 1]


# In[70]:


labels = np.array(labels_list)


# In[71]:


n = 5
fig, ax = plt.subplots(n, n, figsize=(20, 20))

for i in range(n):
    for j in range(n):
        ax[j][i].scatter(x=features[:, i], y=features[:, j], c=labels, s=1, cmap='CMRmap')
        ax[j][i].set_xlabel(f'PC{i+1}')
        ax[j][i].set_ylabel(f'PC{j+1}')
plt.show()


# In[76]:


fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].scatter(x=features[:, 2], y=features[:, 3], c=labels, s=1, cmap='CMRmap')
ax[0].set_xlabel(f'PC 3', fontsize=20)
ax[0].set_ylabel(f'PC 4', fontsize=20)

for label in (ax[0].get_xticklabels() + ax[0].get_yticklabels()):
	label.set_fontsize(14)

ax[1].scatter(x=features[:, 4], y=features[:, 0], c=labels, s=1, label=labels, cmap='CMRmap')
ax[1].set_xlabel(f'PC 5', fontsize=20)
ax[1].set_ylabel(f'PC 1', fontsize=20)

for label in (ax[1].get_xticklabels() + ax[1].get_yticklabels()):
	label.set_fontsize(25)

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)


plt.show()


# In[34]:


labels = np.array(labels_list)
labels


# In[35]:


training_set = dataset[:7374]; testing_set = dataset[7374:];
training_labels = labels[:7374]; testing_labels = labels[7374:];


# # Let's train a model

# In[73]:


# This model was extracted from Project 6.ipynb (online course)
classifier = tf.keras.models.Sequential()
classifier.add(tf.keras.layers.Dense(units = 400, activation = 'relu', input_shape = (75,)))
classifier.add(tf.keras.layers.Dropout(0.2)) # Drop some neurons, along with their weights, to avoid overfitting the data

classifier.add(tf.keras.layers.Dense(units = 400, activation = 'relu'))
classifier.add(tf.keras.layers.Dropout(0.2))

classifier.add(tf.keras.layers.Dense(units = 5, activation = 'softmax'))


# In[74]:


classifier.compile(optimizer = 'Adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


# In[75]:


training_labels


# In[78]:


epochs_hist = classifier.fit(training_set, training_labels, epochs = 100, validation_split=0.2)


# In[79]:


evaluation = classifier.evaluate(testing_set, testing_labels)


# In[27]:


def get_eye_3(image_file):
    image = Image.open(f'dataset_AMLS_22-23_test/cartoon_set_test/img/{image_file}')
    image_array = np.array(image)
    features,_ = run_dlib_shape(image_array)
    eye_features = [features[37], features[38], features[40], features[41]]
    horizontal_pixels = [element[0] for element in eye_features]
    vertical_pixels = [element[1] for element in eye_features] 
    iris_crop = [(min(horizontal_pixels), max(horizontal_pixels)), (min(vertical_pixels), max(vertical_pixels))]
    image_array = image_array[iris_crop[1][0]:iris_crop[1][1], iris_crop[0][0]:iris_crop[0][1], :]
    resized_image_arr = resize_image(image_array)
    """
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    right_eye = img_array[250:277, 190:227, :]
    print(f'Completed eye {iteration}')
    return right_eye.reshape((1, 2997))
    """
    return resized_image_arr[:, :, 0:3].reshape((1, 75))/255


# In[28]:


cartoontest_df = pd.read_csv('dataset_AMLS_22-23_test/cartoon_set_test/labels.csv', sep='\t')

cartoontest_df.drop(cartoontest_df.columns[0], axis=1, inplace=True)

cartoontest_df


# In[45]:


iris_test = []
labels_test = []
images_used_test = []
for i in range(2500):
    print(f'Iteration number: {i} out of 10000')
    try:
        iris_array = get_eye_3(f'{i}.png')
        iris_test.append(iris_array)
        labels_test.append(int(cartoontest_df[cartoontest_df['file_name'] == f'{i}.png']['eye_color']))
        images_used_test.append(f'{i}')
    except:
        continue


# In[30]:


get_ipython().run_line_magic('store', 'iris_test')


# In[31]:


get_ipython().run_line_magic('store', 'labels_test')


# In[46]:


get_ipython().run_line_magic('store', 'images_used_test')


# In[32]:


len(iris_test)


# In[33]:


testdataset = np.ndarray((2041, 75))


# In[34]:


testdataset


# In[35]:


for i in range(len(iris_test)):
    print(f'{i} out of 8194')
    testdataset[i] = iris_test[i]


# In[36]:


testdataset


# In[42]:


test_labels = np.array(labels_test)


# # Let's try some other simpler models

# ## Bagging classfier

# In[37]:


from sklearn.ensemble import BaggingClassifier


# In[97]:


bagmodel=BaggingClassifier(n_estimators=50,max_samples=0.8, max_features=30,random_state=2)


# In[98]:


bagmodel.fit(dataset, labels)


# In[99]:


predicted_labels = bagmodel.predict(testdataset)


# In[100]:


from sklearn import metrics
score=metrics.accuracy_score(test_labels, predicted_labels)
print(score)


# In[102]:


from sklearn.metrics import confusion_matrix
sns.set(font_scale=1.5) # Adjust to fit
cm = confusion_matrix(test_labels, predicted_labels)
plt.figure(figsize = (10, 8))
plt.xlabel('True Labels', fontsize=18)
ax = sns.heatmap(cm, annot = True)
ax.set_xlabel('Predicted Labels', fontsize=18)
ax.set_ylabel('True Labels', fontsize=18)
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(18)


# In[85]:


glass_wearers = [
    "5",
    "6",
    "12",
    "13",
    "14",
    "22",
    "27",
    "29",
    "31",
    "44",
    "49",
    "51",
    "53",
    "57",
    "64",
    "77",
    "96",
    "98",
    "100",
    "103",
    "107",
    "110",
    "123",
    "126",
    "145",
    "148",
    "168",
    "170",
    "182",
    "189",
    "295",
    "201",
    "211",
    "215",
    "216",
    "219",
    "222",
    "234",
    "237",
    "243",
    "264",
    "270",
    "272",
    "279",
    "291",
    "296",
    "309",
    "315",
    "322",
    "329",
    "338",
    "346",
    "247",
    "352",
    "354",
    "358",
    "359",
    "364",
    "369",
    "379",
    "394",
    "397",
    "398",
    "400",
    "414",
    "417",
    "423",
    "433",
    "437",
    "443",
    "444",
    "446",
    "449",
    "455",
    "459",
    "464",
    "473",
    "479",
    "491",
    "503",
    "508",
    "520",
    "521",
    "525",
    "427",
    "528",
    "534",
    "539",
    "545",
    "551",
    "554",
    "560",
    "561",
    "565",
    "566",
    "571",
    "582",
    "585",
    "586",
    "587",
    "592",
    "596",
    "597",
    "616",
    "617",
    "618",
    "622",
    "631",
    "637",
    "645",
    "649",
    "654",
    "660",
    "668",
    "675",
    "684",
    "685",
    "689",
    "694",
    "706",
    "711",
    "714",
    "719",
    "725",
    "749",
    "750",
    "751",
    "753",
    "755",
    "761",
    "767",
    "771",
    "793",
    "798",
    "803",
    "811",
    "812",
    "818",
    "820",
    "827",
    "832",
    "833",
    "835",
    "874",
    "878",
    "880",
    "891",
    "904",
    "907",
    "910",
    "919",
    "935",
    "938",
    "941",
    "944",
    "956",
    "957",
    "959",
    "960",
    "962",
    "966",
    "971",
    "973",
    "974",
    "980",
    "985",
    "987",
    "991"]


# In[86]:


predicted_labels = bagmodel.predict(testdataset[:1000])


# In[49]:


from sklearn import metrics
score=metrics.accuracy_score(test_labels[:1000], predicted_labels)
print(score)


# In[87]:


without_glasses_features = []
without_glasses_labels = []
for i in range(1000):
    if images_used_test[i] not in glass_wearers:
        without_glasses_features.append(testdataset[i])
        without_glasses_labels.append(test_labels[i])


# In[88]:


len(without_glasses_features)


# In[89]:


without_glasses_features = np.array(without_glasses_features)
without_glasses_labels = np.array(without_glasses_labels)


# In[90]:


predicted_labels_test = bagmodel.predict(without_glasses_features)


# In[91]:


score=metrics.accuracy_score(without_glasses_labels, predicted_labels_test)
print(score)


# In[93]:


from sklearn.metrics import confusion_matrix
sns.set(font_scale=1.5) # Adjust to fit
cm = confusion_matrix(without_glasses_labels, predicted_labels_test)
plt.figure(figsize = (10, 8))
plt.xlabel('True Labels')
ax = sns.heatmap(cm, annot = True)
ax.set_xlabel('Predicted Labels', fontsize=18)
ax.set_ylabel('True Labels', fontsize=18)
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(18)


# ## SVM

# In[87]:


from sklearn.svm import LinearSVC


# In[91]:


from sklearn.metrics import accuracy_score


# In[99]:


svm_classifier = LinearSVC(C=0.6)


# In[100]:


svm_classifier.fit(training_set, training_labels)


# In[101]:


y_pred=svm_classifier.predict(testing_set)
print(accuracy_score(testing_labels,y_pred))


# In[ ]:





# ## Boosting

# In[102]:


from sklearn.ensemble import AdaBoostClassifier


# In[118]:


boostmodel=AdaBoostClassifier(n_estimators=50)


# In[119]:


boostmodel.fit(training_set, training_labels, sample_weight=None) # Fit KNN model


# In[120]:


pred_labels_boost = boostmodel.predict(testing_set)


# In[121]:


score1=metrics.accuracy_score(testing_labels, pred_labels_boost)


# In[122]:


score1


# ## KNN

# In[157]:


from sklearn.neighbors import KNeighborsClassifier


# In[158]:


def KNNClassifier(X_train, y_train, X_test,k):

    #Create KNN object with a K coefficient
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train) # Fit KNN model


    Y_pred = neigh.predict(X_test)
    return Y_pred

Y_pred=KNNClassifier(training_set, training_labels, testing_set, 85)

score=metrics.accuracy_score(testing_labels,Y_pred)
print(score)


# ## Decision-Tree

# We will import the DecisionTreeClassifier function from the sklearn library. Recall that we need to specify the attribute selection measures for Decision-Tree algorithm. In this experiment, we will set this '**criterion**' to 'entropy', i.e., information gain.

# In[159]:


from sklearn import tree


# In[160]:


#Importing the Decision tree classifier from the sklearn library.
tree_params={
    'criterion':'entropy'
}
clf = tree.DecisionTreeClassifier( **tree_params )


# Next, we will train and test the classifier on the corresponding dataset. You will be asked to complete the train and test function.

# In[162]:


#Training the decision tree classifier on training set. 
# Please complete the code below.
clf.fit(training_set, training_labels)


#Predicting labels on the test set.
# Please complete the code below.
y_pred =  clf.predict(testing_set)


# We will now evaluate the predicted classes using some metrics. For this case, we will use 'accuracy_score' to calculate the accuracy of the predicted labels.

# In[165]:


#Use accuracy metric from sklearn.metrics library
print('Accuracy Score on train data: ', accuracy_score(y_true=training_labels, y_pred=clf.predict(training_set)))
print('Accuracy Score on test data: ', accuracy_score(y_true=testing_labels, y_pred=y_pred))


# In[28]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(testing_labels, predicted_labels)
plt.figure(figsize = (20, 10))
sns.heatmap(cm, annot = True)


# In[44]:


images_with_features = []
for i in range(10000):
    print(f'Doing image {i} out of 10000')
    image = Image.open(f'dataset_AMLS_22-23/cartoon_set/img/{i}.png')
    image_array = np.array(image)
    features,_ = run_dlib_shape(image_array)
    if features is not None:
        images_with_features.append(f'{i}.png')


# In[45]:


get_ipython().run_line_magic('store', 'images_with_features')


# In[42]:


image = Image.open(f'dataset_AMLS_22-23/cartoon_set/img/1006.png')
image_array = np.array(image)
features,_ = run_dlib_shape(image_array)


# In[43]:


features is not None


# In[31]:


get_ipython().run_line_magic('store', 'images_used')


# In[35]:


len(images_used)


# In[29]:


len(images_with_features)


# In[30]:


training_images = images_with_features[:7374]; testing_images = images_with_features[7374:];


# In[31]:


len(testing_images)


# In[32]:


images_mislabelled = []
for i in range(len(testing_set)):
    if predicted_labels[i] != testing_labels[i]:
        images_mislabelled.append(testing_images[i])


# In[33]:


(820 - len(images_mislabelled)) / 820


# In[34]:


len(images_mislabelled)


# In[56]:


predicted_labels[0] != testing_labels[1]


# In[54]:


testing_labels[0]


# In[36]:


import cv2
from matplotlib import pyplot as plt

# create figure
fig = plt.figure(figsize=(25, 52))

# setting values to rows and column variables
rows = 19
columns = 6

# reading images

images_bgr = []
for image in images_mislabelled:
    images_bgr.append(cv2.imread("dataset_AMLS_22-23/cartoon_set/img/" + image))


images_rgb = []
for im in images_bgr:
    # convert bgr to rgb 
    images_rgb.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    
    
for i in range(len(images_mislabelled)):
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(images_rgb[i])
    plt.axis('off')


# In[37]:


testing_images


# In[45]:


sunglasses = []
for image in testing_images:
    plt.imshow(Image.open(f'dataset_AMLS_22-23/cartoon_set/img/{image}'))
    plt.show()
    wearing_sunglasses = input('Is it wearing sunglasses?')
    if wearing_sunglasses == "1":
        sunglasses.append(image)


# In[46]:


sunglasses


# In[48]:


import cv2
from matplotlib import pyplot as plt

# create figure
fig = plt.figure(figsize=(25, 500))

# setting values to rows and column variables
rows = 137
columns = 6

# reading images

images_bgr = []
for image in testing_images:
    images_bgr.append(cv2.imread("dataset_AMLS_22-23/cartoon_set/img/" + image))


images_rgb = []
for im in images_bgr:
    # convert bgr to rgb 
    images_rgb.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    
    
for i in range(len(testing_images)):
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(images_rgb[i])
    plt.axis('off')
    plt.title(testing_images[i])


# In[49]:


testing_set


# In[50]:


modified_testing_set = testing_set


# In[51]:


modified_testing_set


# In[62]:


modified_testing_list = modified_testing_set.tolist()


# In[63]:


modified_testing_list


# In[61]:


with_sunglasses = [
    '9037.png',
    '9039.png',
    '9049.png',
    '9059.png',
    '9070.png',
    '9071.png',
    '9092.png',
    '9093.png',
    '9100.png',
    '9110.png',
    '9119.png',
    '9122.png',
    '9145.png',
    '9161.png',
    '9179.png',
    '9216.png',
    '9226.png',
    '9234.png',
    '9241.png',
    '9246.png',
    '9262.png',
    '9268.png',
    '9270.png',
    '9297.png',
    '9306.png',
    '9312.png',
    '9314.png',
    '9324.png',
    '9336.png',
    '9337.png',
    '9341.png',
    '9351.png',
    '9360.png',
    '9375.png',
    '9378.png',
    '9390.png',
    '9410.png',
    '9424.png',
    '9430.png',
    '9433.png',
    '9436.png',
    '9448.png',
    '9458.png',
    '9472.png',
    '9476.png',
    '9513.png',
    '9523.png',
    '9525.png',
    '9528.png',
    '9529.png',
    '9535.png',
    '9545.png',
    '9556.png',
    '9568.png',
    '9569.png',
    '9570.png',
    '9582.png',
    '9585.png',
    '9592.png',
    '9594.png',
    '9597.png',
    '9598.png',
    '9606.png',
    '9632.png',
    '9647.png',
    '9650.png',
    '9680.png',
    '9689.png',
    '9690.png',
    '9698.png',
    '9700.png',
    '9706.png',
    '9714.png',
    '9723.png',
    '9739.png',
    '9743.png',
    '9753.png',
    '9761.png',
    '9776.png',
    '9791.png',
    '9805.png',
    '9806.png',
    '9810.png',
    '9814.png',
    '9823.png',
    '9829.png',
    '9831.png',
    '9832.png',
    '9833.png',
    '9843.png',
    '9885.png',
    '9887.png',
    '9889.png',
    '9900.png',
    '9901.png',
    '9917.png',
    '9919.png',
    '9927.png',
    '9936.png',
    '9949.png',
    '9956.png',
    '9962.png',
    '9969.png',
    '9981.png',
    '9983.png',
    '9988.png',
    '9992.png',
    '9998.png',
    '9999.png',
]


# In[65]:


len(with_sunglasses)


# In[64]:


modified_testing_list


# In[69]:


testing_list_no_sunglasses = [modified_testing_list[i] for i in range(len(modified_testing_list)) if testing_images[i] not in with_sunglasses]


# In[70]:


testing_list_no_sunglasses


# In[81]:


testing_labels


# In[77]:


testing_labels_no_sunglasses = [testing_labels[i] for i in range(len(testing_labels)) if testing_images[i] not in with_sunglasses]


# In[79]:


len(testing_labels_no_sunglasses)


# In[82]:


testing_labels_no_sunglasses


# In[83]:


testing_labels_no_sunglasses = np.array(testing_labels_no_sunglasses)


# In[84]:


testing_labels_no_sunglasses


# In[71]:


len(testing_list_no_sunglasses)


# In[72]:


len(modified_testing_list)


# In[73]:


testing_set_no_sunglasses = np.array(testing_list_no_sunglasses)


# In[75]:


testing_set_no_sunglasses


# In[85]:


predicted_labels_no_sunglasses = bagmodel.predict(testing_set_no_sunglasses)


# In[86]:


from sklearn import metrics
score=metrics.accuracy_score(testing_labels_no_sunglasses, predicted_labels_no_sunglasses)
print(score)


# In[ ]:




