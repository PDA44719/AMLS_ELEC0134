#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('store', '-r')


# In[2]:


y_fs


# In[3]:


y = 2*y_fs -1


# In[98]:


y


# In[5]:


X.shape


# In[100]:


from math import atan


# In[101]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np


# In[102]:


import tensorflow as tf
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random


# In[103]:


import cv2
import os
import dlib
import re


# In[104]:


basedir = './dataset_AMLS_22-23_test/cartoon_set_test'
images_dir = os.path.join(basedir,'img')
labels_filename = 'labels.csv'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# In[105]:


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


# In[106]:


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


# In[107]:


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


# In[108]:


# This function was added by myself
def generate_dict_from_df(df, keys_col, values_col):
    zip_ = zip(df[keys_col], df[values_col])
    l = list(zip_)
    d = dict(l)
    return d


# In[109]:


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


# In[15]:


cartoon_df = pd.read_csv('dataset_AMLS_22-23/cartoon_set/labels.csv', sep='\t')


# In[16]:


cartoon_df


# In[17]:


cartoon_df.drop(cartoon_df.columns[0], axis=1, inplace=True)


# In[18]:


cartoon_df


# In[19]:


cartoon_test = pd.read_csv('dataset_AMLS_22-23_test/cartoon_set_test/labels.csv', sep='\t')
cartoon_test.drop(cartoon_test.columns[0], axis=1, inplace=True)
cartoon_test


# In[20]:


from PIL import Image


# In[21]:


image = Image.open('./dataset_AMLS_22-23/cartoon_set/img/1003.png')


# In[22]:


plt.imshow(image)


# In[23]:


X[2][8]


# In[24]:


image


# In[25]:


image_array = np.array(image)


# In[26]:


image_array.shape


# In[27]:


image_grayscale = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY )


# In[28]:


image_array[249, 389]


# In[29]:


plt.plot(249, 224, marker='o', color="red")
plt.imshow(image_grayscale, cmap='gray')
plt.show()


# In[30]:


image_grayscale[49, 389]


# In[31]:


image_grayscale[249, 224]


# In[32]:


hairline = 224
dot_color = image_grayscale[249, 224]
for i in range(223, -1, -1):
    print(image_grayscale[249, i])
    print('LOL')
    print(image_grayscale[249, i-1])
    if image_grayscale[249, i] not in range(image_grayscale[249, i-1]-10, image_grayscale[249, i-1]+10):
        hairline = i
        break


# In[33]:


dot_color


# In[34]:


hairline


# In[35]:


plt.plot(249, hairline, marker='o', color="red")
plt.imshow(image_grayscale, cmap='gray')
plt.show()


# In[36]:


image_grayscale[249, 165]


# In[37]:


image_files


# In[ ]:





# In[38]:


first_point = X[2][0]
first_point


# In[39]:


second_point = X[2][16]
second_point


# In[40]:


distance2 = np.linalg.norm(first_point - second_point)


# In[41]:


distance2


# In[42]:


distance = X[2][8][1] - X[2][19][1]


# In[43]:


distance


# In[44]:


distance/distance2


# In[45]:


leng = np.linalg.norm(firstPoint - secPoint)


# In[21]:


feature_1 = [(element[8][1]-element[19][1])/np.linalg.norm(element[0] - element[16]) for element in X]


# In[22]:


feature_1


# In[23]:


feature_2 = [(np.linalg.norm(element[4] - element[12]))/(np.linalg.norm(element[0] - element[16])) for element in X]


# In[24]:


feature_2


# In[25]:


feature_3 = [(np.linalg.norm(element[8] - element[57]))/(np.linalg.norm(element[4] - element[12])) for element in X]


# In[26]:


feature_3


# In[27]:


images_features = np.ones([8194, 3])


# In[28]:


images_features


# In[29]:


images_features[0] = [1, 2, 3] 


# In[30]:


images_features


# In[31]:


for i in range(len(feature_1)):
    images_features[i] = (feature_1[i], feature_2[i], feature_3[i])


# In[32]:


images_features.shape


# In[33]:


training_features


# In[34]:


training_labels


# In[85]:


y


# In[35]:


training_features = images_features[:7374]; testing_features = images_features[7374:];
training_labels = y[:7374]; testing_labels = y[7374:];


# In[36]:


X[0][8]


# In[37]:


feature_4 = [atan(abs(element[0][0] - element[8][0])/abs(element[0][1] - element[8][1])) for element in X]


# In[38]:


feature_4


# In[39]:


feature_5 = [atan(abs(element[1][0] - element[8][0])/abs(element[1][1] - element[8][1])) for element in X]


# In[40]:


feature_6 = [atan(abs(element[2][0] - element[8][0])/abs(element[2][1] - element[8][1])) for element in X]


# In[41]:


feature_7 = [atan(abs(element[3][0] - element[8][0])/abs(element[3][1] - element[8][1])) for element in X]


# In[42]:


feature_8 = [atan(abs(element[4][0] - element[8][0])/abs(element[4][1] - element[8][1])) for element in X]


# In[43]:


feature_9 = [atan(abs(element[5][0] - element[8][0])/abs(element[5][1] - element[8][1])) for element in X]


# In[44]:


feature_10 = [atan(abs(element[6][0] - element[8][0])/abs(element[6][1] - element[8][1])) for element in X]


# In[45]:


feature_11 = [atan(abs(element[7][0] - element[8][0])/abs(element[7][1] - element[8][1])) for element in X]


# In[46]:


feature_12 = [atan(abs(element[9][0] - element[8][0])/abs(element[9][1] - element[8][1])) for element in X]


# In[47]:


feature_13 = [atan(abs(element[10][0] - element[8][0])/abs(element[10][1] - element[8][1])) for element in X]


# In[48]:


feature_14 = [atan(abs(element[11][0] - element[8][0])/abs(element[11][1] - element[8][1])) for element in X]


# In[49]:


feature_15 = [atan(abs(element[12][0] - element[8][0])/abs(element[12][1] - element[8][1])) for element in X]


# In[50]:


feature_16 = [atan(abs(element[13][0] - element[8][0])/abs(element[13][1] - element[8][1])) for element in X]


# In[51]:


feature_17 = [atan(abs(element[14][0] - element[8][0])/abs(element[14][1] - element[8][1])) for element in X]


# In[52]:


feature_18 = [atan(abs(element[15][0] - element[8][0])/abs(element[15][1] - element[8][1])) for element in X]


# In[53]:


feature_19 = [atan(abs(element[16][0] - element[8][0])/abs(element[16][1] - element[8][1])) for element in X]


# In[54]:


images_features = np.ones([8194, 19])


# In[55]:


for i in range(len(feature_1)):
    images_features[i] = (feature_1[i], feature_2[i], feature_3[i], feature_4[i], feature_5[i], feature_6[i],feature_7[i], feature_8[i], feature_9[i],feature_10[i], feature_11[i], feature_12[i],feature_13[i], feature_14[i], feature_15[i],feature_16[i], feature_17[i], feature_18[i],feature_19[i])


# In[56]:


images_features


# In[109]:


training_features.shape


# In[111]:


def get_data():
    X, y_fs, image_files = extract_features_labels()
    #Yfs = np.array([y_gender, -(y_gender - 1)]).T
    ##Ys = np.array([y_smiling, -(y_smiling - 1)]).T
    #tr_X = X[:3836] ; tr_Yfs = Yfs[:3836]; #tr_Ys = Ys[:3836];
    #te_X = X[3836:] ; te_Yfs = Yfs[3836:]; #te_Ys = Ys[3836:]
    #te_images = image_files[3836:]
    return X, y_fs, image_files


# In[112]:


X_test, y_fs_test, image_files_test = get_data()


# In[114]:


feature_1_test = [(element[8][1]-element[19][1])/np.linalg.norm(element[0] - element[16]) for element in X_test]


# In[115]:


feature_1


# In[116]:


feature_2_test = [(np.linalg.norm(element[4] - element[12]))/(np.linalg.norm(element[0] - element[16])) for element in X_test]


# In[117]:


feature_2


# In[118]:


feature_3_test = [(np.linalg.norm(element[8] - element[57]))/(np.linalg.norm(element[4] - element[12])) for element in X_test]


# In[119]:


feature_3


# In[120]:


feature_4_test = [atan(abs(element[0][0] - element[8][0])/abs(element[0][1] - element[8][1])) for element in X_test]


# In[38]:


feature_4


# In[122]:


feature_5_test = [atan(abs(element[1][0] - element[8][0])/abs(element[1][1] - element[8][1])) for element in X_test]


# In[123]:


feature_6_test = [atan(abs(element[2][0] - element[8][0])/abs(element[2][1] - element[8][1])) for element in X_test]


# In[124]:


feature_7_test = [atan(abs(element[3][0] - element[8][0])/abs(element[3][1] - element[8][1])) for element in X_test]


# In[125]:


feature_8_test = [atan(abs(element[4][0] - element[8][0])/abs(element[4][1] - element[8][1])) for element in X_test]


# In[126]:


feature_9_test = [atan(abs(element[5][0] - element[8][0])/abs(element[5][1] - element[8][1])) for element in X_test]


# In[127]:


feature_10_test = [atan(abs(element[6][0] - element[8][0])/abs(element[6][1] - element[8][1])) for element in X_test]


# In[128]:


feature_11_test = [atan(abs(element[7][0] - element[8][0])/abs(element[7][1] - element[8][1])) for element in X_test]


# In[129]:


feature_12_test = [atan(abs(element[9][0] - element[8][0])/abs(element[9][1] - element[8][1])) for element in X_test]


# In[130]:


feature_13_test = [atan(abs(element[10][0] - element[8][0])/abs(element[10][1] - element[8][1])) for element in X_test]


# In[131]:


feature_14_test = [atan(abs(element[11][0] - element[8][0])/abs(element[11][1] - element[8][1])) for element in X_test]


# In[132]:


feature_15_test = [atan(abs(element[12][0] - element[8][0])/abs(element[12][1] - element[8][1])) for element in X_test]


# In[133]:


feature_16_test = [atan(abs(element[13][0] - element[8][0])/abs(element[13][1] - element[8][1])) for element in X_test]


# In[134]:


feature_17_test = [atan(abs(element[14][0] - element[8][0])/abs(element[14][1] - element[8][1])) for element in X_test]


# In[135]:


feature_18_test = [atan(abs(element[15][0] - element[8][0])/abs(element[15][1] - element[8][1])) for element in X_test]


# In[136]:


feature_19_test = [atan(abs(element[16][0] - element[8][0])/abs(element[16][1] - element[8][1])) for element in X_test]


# In[138]:


X_test.shape


# In[140]:


X.shape


# In[141]:


images_features_test = np.ones([2041, 19])


# In[142]:


for i in range(len(feature_1_test)):
    images_features_test[i] = (feature_1_test[i], feature_2_test[i], feature_3_test[i], feature_4_test[i], feature_5_test[i], feature_6_test[i],feature_7_test[i], feature_8_test[i], feature_9_test[i],feature_10_test[i], feature_11_test[i], feature_12_test[i],feature_13_test[i], feature_14_test[i], feature_15_test[i],feature_16_test[i], feature_17_test[i], feature_18_test[i],feature_19_test[i])


# In[143]:


y_test = 2*y_fs_test - 1


# In[144]:


testing_labels = np.array(y_test)


# # Create the model

# In[268]:


from sklearn.preprocessing import StandardScaler  # Maybe Scaling is not Necessary
sc = StandardScaler()
training_features_scaled = sc.fit_transform(training_features)


# In[61]:


from scipy import stats


# In[62]:


training_features_normalized = stats.zscore(images_features, axis=1)


# In[63]:


training_features_normalized


# In[64]:


training_features_normalized.shape


# In[65]:


from sklearn.decomposition import PCA


# In[66]:


pca = PCA(n_components=19)


# In[67]:


pca.fit(training_features_normalized)


# In[68]:


s_pca=pca.transform(training_features_normalized)


# In[69]:


features = s_pca[:, :4]
features


# In[70]:


y


# In[71]:


y.shape


# In[72]:


y_int = np.ndarray(8194,)


# In[73]:


for i in range(len(y_int)):
    y_int[i] = int(y[i])


# In[74]:


y_int = y.astype(int)
y_int


# In[75]:


type(y_int[0])


# In[76]:


colors[y_int]


# In[77]:


n = 4
fig, ax = plt.subplots(n, n, figsize=(20, 20))
colors = ['red', 'green', 'blue', 'black', 'brown']

for i in range(n):
    for j in range(n):
        ax[j][i].scatter(x=features[:, i], y=features[:, j], c=y_int, s=1, cmap='CMRmap')
        ax[j][i].set_xlabel(f'PC{i+1}')
        ax[j][i].set_ylabel(f'PC{j+1}')
plt.show()


# In[96]:


fig = plt.figure(figsize=(5, 5))

plt.scatter(x=features[:, 2], y=features[:, 1], c=y, s=1, cmap='CMRmap')
plt.xlabel(f'PC 3', fontsize=20)
plt.ylabel(f'PC 4', fontsize=20)

plt.xticks(fontsize=16, rotation=-45)
plt.yticks(fontsize=16)
    


# In[120]:


testing_features_normalized = stats.zscore(testing_features, axis=1)


# In[115]:


classifier = tf.keras.models.Sequential()
classifier.add(tf.keras.layers.Dense(units = 100, activation = 'relu', input_shape = (19,)))
classifier.add(tf.keras.layers.Dropout(0.2)) # Drop some neurons, along with their weights, to avoid overfitting the data

classifier.add(tf.keras.layers.Dense(units = 50, activation = 'relu'))
classifier.add(tf.keras.layers.Dropout(0.2))

classifier.add(tf.keras.layers.Dense(units = 5, activation = 'softmax'))


# In[116]:


classifier.compile(optimizer = 'Adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


# In[117]:


epochs_hist = classifier.fit(training_features_normalized, training_labels, epochs = 100)


# In[118]:


testing_features_scaled = sc.transform(testing_features)


# In[119]:


evaluation = classifier.evaluate(stats.zscore(testing_features, axis=1), testing_labels)


# # Try an SVM

# In[145]:


from sklearn.svm import SVC


# In[146]:


from sklearn.metrics import classification_report,accuracy_score


# In[148]:


def SVM(x_train,y_train, x_test):
    model = SVC(kernel = 'rbf', probability=True, C=6, gamma=0.6)
    model.fit(x_train, y_train)  #fit using x_train and y_train
    y_pred = model.predict(x_test)
    return y_pred
# Scikit learn library results
y_pred=SVM(images_features, y, images_features_test)
print(accuracy_score(testing_labels,y_pred))


# In[150]:


def SVM2(x_train,y_train, x_test):
    model = SVC(kernel = 'rbf', probability=True, C=6, gamma=0.6)
    model.fit(stats.zscore(x_train, axis=1), y_train)  #fit using x_train and y_train
    y_pred = model.predict(stats.zscore(x_test, axis=1))
    return y_pred
# Scikit learn library results
y_pred=SVM(images_features, y, images_features_test)
print(accuracy_score(testing_labels,y_pred))


# In[149]:


images_features_test


# In[108]:


from tensorflow.keras import applications


# In[111]:


base_model = applications.InceptionV3(weights='imagenet',
                                      include_top=False, 
                                      input_shape=(7374, 19, 3))


# In[151]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(testing_labels, y_pred)
plt.figure(figsize = (20, 10))
sns.heatmap(cm, annot = True)


# In[158]:


from sklearn.metrics import confusion_matrix
sns.set(font_scale=1.5) # Adjust to fit
cm = confusion_matrix(testing_labels, y_pred)
plt.figure(figsize = (10, 8))
plt.xlabel('True Labels', fontsize=18)
ax = sns.heatmap(cm, annot = True)
ax.set_xlabel('Predicted Labels', fontsize=18)
ax.set_ylabel('True Labels', fontsize=18)
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(18)


# In[157]:


print(classification_report(testing_labels, y_pred))


# In[152]:


face3 = cartoon_df[cartoon_df['face_shape'] == 3]


# In[153]:


face3


# In[154]:


face4 = cartoon_df[cartoon_df['face_shape'] == 4]


# In[155]:


face4


# # Let's try transfer learning now

# In[33]:


from PIL import Image


# In[259]:


image = Image.open('dataset_AMLS_22-23/cartoon_set/img/1.png')
image_array = np.array(image)
resized_image = resize_image(image_array)


# In[261]:


resized_image[:, :, 0]


# In[34]:


def resize_image(img_array):
    pil_image = Image.fromarray(img_array)
    resized_image = pil_image.resize((150, 150))
    return np.array(resized_image)


# In[263]:


image_array[:, :, 3]


# In[207]:


plt.imshow(image_array)
plt.show()


# In[202]:


image_array[0]


# In[156]:


image_array = image_array/255


# In[157]:


image_array


# In[144]:


plt.imshow(image_array, cmap='gray')
plt.show()


# In[166]:


type(int(cartoon_df.iloc[0].face_shape))


# In[181]:


dataset = np.ndarray((10000, 150, 150, 3))


# In[179]:


dataset = np.ndarray((10000, 150, 150, 3))
labels = []
for i in range(10000):
    print(f'Doing image {i+1} of 10000')
    image = Image.open(f'dataset_AMLS_22-23/cartoon_set/img/{i}.png')
    image=image.resize((150, 150))
    image_array = np.array(image)
    image_array = image_array[:, :, 0:3]
    dataset[i] = image_array/255
    labels.append(int(cartoon_df.iloc[i].face_shape))


# In[265]:


dataset.shape


# In[149]:


dataset = dataset/255
dataset


# In[210]:


dataset


# In[153]:


dataset_scaled = dataset/255


# In[228]:


base_model = applications.InceptionV3(weights='imagenet', 
                                include_top=False, 
                                input_shape=(150, 150, 3))
base_model.trainable = False

add_model = tf.keras.models.Sequential()
add_model.add(base_model)
add_model.add(tf.keras.layers.GlobalAveragePooling2D())
add_model.add(tf.keras.layers.Dropout(0.5))
add_model.add(tf.keras.layers.Dense(5, activation='softmax'))

model = add_model
model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='Adam',
              metrics=['accuracy'])
model.summary()


# In[221]:


labels = np.array(labels)


# In[222]:


labels


# In[223]:


training_features = dataset[:7374]; testing_features = dataset[7374:];
training_labels = labels[:7374]; testing_labels = labels[7374:];


# In[266]:


file_path="weights.best.hdf5"

#checkpoint = ModelCheckpoint(file_path, monitor='acc', verbose=1, save_best_only=True, mode='max')

#early = EarlyStopping(monitor="acc", mode="max", patience=15)

#callbacks_list = [checkpoint, early] #early

history = model.fit(training_features, training_labels, epochs=50)


# In[ ]:




