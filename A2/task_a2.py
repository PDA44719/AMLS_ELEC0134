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
import os
import dlib
import re


# In[3]:


import cv2


# # Define a few variables

# In[4]:


basedir = './dataset_AMLS_22-23/celeba'
images_dir = os.path.join(basedir,'img')
labels_filename = 'labels.csv'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# # Define a set of functions to be used to extract the features 

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


# In[10]:


images_dir = 'dataset_AMLS_22-23/celeba/img/'

data_dir = 'dataset_AMLS_22-23/celeba/'


# In[11]:


get_ipython().run_line_magic('store', '-r')


# In[12]:


X


# In[12]:


df = pd.read_csv(data_dir + 'labels.csv', sep='\t')


# In[13]:


df


# In[14]:


df.drop(df.columns[0], axis=1, inplace=True)
df


# In[15]:


all_features = []
all_smiling_labels = []
all_images_used = []


# In[16]:


type(df.iloc[2]['smiling'])


# In[17]:


from PIL import Image


# In[162]:


image = Image.open(f'dataset_AMLS_22-23/celeba/img/0.jpg')
image_array = np.array(image)
features,_ = run_dlib_shape(image_array)


# In[163]:


features[48:]


# In[18]:


for i in range(5000):
    print(f'Processing image {i+1} out of 5000')
    image = Image.open(f'dataset_AMLS_22-23/celeba/img/{i}.jpg')
    image_array = np.array(image)
    features,_ = run_dlib_shape(image_array)
    if features is not None:  # The model detected it as a face
        all_features.append(features[48:])
        all_smiling_labels.append(int(df.iloc[i].smiling))
        all_images_used.append(f'{i}.jpg')
    


# In[19]:


len(all_features[0])


# In[20]:


all_images_used


# In[167]:


dataset = np.ndarray((4795, 40))


# In[168]:


dataset


# In[169]:


for i in range(len(all_features)):
    print(f'{i} out of 4795')
    dataset[i] = all_features[i].reshape((40,))


# In[59]:


dataset


# In[70]:


all_smiling_labels = [int((element+1)/2) for element in all_smiling_labels]


# In[ ]:





# In[66]:


labels = np.array(all_smiling_labels)


# In[72]:


labels


# In[73]:


training_features = dataset[:4000]; testing_features = dataset[4000:];
training_labels = labels[:4000]; testing_labels = labels[4000:];


# In[74]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
training_features = sc.fit_transform(training_features)


# In[75]:


training_features


# In[57]:


all_features[0].reshape((40,))


# # NN first

# In[95]:


# This model was extracted from Project 6.ipynb (online course)
classifier = tf.keras.models.Sequential()
classifier.add(tf.keras.layers.Dense(units = 50, activation = 'relu', input_shape = (40,)))
classifier.add(tf.keras.layers.Dropout(0.2)) # Drop some neurons, along with their weights, to avoid overfitting the data

classifier.add(tf.keras.layers.Dense(units = 20, activation = 'relu'))
classifier.add(tf.keras.layers.Dropout(0.2))

classifier.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))


# In[96]:


classifier.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[97]:


epochs_hist = classifier.fit(training_features, training_labels, epochs = 100, validation_split=0.1)


# In[98]:


evaluation = classifier.evaluate(sc.transform(testing_features), testing_labels)


# # Let's try some other simpler models

# ## Bagging classfier

# In[99]:


from sklearn.ensemble import BaggingClassifier


# In[113]:


bagmodel=BaggingClassifier(n_estimators=100,max_samples=0.4, max_features=5,random_state=2)


# In[114]:


bagmodel.fit(training_features, training_labels)


# In[115]:


predicted_labels = bagmodel.predict(sc.transform(testing_features))


# In[116]:


from sklearn import metrics
score=metrics.accuracy_score(testing_labels, predicted_labels)
print(score)


# ## SVM

# In[117]:


from sklearn.svm import LinearSVC


# In[118]:


from sklearn.metrics import accuracy_score


# In[148]:


svm_classifier = LinearSVC(C=0.25)


# In[149]:


svm_classifier.fit(training_features, training_labels)


# In[150]:


y_pred=svm_classifier.predict(sc.transform(testing_features))
print(accuracy_score(testing_labels,y_pred))


# ## Boosting

# In[151]:


from sklearn.ensemble import AdaBoostClassifier


# In[163]:


boostmodel=AdaBoostClassifier(n_estimators=50)


# In[164]:


boostmodel.fit(training_features, training_labels, sample_weight=None) # Fit KNN model


# In[165]:


pred_labels_boost = boostmodel.predict(sc.transform(testing_features))


# In[166]:


score1=metrics.accuracy_score(testing_labels, pred_labels_boost)


# In[167]:


score1


# ## KNN

# In[168]:


from sklearn.neighbors import KNeighborsClassifier


# In[171]:


def KNNClassifier(X_train, y_train, X_test,k):

    #Create KNN object with a K coefficient
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train) # Fit KNN model


    Y_pred = neigh.predict(X_test)
    return Y_pred

Y_pred=KNNClassifier(training_features, training_labels, sc.transform(testing_features), 63)

score=metrics.accuracy_score(testing_labels,Y_pred)
print(score)


# ## Decision-Tree

# We will import the DecisionTreeClassifier function from the sklearn library. Recall that we need to specify the attribute selection measures for Decision-Tree algorithm. In this experiment, we will set this '**criterion**' to 'entropy', i.e., information gain.

# In[172]:


from sklearn import tree


# In[173]:


#Importing the Decision tree classifier from the sklearn library.
tree_params={
    'criterion':'entropy'
}
clf = tree.DecisionTreeClassifier( **tree_params )


# Next, we will train and test the classifier on the corresponding dataset. You will be asked to complete the train and test function.

# In[174]:


#Training the decision tree classifier on training set. 
# Please complete the code below.
clf.fit(training_features, training_labels)


#Predicting labels on the test set.
# Please complete the code below.
y_pred =  clf.predict(sc.transform(testing_features))


# We will now evaluate the predicted classes using some metrics. For this case, we will use 'accuracy_score' to calculate the accuracy of the predicted labels.

# In[177]:


#Use accuracy metric from sklearn.metrics library
print('Accuracy Score on train data: ', accuracy_score(y_true=training_labels, y_pred=clf.predict(training_features)))
print('Accuracy Score on test data: ', accuracy_score(y_true=testing_labels, y_pred=y_pred))


# In[178]:


all_features


# # Let's work with the Euclidean distances

# In[156]:


def euclidean_distance(feature1, feature2):
    np.linalg.norm()


# In[170]:


all_features[0]


# In[158]:


np.linalg.norm(all_features[0][0] - all_features[0][2])


# In[190]:


all_features[0][0]


# In[21]:


distances_matrix = np.ndarray((20, 20))
for i in range(len(all_features[0])):
    for j in range(len(all_features[0])):
        distances_matrix[i][j] = np.linalg.norm(all_features[0][i] - all_features[0][j])


# In[22]:


distances_matrix


# In[24]:


D = []
for i in range(20):
    for j in range(i+1, 20):
        D.append(distances_matrix[i][j])


# In[25]:


D


# In[197]:


for p in range(20, 20):
    print(p)


# In[198]:


distances_matrix[0][2]


# In[204]:


D2 = distances_matrix[np.triu_indices(20, 1)]


# In[206]:


D2 = list(D2)
D2


# In[207]:


D == D2


# In[208]:


from sklearn import preprocessing


# In[210]:


D2_nomalized = sum(D2)


# In[211]:


D2_nomalized


# In[212]:


F = D / D2_nomalized


# In[215]:


len(F)


# # Let's first create the dataset

# In[23]:


def obtain_distances_matrix(features):
    distances_matrix = np.ndarray((20, 20))
    for i in range(len(features)):
        for j in range(len(features)):
            distances_matrix[i][j] = np.linalg.norm(features[i] - features[j])
    return distances_matrix


# In[24]:


dataset = np.ndarray((len(all_features), 190))


# In[25]:


for i in range(len(all_features)):
    print(f'Completing image {i+1} out of {len(all_features)}')
    distances_matrix = obtain_distances_matrix(all_features[i])  # Obtain upper triangle
    D = distances_matrix[np.triu_indices(20, 1)]
    D_l1 = sum(D)
    F = D / D_l1
    dataset[i] = F


# In[26]:


dataset


# In[27]:


labels = np.array(all_smiling_labels)


# In[28]:


labels = [int((element+1)/2) for element in labels]


# In[29]:


labels = np.array(labels)


# In[30]:


labels


# In[39]:


training_features = dataset[:4000]; testing_features = dataset[4000:];
training_labels = labels[:4000]; testing_labels = labels[4000:];


# In[31]:


from scipy import stats


# In[180]:


dataset_transformed = stats.zscore(dataset, axis=1)


# In[181]:


dataset_transformed


# In[32]:


test_df = pd.read_csv('dataset_AMLS_22-23_test/celeba_test/labels.csv', sep='\t')


# In[33]:


test_df.drop(test_df.columns[0], axis=1, inplace=True)
test_df


# In[81]:


test_df.iloc[18]


# # Let's now try the NN

# In[219]:


from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit


# In[34]:


def mlp(m, n):
    classifier = tf.keras.models.Sequential()
    classifier.add(tf.keras.layers.Dense(units = m, activation = 'relu', input_shape = (190,)))
    classifier.add(tf.keras.layers.Dropout(0.2)) # Drop some neurons, along with their weights, to avoid overfitting the data

    classifier.add(tf.keras.layers.Dense(units = n, activation = 'relu'))
    classifier.add(tf.keras.layers.Dropout(0.2))

    classifier.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))
    classifier.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier


# In[212]:


model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=nn)


# In[213]:


parameters = {'m':[100, 200, 400], 'C':[10, 20, 50]}
clf = GridSearchCV(nn, parameters)


# In[214]:


clf.fit(dataset, labels)


# In[234]:


classifier = tf.keras.models.Sequential()
classifier.add(tf.keras.layers.Dense(units = 400, activation = 'relu', input_shape = (190,)))
classifier.add(tf.keras.layers.Dropout(0.2)) # Drop some neurons, along with their weights, to avoid overfitting the data

classifier.add(tf.keras.layers.Dense(units = 10, activation = 'relu'))
classifier.add(tf.keras.layers.Dropout(0.2))

classifier.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))


# In[235]:


classifier.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[35]:


from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=10)


# In[36]:


best_classifier = []
best_history = []
for m in [100, 200, 400]:
    for n in [10, 20, 50]:
        print(f'Training model with parameters m={m} and n={n}')
        classifier = mlp(m, n)
        hist = classifier.fit(dataset, labels, epochs = 100, validation_split=0.2, callbacks=[early_stopping], verbose=0)
        if len(best_classifier) == 0:
            best_classifier.append(classifier)
            best_history.append(hist)
        else:
            if hist.history['val_loss'][-1] > best_history[0].history['val_loss'][-1]:
                best_classifier.pop()
                best_classifier.append(classifier)
                best_history.pop()
                best_history.append(hist)


# In[100]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(20, 7))

# summarize history for accuracy
ax[0].plot(best_history[0].history['loss'])
ax[0].plot(best_history[0].history['val_loss'])
#ax[0].set_title('odel accuracy')
ax[0].set_ylabel('Loss', fontsize=25)
ax[0].set_xlabel('Epoch', fontsize=25)
ax[0].legend(['Train', 'Validation'], loc='upper right', fontsize=25)
ax[0].grid()

for label in (ax[0].get_xticklabels() + ax[0].get_yticklabels()):
	label.set_fontsize(19)


# summarize history for loss
ax[1].plot(best_history[0].history['loss'])
ax[1].plot(best_history[0].history['val_loss'])
#ax[1].set_title('model loss')
ax[1].set_ylabel('Loss', fontsize=16)
ax[1].set_xlabel('Epoch', fontsize=16)
ax[1].legend(['Train', 'Validation'], loc='upper right', fontsize=16)
ax[1].grid()

for label in (ax[1].get_xticklabels() + ax[1].get_yticklabels()):
	label.set_fontsize(14)

    
plt.show()


# In[37]:


best_classifier[0].summary()


# In[ ]:





# In[244]:


epochs_hist.history['val_loss'][-1]


# In[237]:


epochs_hist = classifier.fit(dataset, labels, epochs = 500, validation_split=0.2, callbacks=[early_stopping])


# In[241]:


classifier2 = mlp(400, 20)
epochs_hist2 = classifier2.fit(dataset, labels, epochs = 500, validation_split=0.2, callbacks=[early_stopping])


# In[233]:


classifier2.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[ ]:


epochs_hist2 = classifier.fit(dataset, labels, epochs = 500, validation_split=0.2, callbacks=[early_stopping])


# In[105]:


evaluation = classifier.evaluate(testing_features, testing_labels)


# In[47]:


features_test = []
labels_test = []
test_images_used = []
for i in range(1000):
    print(f'Processing image {i+1} out of 5000')
    image = Image.open(f'dataset_AMLS_22-23_test/celeba_test/img/{i}.jpg')
    image_array = np.array(image)
    features,_ = run_dlib_shape(image_array)
    if features is not None:  # The model detected it as a face
        features_test.append(features[48:])
        labels_test.append(int(test_df.iloc[i].smiling))
        test_images_used.append(f'{i}.jpg')


# In[48]:


test_dataset = np.ndarray((len(features_test), 190))


# In[49]:


for i in range(len(features_test)):
    print(f'Completing image {i+1} out of {len(features_test)}')
    distances_matrix = obtain_distances_matrix(features_test[i])  # Obtain upper triangle
    D = distances_matrix[np.triu_indices(20, 1)]
    D_l1 = sum(D)
    F = D / D_l1
    test_dataset[i] = F


# In[50]:


test_labels = [int((element+1)/2) for element in labels_test]


# In[51]:


test_labels


# In[52]:


test_labels = np.array(test_labels)


# In[53]:


test_labels


# In[87]:


best_history[0].history['val_accuracy']


# In[59]:


evaluation = best_classifier[0].evaluate(test_dataset, test_labels)


# In[57]:


pred_test_labels = best_classifier[0].predict(test_dataset)

pred_test_labels = (pred_test_labels > 0.5)


# In[95]:


from sklearn.metrics import confusion_matrix
sns.set(font_scale=1.5) # Adjust to fit
cm = confusion_matrix(test_labels, pred_test_labels)
plt.figure(figsize = (6, 3))
plt.xlabel('True Labels')
ax = sns.heatmap(cm, annot = True)
ax.set_xlabel('Predicted Labels', fontsize=18)
ax.set_ylabel('True Labels', fontsize=18)
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(18)


# In[79]:


wrong_predictions = []
for i in range(len(pred_test_labels)):
    if pred_test_labels[i] == test_labels[i]:
        continue
    else:
        if test_labels[i] == 1:
            wrong_predictions.append(f'{i}.jpg')


# In[83]:


wrong_predictions


# In[ ]:


# create figure
fig = plt.figure(figsize=(15, 21))

# setting values to rows and column variables
rows = 6
columns = 6


# In[78]:


for i in range(36):
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(np.array(Image.open(f'dataset_AMLS_22-23_test/celeba_test/img/{wrong_predictions[i]}')))
    plt.axis('off')
    plt.title(f'{wrong_predictions[i]}\nReal: {test_labels[i]}\nPrediction: {1 if pred_test_labels[i] else 0}')
    


# ## Bagging classfier

# In[54]:


from sklearn.ensemble import BaggingClassifier


# In[69]:


bagmodel=BaggingClassifier(n_estimators=200,max_samples=0.6, max_features=100,random_state=2)


# In[70]:


bagmodel.fit(training_features, training_labels)


# In[71]:


predicted_labels = bagmodel.predict(testing_features)


# In[72]:


from sklearn import metrics
score=metrics.accuracy_score(testing_labels, predicted_labels)
print(score)


# ## SVM

# In[215]:


from sklearn.svm import SVC


# In[216]:


from sklearn.metrics import accuracy_score


# In[84]:


svm_classifier = SVC(C=0.8)


# In[223]:


C_range = [0.1, 0.5, 0.8]
gamma_range = [-0.5, 0.2, 0.5]
param_grid = dict(gamma=gamma_range, C=C_range)


# In[224]:


svm_classifier.fit(training_features, training_labels)


# In[225]:


cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)


# In[226]:


grid.fit(dataset, labels)


# In[227]:


print(f'The best parameters are {grid.best_params_} with a score of {grid.best_score_}')


# In[86]:


y_pred=svm_classifier.predict(testing_features)
print(accuracy_score(testing_labels,y_pred))


# ## Boosting

# In[87]:


from sklearn.ensemble import AdaBoostClassifier


# In[97]:


boostmodel=AdaBoostClassifier(n_estimators=100)


# In[98]:


boostmodel.fit(training_features, training_labels, sample_weight=None) # Fit KNN model


# In[99]:


pred_labels_boost = boostmodel.predict(testing_features)


# In[100]:


score1=metrics.accuracy_score(testing_labels, pred_labels_boost)


# In[101]:


score1


# ## KNN

# In[168]:


from sklearn.neighbors import KNeighborsClassifier


# In[171]:


def KNNClassifier(X_train, y_train, X_test,k):

    #Create KNN object with a K coefficient
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train) # Fit KNN model


    Y_pred = neigh.predict(X_test)
    return Y_pred

Y_pred=KNNClassifier(training_features, training_labels, sc.transform(testing_features), 63)

score=metrics.accuracy_score(testing_labels,Y_pred)
print(score)


# ## Decision-Tree

# We will import the DecisionTreeClassifier function from the sklearn library. Recall that we need to specify the attribute selection measures for Decision-Tree algorithm. In this experiment, we will set this '**criterion**' to 'entropy', i.e., information gain.

# In[172]:


from sklearn import tree


# In[173]:


#Importing the Decision tree classifier from the sklearn library.
tree_params={
    'criterion':'entropy'
}
clf = tree.DecisionTreeClassifier( **tree_params )


# Next, we will train and test the classifier on the corresponding dataset. You will be asked to complete the train and test function.

# In[174]:


#Training the decision tree classifier on training set. 
# Please complete the code below.
clf.fit(training_features, training_labels)


#Predicting labels on the test set.
# Please complete the code below.
y_pred =  clf.predict(sc.transform(testing_features))


# We will now evaluate the predicted classes using some metrics. For this case, we will use 'accuracy_score' to calculate the accuracy of the predicted labels.

# In[177]:


#Use accuracy metric from sklearn.metrics library
print('Accuracy Score on train data: ', accuracy_score(y_true=training_labels, y_pred=clf.predict(training_features)))
print('Accuracy Score on test data: ', accuracy_score(y_true=testing_labels, y_pred=y_pred))


# In[ ]:





# # Let's try the PCA analysis

# In[106]:


dataset


# In[78]:


from sklearn.decomposition import PCA


# In[80]:


pca = PCA(n_components=190)


# In[81]:


pca.fit(dataset)


# In[82]:


s_pca=pca.transform(dataset)


# In[111]:


s_pca


# In[112]:


cov_X = np.cov(dataset.T)


# In[113]:


cov_X


# In[114]:


eigen_vals, eigen_vecs = np.linalg.eig(cov_X)


# In[115]:


eigen_vals


# In[116]:


sum_of_eig = np.sum(eigen_vals)
sum_of_eig


# In[117]:


len(eigen_vals)


# In[131]:


def summation(variable, k):
    result = 0
    for i in range(k):
        result += variable[i]
    return result


# In[174]:


a = np.array([1, 2, 3, 4, 5, 6])


# In[175]:


a = a.reshape((2, 3))
a


# In[178]:


a[:, :2]


# In[127]:


summation(eig_that_meet_cond, 1)


# In[159]:


# Find the first eigen values that meet the condition
eig_that_meet_cond = []
for i in range(190):
    if summation(eigen_vals, i) >=  0.9999995 * sum_of_eig:
        eig_that_meet_cond.append(eigen_vals[i])


# In[160]:


len(eig_that_meet_cond)


# In[95]:


features = s_pca[:, :4]
features


# In[181]:


len(features)


# In[188]:


features[:, 1]


# In[191]:


labels


# In[97]:


n = 4
fig, ax = plt.subplots(n, n, figsize=(20, 20))

for i in range(n):
    for j in range(n):
        ax[j][i].scatter(x=features[:, i], y=features[:, j], c=labels, s=1)
        ax[j][i].set_xlabel(f'PC{i+1}')
        ax[j][i].set_ylabel(f'PC{j+1}')
plt.show()


# In[138]:


fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].scatter(x=features[:, 1], y=features[:, 2], c=labels, s=1)
ax[0].set_xlabel(f'PC 2', fontsize=20)
ax[0].set_ylabel(f'PC 3', fontsize=20)

for label in (ax[0].get_xticklabels() + ax[0].get_yticklabels()):
	label.set_fontsize(14)

ax[1].scatter(x=features[:, 2], y=features[:, 3], c=labels, s=1, label=labels)
ax[1].set_xlabel(f'PC 3', fontsize=20)
ax[1].set_ylabel(f'PC 4', fontsize=20)

for label in (ax[1].get_xticklabels() + ax[1].get_yticklabels()):
	label.set_fontsize(14)

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)


plt.show()


# In[130]:


features[:, 2].argmax()


# In[134]:


df.iloc[1126]


# In[187]:


ax[0][3]


# In[234]:


training_features = features[:4000]; testing_features = features[4000:];
training_labels = labels[:4000]; testing_labels = labels[4000:];


# # Let's now try the NN

# In[186]:


classifier = tf.keras.models.Sequential()
classifier.add(tf.keras.layers.Dense(units = 200, activation = 'relu', input_shape = (100,)))
classifier.add(tf.keras.layers.Dropout(0.2)) # Drop some neurons, along with their weights, to avoid overfitting the data

classifier.add(tf.keras.layers.Dense(units = 50, activation = 'relu'))
classifier.add(tf.keras.layers.Dropout(0.2))

classifier.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))


# In[187]:


classifier.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[188]:


epochs_hist = classifier.fit(training_features, training_labels, epochs = 100, validation_split=0.1)


# In[252]:


evaluation = classifier.evaluate(testing_features, testing_labels)


# ## Bagging classfier

# In[253]:


from sklearn.ensemble import BaggingClassifier


# In[254]:


bagmodel=BaggingClassifier(n_estimators=200,max_samples=0.6, max_features=4,random_state=2)


# In[255]:


bagmodel.fit(training_features, training_labels)


# In[256]:


predicted_labels = bagmodel.predict(testing_features)


# In[257]:


from sklearn import metrics
score=metrics.accuracy_score(testing_labels, predicted_labels)
print(score)


# ## SVM

# In[258]:


from sklearn.svm import LinearSVC


# In[259]:


from sklearn.metrics import accuracy_score


# In[260]:


svm_classifier = LinearSVC(C=0.6)


# In[261]:


svm_classifier.fit(training_features, training_labels)


# In[262]:


y_pred=svm_classifier.predict(testing_features)
print(accuracy_score(testing_labels,y_pred))


# ## Boosting

# In[263]:


from sklearn.ensemble import AdaBoostClassifier


# In[264]:


boostmodel=AdaBoostClassifier(n_estimators=100)


# In[265]:


boostmodel.fit(training_features, training_labels, sample_weight=None) # Fit KNN model


# In[266]:


pred_labels_boost = boostmodel.predict(testing_features)


# In[267]:


score1=metrics.accuracy_score(testing_labels, pred_labels_boost)


# In[268]:


score1


# ## KNN

# In[269]:


from sklearn.neighbors import KNeighborsClassifier


# In[271]:


def KNNClassifier(X_train, y_train, X_test,k):

    #Create KNN object with a K coefficient
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train) # Fit KNN model


    Y_pred = neigh.predict(X_test)
    return Y_pred

Y_pred=KNNClassifier(training_features, training_labels, testing_features, 10)

score=metrics.accuracy_score(testing_labels,Y_pred)
print(score)


# ## Decision-Tree

# We will import the DecisionTreeClassifier function from the sklearn library. Recall that we need to specify the attribute selection measures for Decision-Tree algorithm. In this experiment, we will set this '**criterion**' to 'entropy', i.e., information gain.

# In[272]:


from sklearn import tree


# In[273]:


#Importing the Decision tree classifier from the sklearn library.
tree_params={
    'criterion':'entropy'
}
clf = tree.DecisionTreeClassifier( **tree_params )


# Next, we will train and test the classifier on the corresponding dataset. You will be asked to complete the train and test function.

# In[275]:


#Training the decision tree classifier on training set. 
# Please complete the code below.
clf.fit(training_features, training_labels)


#Predicting labels on the test set.
# Please complete the code below.
y_pred =  clf.predict(testing_features)


# We will now evaluate the predicted classes using some metrics. For this case, we will use 'accuracy_score' to calculate the accuracy of the predicted labels.

# In[276]:


#Use accuracy metric from sklearn.metrics library
print('Accuracy Score on train data: ', accuracy_score(y_true=training_labels, y_pred=clf.predict(training_features)))
print('Accuracy Score on test data: ', accuracy_score(y_true=testing_labels, y_pred=y_pred))


# # Let's now try the NN

# In[102]:


classifier = tf.keras.models.Sequential()
classifier.add(tf.keras.layers.Dense(units = 400, activation = 'relu', input_shape = (190,)))
classifier.add(tf.keras.layers.Dropout(0.2)) # Drop some neurons, along with their weights, to avoid overfitting the data

classifier.add(tf.keras.layers.Dense(units = 400, activation = 'relu'))
classifier.add(tf.keras.layers.Dropout(0.2))

classifier.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))


# In[103]:


classifier.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[104]:


epochs_hist = classifier.fit(training_features, training_labels, epochs = 500, validation_split=0.1)


# In[105]:


evaluation = classifier.evaluate(testing_features, testing_labels)


# ## Bagging classfier

# In[54]:


from sklearn.ensemble import BaggingClassifier


# In[69]:


bagmodel=BaggingClassifier(n_estimators=200,max_samples=0.6, max_features=100,random_state=2)


# In[70]:


bagmodel.fit(training_features, training_labels)


# In[71]:


predicted_labels = bagmodel.predict(testing_features)


# In[72]:


from sklearn import metrics
score=metrics.accuracy_score(testing_labels, predicted_labels)
print(score)


# ## SVM

# In[73]:


from sklearn.svm import LinearSVC


# In[74]:


from sklearn.metrics import accuracy_score


# In[84]:


svm_classifier = LinearSVC(C=0.8)


# In[85]:


svm_classifier.fit(training_features, training_labels)


# In[86]:


y_pred=svm_classifier.predict(testing_features)
print(accuracy_score(testing_labels,y_pred))


# ## Boosting

# In[87]:


from sklearn.ensemble import AdaBoostClassifier


# In[97]:


boostmodel=AdaBoostClassifier(n_estimators=100)


# In[98]:


boostmodel.fit(training_features, training_labels, sample_weight=None) # Fit KNN model


# In[99]:


pred_labels_boost = boostmodel.predict(testing_features)


# In[100]:


score1=metrics.accuracy_score(testing_labels, pred_labels_boost)


# In[101]:


score1


# ## KNN

# In[168]:


from sklearn.neighbors import KNeighborsClassifier


# In[171]:


def KNNClassifier(X_train, y_train, X_test,k):

    #Create KNN object with a K coefficient
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train) # Fit KNN model


    Y_pred = neigh.predict(X_test)
    return Y_pred

Y_pred=KNNClassifier(training_features, training_labels, sc.transform(testing_features), 63)

score=metrics.accuracy_score(testing_labels,Y_pred)
print(score)


# ## Decision-Tree

# We will import the DecisionTreeClassifier function from the sklearn library. Recall that we need to specify the attribute selection measures for Decision-Tree algorithm. In this experiment, we will set this '**criterion**' to 'entropy', i.e., information gain.

# In[172]:


from sklearn import tree


# In[173]:


#Importing the Decision tree classifier from the sklearn library.
tree_params={
    'criterion':'entropy'
}
clf = tree.DecisionTreeClassifier( **tree_params )


# Next, we will train and test the classifier on the corresponding dataset. You will be asked to complete the train and test function.

# In[174]:


#Training the decision tree classifier on training set. 
# Please complete the code below.
clf.fit(training_features, training_labels)


#Predicting labels on the test set.
# Please complete the code below.
y_pred =  clf.predict(sc.transform(testing_features))


# We will now evaluate the predicted classes using some metrics. For this case, we will use 'accuracy_score' to calculate the accuracy of the predicted labels.

# In[177]:


#Use accuracy metric from sklearn.metrics library
print('Accuracy Score on train data: ', accuracy_score(y_true=training_labels, y_pred=clf.predict(training_features)))
print('Accuracy Score on test data: ', accuracy_score(y_true=testing_labels, y_pred=y_pred))


# In[ ]:




