# SOLUTION BY A KAGGLE CONTRIBUTOR Shanan93
# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T09:38:09.021817Z","iopub.execute_input":"2021-07-13T09:38:09.022190Z","iopub.status.idle":"2021-07-13T09:38:09.032075Z","shell.execute_reply.started":"2021-07-13T09:38:09.022158Z","shell.execute_reply":"2021-07-13T09:38:09.030911Z"}}
#
# Import the necessary packages

import pandas as pd
import numpy as np
import os
import PIL
import seaborn as sns
import pickle
from PIL import *
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from IPython.display import display
from tensorflow.python.keras import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T09:38:10.480753Z","iopub.execute_input":"2021-07-13T09:38:10.481135Z","iopub.status.idle":"2021-07-13T09:38:14.951294Z","shell.execute_reply.started":"2021-07-13T09:38:10.481083Z","shell.execute_reply":"2021-07-13T09:38:14.950312Z"}}
raw = pd.read_csv('../input/facial-keypoints-detection/training.zip', compression='zip', header=0, sep=',', quotechar='"')
test_data = pd.read_csv('../input/facial-keypoints-detection/test.zip', compression='zip', header=0, sep=',', quotechar='"')
IdLookupTable = pd.read_csv('../input/facial-keypoints-detection/IdLookupTable.csv',header=0, sep=',', quotechar='"')
SampleSubmission = pd.read_csv('../input/facial-keypoints-detection/SampleSubmission.csv',header=0, sep=',', quotechar='"')

# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T08:41:06.241599Z","iopub.execute_input":"2021-07-13T08:41:06.241959Z","iopub.status.idle":"2021-07-13T08:41:06.289488Z","shell.execute_reply.started":"2021-07-13T08:41:06.241928Z","shell.execute_reply":"2021-07-13T08:41:06.288294Z"}}
raw.head()

# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T08:41:50.373905Z","iopub.execute_input":"2021-07-13T08:41:50.374350Z","iopub.status.idle":"2021-07-13T08:41:50.474145Z","shell.execute_reply.started":"2021-07-13T08:41:50.374311Z","shell.execute_reply":"2021-07-13T08:41:50.473123Z"}}
raw.describe()

# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T08:44:42.027853Z","iopub.execute_input":"2021-07-13T08:44:42.028238Z","iopub.status.idle":"2021-07-13T08:44:42.041077Z","shell.execute_reply.started":"2021-07-13T08:44:42.028199Z","shell.execute_reply":"2021-07-13T08:44:42.040144Z"}}
#Every null value is filled according to the value of the previous entry. this is in order to avoid losing data
raw.fillna(method = "ffill", inplace = True)
raw.isnull().any().value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T08:44:44.837924Z","iopub.execute_input":"2021-07-13T08:44:44.838472Z","iopub.status.idle":"2021-07-13T08:44:44.931385Z","shell.execute_reply.started":"2021-07-13T08:44:44.838422Z","shell.execute_reply":"2021-07-13T08:44:44.930394Z"}}
raw.describe()

# %% [raw]
# 

# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T08:51:58.211786Z","iopub.execute_input":"2021-07-13T08:51:58.212126Z","iopub.status.idle":"2021-07-13T08:52:09.512059Z","shell.execute_reply.started":"2021-07-13T08:51:58.212082Z","shell.execute_reply":"2021-07-13T08:52:09.511096Z"}}
#A list of images is created. iterating through the image column of the dataset which will become the input data. 
#Each image is turned into a 1D list with 0 filled in for blank values
images = []
for i in range (0 , 7049):
    img = raw["Image"][i].split(" ")
    img = ['0' if x == " " else x for x in img]
    images.append(img)




# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T08:54:32.688358Z","iopub.execute_input":"2021-07-13T08:54:32.688709Z","iopub.status.idle":"2021-07-13T08:54:53.582142Z","shell.execute_reply.started":"2021-07-13T08:54:32.688680Z","shell.execute_reply":"2021-07-13T08:54:53.581060Z"}}
images = np.array(images, dtype = 'float')
images.shape

# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T08:56:47.567630Z","iopub.execute_input":"2021-07-13T08:56:47.568004Z","iopub.status.idle":"2021-07-13T08:56:47.574410Z","shell.execute_reply.started":"2021-07-13T08:56:47.567976Z","shell.execute_reply":"2021-07-13T08:56:47.573574Z"}}
images

# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T09:04:21.377262Z","iopub.execute_input":"2021-07-13T09:04:21.377624Z","iopub.status.idle":"2021-07-13T09:04:21.517051Z","shell.execute_reply.started":"2021-07-13T09:04:21.377593Z","shell.execute_reply":"2021-07-13T09:04:21.516220Z"}}
#A sample image is displayed below with a monochrome color map
xtrain = images.reshape(-1, 96, 96)
plt.imshow(xtrain[5].reshape(96, 96), cmap = "gray")
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T09:06:34.717959Z","iopub.execute_input":"2021-07-13T09:06:34.718351Z","iopub.status.idle":"2021-07-13T09:06:34.823983Z","shell.execute_reply.started":"2021-07-13T09:06:34.718319Z","shell.execute_reply":"2021-07-13T09:06:34.822668Z"}}
#By using broadcasting we can normalize each image by understanding that 
#the maximum value for any pixel is 355
for row in images:
    row /= 255
images

# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T09:11:01.310010Z","iopub.execute_input":"2021-07-13T09:11:01.310501Z","iopub.status.idle":"2021-07-13T09:11:01.448762Z","shell.execute_reply.started":"2021-07-13T09:11:01.310471Z","shell.execute_reply":"2021-07-13T09:11:01.448084Z"}}
#Now drop the image axis from the training data as we will now prepare the yvalues for the data
plt.imshow(images[4].reshape(96,96),cmap='gray')
plt.show()

# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T09:13:51.780307Z","iopub.execute_input":"2021-07-13T09:13:51.780633Z","iopub.status.idle":"2021-07-13T09:13:51.812609Z","shell.execute_reply.started":"2021-07-13T09:13:51.780605Z","shell.execute_reply":"2021-07-13T09:13:51.811631Z"}}
#We now drop the image axis from the training data as we will now prepare the yvalues for the data
training = raw.drop(["Image"], axis = 1)
training.head()

# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T09:28:14.610404Z","iopub.execute_input":"2021-07-13T09:28:14.610895Z","iopub.status.idle":"2021-07-13T09:28:15.647022Z","shell.execute_reply.started":"2021-07-13T09:28:14.610865Z","shell.execute_reply":"2021-07-13T09:28:15.646301Z"}}
#The rest of the data less the images is the ydata to evaluate on

ytrain = []

for i in range (0, 7049):
    y = training.iloc[i, :]
    ytrain.append(y)
ytrain = np.array(ytrain, dtype = 'float')    


# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T09:28:17.484344Z","iopub.execute_input":"2021-07-13T09:28:17.484843Z","iopub.status.idle":"2021-07-13T09:28:17.644282Z","shell.execute_reply.started":"2021-07-13T09:28:17.484807Z","shell.execute_reply":"2021-07-13T09:28:17.643293Z"}}
#This is a convolutional network model. some of the code and logic has been taken 
#from online tutorials for learning purposes

model = Sequential()
# model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(96, 96, 1)))
# model.add(Convolution2D(32, 3, 3, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Convolution2D(32, 3, 3, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(30, activation='softmax'))
model.add(Convolution2D(32, (3, 3) ,activation='relu', input_shape=(96, 96, 1)))
model.add(Convolution2D(32, (3, 3) , padding="same",activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, (3, 3) , padding="same",activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(128, (3, 3) , padding="same",activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(30, activation='relu'))
model.summary()


# %% [markdown]
# 

# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T09:28:22.875390Z","iopub.execute_input":"2021-07-13T09:28:22.875734Z","iopub.status.idle":"2021-07-13T09:28:22.888223Z","shell.execute_reply.started":"2021-07-13T09:28:22.875703Z","shell.execute_reply":"2021-07-13T09:28:22.887352Z"}}
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae', 'accuracy'])

# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T09:28:24.344612Z","iopub.execute_input":"2021-07-13T09:28:24.344977Z","iopub.status.idle":"2021-07-13T09:28:24.352254Z","shell.execute_reply.started":"2021-07-13T09:28:24.344945Z","shell.execute_reply":"2021-07-13T09:28:24.351142Z"}}
images = images.reshape(-1, 96, 96, 1)
images.shape

# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T09:28:25.174577Z","iopub.execute_input":"2021-07-13T09:28:25.174931Z","iopub.status.idle":"2021-07-13T09:28:25.178681Z","shell.execute_reply.started":"2021-07-13T09:28:25.174901Z","shell.execute_reply":"2021-07-13T09:28:25.177710Z"}}
#now the model is trained on the images and ytrain data with a 0.2 validation split

# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T09:28:26.041180Z","iopub.execute_input":"2021-07-13T09:28:26.041584Z","iopub.status.idle":"2021-07-13T09:35:03.889829Z","shell.execute_reply.started":"2021-07-13T09:28:26.041540Z","shell.execute_reply":"2021-07-13T09:35:03.888853Z"}}
#PERFORM IMAGE VISUALIZATION
history = model.fit(images, ytrain, epochs=3, batch_size=255, validation_split=0.2)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T09:35:40.561814Z","iopub.execute_input":"2021-07-13T09:35:40.562220Z","iopub.status.idle":"2021-07-13T09:35:40.649247Z","shell.execute_reply.started":"2021-07-13T09:35:40.562173Z","shell.execute_reply":"2021-07-13T09:35:40.648456Z"}}
model.save("facial_keypoints_model.h5")

# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T09:38:23.005618Z","iopub.execute_input":"2021-07-13T09:38:23.006006Z","iopub.status.idle":"2021-07-13T09:38:23.017395Z","shell.execute_reply.started":"2021-07-13T09:38:23.005971Z","shell.execute_reply":"2021-07-13T09:38:23.016351Z"}}
#now the test data is prepared in a similar fashion
test_data.head()

# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T09:39:23.098245Z","iopub.execute_input":"2021-07-13T09:39:23.098601Z","iopub.status.idle":"2021-07-13T09:39:23.103185Z","shell.execute_reply.started":"2021-07-13T09:39:23.098569Z","shell.execute_reply":"2021-07-13T09:39:23.102066Z"}}
testy = test_data

# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T09:40:38.960680Z","iopub.execute_input":"2021-07-13T09:40:38.961229Z","iopub.status.idle":"2021-07-13T09:40:38.967556Z","shell.execute_reply.started":"2021-07-13T09:40:38.961195Z","shell.execute_reply":"2021-07-13T09:40:38.966556Z"}}
testdata = testy["Image"]
testdata.shape
    

# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T09:42:54.840684Z","iopub.execute_input":"2021-07-13T09:42:54.841298Z","iopub.status.idle":"2021-07-13T09:43:03.852226Z","shell.execute_reply.started":"2021-07-13T09:42:54.841263Z","shell.execute_reply":"2021-07-13T09:43:03.850770Z"}}
testimages = [] 
for i in range (0, 1783):
    img = raw["Image"][i].split(" ")
    img = ['0' if x == " " else x for x in img]
    testimages.append(img)
testimages = np.array(testimages, dtype = 'float')

# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T09:43:21.831358Z","iopub.execute_input":"2021-07-13T09:43:21.831716Z","iopub.status.idle":"2021-07-13T09:43:21.838604Z","shell.execute_reply.started":"2021-07-13T09:43:21.831686Z","shell.execute_reply":"2021-07-13T09:43:21.837360Z"}}
testimages.shape

# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T09:44:07.289602Z","iopub.execute_input":"2021-07-13T09:44:07.289982Z","iopub.status.idle":"2021-07-13T09:44:07.318274Z","shell.execute_reply.started":"2021-07-13T09:44:07.289948Z","shell.execute_reply":"2021-07-13T09:44:07.317230Z"}}
testimages /= 255

# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T09:44:59.913702Z","iopub.execute_input":"2021-07-13T09:44:59.914095Z","iopub.status.idle":"2021-07-13T09:44:59.921220Z","shell.execute_reply.started":"2021-07-13T09:44:59.914061Z","shell.execute_reply":"2021-07-13T09:44:59.919979Z"}}
testimages = testimages.reshape(-1, 96, 96, 1)
testimages.shape

# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T09:46:09.665714Z","iopub.execute_input":"2021-07-13T09:46:09.666070Z","iopub.status.idle":"2021-07-13T09:46:19.679738Z","shell.execute_reply.started":"2021-07-13T09:46:09.666040Z","shell.execute_reply":"2021-07-13T09:46:19.678734Z"}}
result = model.predict(testimages, verbose = 1)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T09:50:01.570553Z","iopub.execute_input":"2021-07-13T09:50:01.570960Z","iopub.status.idle":"2021-07-13T09:50:01.779679Z","shell.execute_reply.started":"2021-07-13T09:50:01.570925Z","shell.execute_reply":"2021-07-13T09:50:01.777540Z"}}
for imgind in range (200):
    imgind = 5
    img = xtrain[imgind]
    fig, ax = plt.subplots()
    x = range(300)
    ax.imshow(img)
    ax.scatter(history[imgind][0], history[imgind][1], color = 'firebrick')
    
    ax.scatter(ytrain["nose_tip_y"][imgind], ytrain["nose_tip_x"][imgind], color = 'blue')

# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T09:51:14.088956Z","iopub.execute_input":"2021-07-13T09:51:14.089363Z","iopub.status.idle":"2021-07-13T09:51:14.149868Z","shell.execute_reply.started":"2021-07-13T09:51:14.089327Z","shell.execute_reply":"2021-07-13T09:51:14.148937Z"}}
#The following code was used to create a properly formatted submission file for 
#kaggle using a loockup table.
lookid_list = list(IdLookupTable['FeatureName'])
imageID = list(IdLookupTable['ImageId']-1)
pre_list = list(result)


rowid = IdLookupTable['RowId']
rowid=list(rowid)



feature = []
for f in list(IdLookupTable['FeatureName']):
    feature.append(lookid_list.index(f))

# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T09:51:20.201988Z","iopub.execute_input":"2021-07-13T09:51:20.202415Z","iopub.status.idle":"2021-07-13T09:51:20.245788Z","shell.execute_reply.started":"2021-07-13T09:51:20.202380Z","shell.execute_reply":"2021-07-13T09:51:20.244784Z"}}
preded = []
for x,y in zip(imageID,feature):
    preded.append(pre_list[x][y])


rowid = pd.Series(rowid,name = 'RowId')

loc = pd.Series(preded,name = 'Location')
submission = pd.concat([rowid,loc],axis = 1)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T09:51:29.340754Z","iopub.execute_input":"2021-07-13T09:51:29.341138Z","iopub.status.idle":"2021-07-13T09:51:29.350935Z","shell.execute_reply.started":"2021-07-13T09:51:29.341086Z","shell.execute_reply":"2021-07-13T09:51:29.349927Z"}}
SampleSubmission.head()

# %% [code] {"execution":{"iopub.status.busy":"2021-07-13T09:52:08.380330Z","iopub.execute_input":"2021-07-13T09:52:08.380726Z","iopub.status.idle":"2021-07-13T09:52:08.436045Z","shell.execute_reply.started":"2021-07-13T09:52:08.380696Z","shell.execute_reply":"2021-07-13T09:52:08.435223Z"}}
SampleSubmission.to_csv('face_key_detection_submission.csv',index = False)

# %% [code]
