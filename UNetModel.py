__author__ = "Klas Holmgren"
__license__ = "Feel free to copy"

#Imports
import numpy as np
#Set the `numpy` pseudo-random generator at a fixed value
#This helps with repeatable results everytime you run the code.
np.random.seed(42)

import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
import keras
from skimage.transform import resize


os.environ['KERAS_BACKEND'] = 'tensorflow' # Added to set the backend as Tensorflow


def loadZoomLevles():

    imgDir = 'images_for_preview/60x images/'
    #Load Input data
    inputIMGs = [] #May want to use pandas??
    imgNameList = os.listdir(imgDir + 'input/')

    inputIMGs = np.zeros((1,len(imgNameList), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

    #inputIMGs = np.array([])
    for i, imgName in enumerate(imgNameList):
        img = cv2.imread(imgDir + 'input/' + imgName,0)
        print(img.shape)
        #img = Image.fromarray(img, 'GRAY')
        #image = image.resize((SIZE, SIZE))
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        #if i == 0:
        #    inputIMGs = img
        #else:
        #    inputIMGs = np.dstack((inputIMGs, img))
        #inputIMGs.append(np.array(img))
        inputIMGs[0,i] = img

    #Load Target Data
    targetIMGs = []
    imgNameList = os.listdir(imgDir + 'targets/')
    #targetIMGs = np.zeros((len(imgNameList), IMG_HEIGHT, IMG_WIDTH  ), dtype=np.uint8)
    targetIMGs = np.zeros((1, IMG_HEIGHT, IMG_WIDTH  ), dtype=np.uint8)

    #TODO Add ass channels

    for i, imgName in enumerate(imgNameList):
        img = cv2.imread(imgDir + 'targets/' + imgName,0)
        #img = Image.fromarray(img, 'GRAY')
        #image = image.resize((SIZE, SIZE))
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        #targetIMGs.append(np.array(img))
        #targetIMGs[i] = img
        targetIMGs[0] = img

    return(inputIMGs, targetIMGs)



#Load Data
(inputIMGs,targetIMGs) = loadZoomLevles()

#print(inputIMGs.shape)
#print(targetIMGs.shape)

#Defining Model
SIZE = inputIMGs[0].shape[1]
levels = 7

#print(SIZE, SIZE,levels)

INPUT_SHAPE = (SIZE, SIZE,levels)


#TODO Addappt model to data
#Build the model
inputs = tf.keras.layers.Input(INPUT_SHAPE)
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

#Contraction path
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive path
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#Modelcheckpoint
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose=1, save_best_only=True)

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]

print(inputIMGs.shape)
print(targetIMGs.shape)
results = model.fit(inputIMGs, targetIMGs, validation_split=0.0, batch_size=16, epochs=25, callbacks=callbacks)
