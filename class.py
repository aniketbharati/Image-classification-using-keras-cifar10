from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import keras 
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.datasets import cifar10

(train_images,train_labels),(test_images,test_labels)=cifar10.load_data()

print("training data shape:",train_images.shape,train_labels.shape)
print("testing data shape:",test_images.shape,test_labels.shape)

classes=np.unique(train_labels)
nclasses=len(classes)
#print(train_images,"\n\n\n\n")
print("total number of outputs:",nclasses)
print("output classes:",classes)

nrows,ncols,ndims=train_images.shape[1:]

#print(len(train_images),nrows,ncols,ndims)
#print(train_images.shape[0],train_images.shape[1],train_images.shape[2],train_images.shape[3])

train_data=train_images.reshape(train_images.shape[0],nrows,ncols,ndims)

test_data=test_images.reshape(test_images.shape[0],nrows,ncols,ndims)

#print(test_data,"\n\n\n\n")

train_data=train_data.astype('float32')
test_data=test_data.astype('float32')

#print(test_data,"\n\n\n\n")

train_data/=255
test_data/=255

#print(test_data)

train_labels_one_hot=to_categorical(train_labels)
test_labels_one_hot=to_categorical(test_labels)

print(train_labels[0])
print(train_labels_one_hot[0])

def createModel():

	model=Sequential()

	model.add(Conv2D(32,(3,3),padding='same',activation='relu',input_shape=(nrows,ncols,ndims)))
	model.add(Conv2D(32,(3,3),activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64,(3,3),padding='same',activation='relu',input_shape=(nrows,ncols,ndims)))
	model.add(Conv2D(64,(3,3),activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64,(3,3),padding='same',activation='relu',input_shape=(nrows,ncols,ndims)))
	model.add(Conv2D(64,(3,3),activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nclasses,activation='softmax'))

	return model


model=createModel()

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

batch_size=256
epochs=50

datagen=ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True,
vertical_flip=False)

history=model.fit_generator(datagen.flow(train_data,
train_labels_one_hot,batch_size=batch_size),steps_per_epoch=int(np.ceil(train_data.shape[0]/float(batch_size))),epochs=epochs,validation_data=(test_data,test_labels_one_hot
),workers=4)


model_json = model.to_json()
with open("modelclass.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("modelclass.h5")
print("Saved model to disk")

