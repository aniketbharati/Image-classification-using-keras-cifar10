import json
import cv2
from keras.models import model_from_json
import numpy as np

classes=['aeroplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

json_file=open('modelclass.json','r')
loaded_model_json=json_file.read()

json_file.close()

loaded_model=model_from_json(loaded_model_json)

loaded_model.load_weights('modelclass.h5')

loaded_model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])


test=cv2.imread('im2.jpg')
cv2.imshow('im1.jpg',test)
resized_test = cv2.resize(test, (32,32))
resized_test = np.reshape(resized_test,[1,32,32,3])
resized_data=resized_test.astype('float32')
resized_data/=255

pred=loaded_model.predict(resized_data)


n=0
for i in pred[0]:
		
	print format(i*100, '.2f'),'% :',classes[n] 
	n=n+1

#print pred








