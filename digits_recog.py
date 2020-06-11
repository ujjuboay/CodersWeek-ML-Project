# -*- coding: utf-8 -*-
"""digits_recog.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Rg7IyEJ8ZU9aiGBwyT2Sw0yd4eXh6rdh
"""

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.models import model_from_json
from keras import optimizers
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib.image import imread

img_width, img_height = 130, 130

train_data_dir = '/content/drive/My Drive/DevanagariHandwrittenCharacterDataset/Train'
validation_data_dir = '/content/drive/My Drive/DevanagariHandwrittenCharacterDataset/Test'

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=512,
        color_mode= 'grayscale',
        class_mode='categorical')

validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=512,
        color_mode= 'grayscale',
        class_mode='categorical')

x_train,y_train = train_generator.next()
x_test,y_test = validation_generator.next()

model = Sequential()
model.add(Convolution2D(32,(3,3), input_shape=(img_width, img_height,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

history=model.fit_generator(train_generator,steps_per_epoch=33,epochs=30,validation_data=validation_generator,validation_steps=6)

# evaluate
loss, acc = model.evaluate(x_train, y_train, verbose=0)
print('Test Accuracy: %f' % (acc*100))
print('Test Loss: %f' % (loss))

#print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# serialize model to JSON
model_json = model.to_json()
with open("hindi_digits_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("hindi_digits_model.h5")
print("Saved model to disk")

# load json and create model
json_file = open('hindi_digits_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("hindi_digits_model.h5")
print("Loaded model from disk")

pred = x_test[83]
#print(pred.shape)
prediction = model.predict(pred.reshape(1,130,130,1))
#print(pred.shape)
plt.imshow(np.squeeze(pred),cmap="gray")
plt.show()
print("\n\nFinal output:{}".format(np.argmax(prediction)))

#for showing one more prediction
pred1 = x_test[91] 
prediction1 = model.predict(pred1.reshape(1,130,130,1)) 
plt.imshow(np.squeeze(pred1),cmap="gray")
plt.show()
print("\n\nFinal output:{}".format(np.argmax(prediction1)))

