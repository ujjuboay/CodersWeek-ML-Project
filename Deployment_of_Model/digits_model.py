import numpy as np
import cv2
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten

class digits_recog(object):

    def __init__(self):
        self.img_width, self.img_height = 130, 130

        self.train_data_dir = '/home/ujjuboay/Digits_recognition/DevanagariHandwrittenCharacterDataset/Train'
        self.validation_data_dir = '/home/ujjuboay/Digits_recognition/DevanagariHandwrittenCharacterDataset/Test'

        self.datagen = ImageDataGenerator(rescale = 1./255)

        self.train_generator = self.datagen.flow_from_directory(
        self.train_data_dir,
        target_size = (self.img_width, self.img_height),
        batch_size = 512,
	color_mode = 'grayscale',
        class_mode = 'categorical')

        self.validation_generator = self.datagen.flow_from_directory(
        self.validation_data_dir,
        target_size = (self.img_width, self.img_height),
        batch_size = 512,
	color_mode = 'grayscale',
        class_mode = 'categorical')

        self.x_train, self.y_train = self.train_generator.next()
        self.x_test, self.y_test = self.validation_generator.next()

        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (self.img_width, self.img_height, 1)))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation = 'relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation = 'relu'))
        self.model.add(Dense(10, activation = 'softmax'))

        self.model.compile(optimizer = 'adam',loss = 'categorical_crossentropy', metrics = ['accuracy'])
        self.model.fit_generator(self.train_generator, steps_per_epoch 33, epochs = 30, validation_data = self.validation_generator, validation_steps = 6)

        self.model_json = self.model.to_json()
        with open("hindi_digits_model.json", "w") as json_file:
            json_file.write(self.model_json)
        self.model.save_weights("hindi_digits_model.h5")
        print("Saved model to disk")

digit = digits_recog()





