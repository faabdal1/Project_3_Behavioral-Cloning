import os
import csv
import math
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sklearn
import random
from keras.models import Sequential 
from keras.layers import Flatten, Dense, Lambda, Activation, MaxPooling2D, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D



# Load recorded csv to load training images and angles
samples = []
with open('../Data_training_new/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


#Split training data and alocate random 20% to validation set
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#Define generator to load images continously instead of loading all of them at once and having problem with memory 
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # loading images and angles from center camera
                name = '../Data_training_new/IMG/'+batch_sample[0].split('/')[-1]
                #images are saved in BGR
                center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                # creation extra training image by flipping it on shorter side and using opposite angle
                images.append(cv2.flip(center_image,1))
                angles.append(center_angle*-1.0)
                
                # loading images and angles from left camera
                # compensation to the angle + 0.2
                name = '../Data_training_new/IMG/'+batch_sample[1].split('/')[-1]
                #images are saved in BGR
                left_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                left_angle = (float(batch_sample[3])+0.2)
                images.append(left_image)
                angles.append(left_angle)
                # creation extra training image by flipping it on shorter side and using opposite angle
                images.append(cv2.flip(left_image,1))
                angles.append(left_angle*-1.0)
                
                # loading images and angles from right camera
                # compensation to the angle - 0.2
                name = '../Data_training_new/IMG/'+batch_sample[2].split('/')[-1]
                #images are saved in BGR
                right_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                right_angle = (float(batch_sample[3])-0.2)
                images.append(right_image)
                angles.append(right_angle)
                # creation extra training image by flipping it on shorter side and using opposite angle
                images.append(cv2.flip(right_image,1))
                angles.append(right_angle*-1.0)

            # Create training set
            X_train = np.array(images)
            y_train = np.array(angles)
            # Shuggle the dataset 
            yield sklearn.utils.shuffle(X_train, y_train)

# Set batch size
batch_size=32

# Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
# Cropp part of the image relevant for the driving
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

#Nvidia model (https://arxiv.org/pdf/1604.07316.pdf)
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# compile model
model.compile(loss='mse', optimizer='adam')
# train the model
history_object = model.fit_generator(train_generator, 
            steps_per_epoch=math.ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=math.ceil(len(validation_samples)/batch_size), 
            epochs=10, verbose=1)


#save the trained model 
model.save("model.h5")

                      
### print the keys contained in the history object
#print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.figure()
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
#save the plot as and image 
plt.savefig('loss.png')

#plot the image
#plt.show()