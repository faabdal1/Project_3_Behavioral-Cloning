import csv
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

lines = []

#with open("../data_training/driving_log.csv") as csvfile:
with open("../New_data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
#print(lines[0][2])

images = []
measurements = []
car_images = []
steering_angles = []

for line in lines:
    steering_center = float(line[3])
    # create adjusted steering measurements for the side camera images
    correction = 0.2 # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    
    #source_path = line[0]
    #filename = source_path.split("/")[-1]
    #current_path = "../data_training/IMG/" + filename
    #images.append(cv2.imread(current_path))
    #measurements.append(float(line[3]))
        
    # read in images from center, left and right cameras
    #path = "../data_training/IMG/" # fill in the path to your training IMG directory
    path = "../New_data/IMG/" # fill in the path to your training IMG directory
    #print(line[0])
    filename = line[0].split("/")[-1]
    #print(path + filename)
    img_center = cv2.imread(path + filename)
    img_center = cv2.cvtColor(img_center, cv2.COLOR_BGR2RGB)
    
    
    filename = line[1].split("/")[-1]
    #print(path + filename)
    img_left = cv2.imread(path + filename)
    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)

    filename = line[2].split("/")[-1]
    img_right = cv2.imread(path + filename)
    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)

    
    images.append(img_center)
    measurements.append(steering_center)
    images.append(cv2.flip(img_center,1))
    measurements.append(steering_center*-1.0)
    
    
    images.append(img_left)
    measurements.append(steering_left)
    images.append(cv2.flip(img_left,1))
    measurements.append(steering_left*-1.0)
    
    images.append(img_right)
    measurements.append(steering_right)
    images.append(cv2.flip(img_right,1))
    measurements.append(steering_right*-1.0)
        
    
    # add images and angles to data set
    #car_images.extend(img_center, img_left, img_right)
    #steering_angles.extend(steering_center, steering_left, steering_right)


print(len(images))
#cv2.imshow(img)
#cv2.waitKey(10)

    
X = np.array(images)
y = np.array(measurements)   

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

print(len(X_train))
print(len(X_test))


from keras.models import Sequential 
from keras.layers import Flatten, Dense, Lambda, Activation, MaxPooling2D, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D



def leNet():
    
    model = Sequential()
    model.add(Conv2D(6, (5, 5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(6, (5, 5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model


def Nvidia():

    model = Sequential()
    #nomalization
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    #We use strided convolutions in the first three convolutional layers with a 2×2 stride 
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
    return model


model = Nvidia()

model.compile(loss="mse", optimizer="adam")
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10, verbose = 1)
model.save("mode_new_data2.h5")
test = model.evaluate(X_test, y_test)

print("Test loss: " + str(test))
                       
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.figure()
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.plot([0,1,2,3,4,5,6,7,8,9], [test,test,test,test,test,test,test,test,test,test])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set', 'test set (after training)'], loc='upper right')
plt.savefig('loss_final2.png')

#plt.show()