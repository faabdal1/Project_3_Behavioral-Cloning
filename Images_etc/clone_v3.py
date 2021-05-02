import csv
import cv2
import numpy as np
import tensorflow as tf

lines = []

with open("../data_training/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
#print(lines[0][2])

images = []
measurements = []

for line in lines:
    source_path = line[0]
    filename = source_path.split("/")[-1]
    current_path = "../data_training/IMG/" + filename
    images.append(cv2.imread(current_path))
    measurements.append(float(line[3]))

#img = images[0] 
#cv2.imshow(img)
#cv2.waitKey(10)

    
X_train = np.array(images)
y_train = np.array(measurements)   


freeze_flag = True  # `True` to freeze layers, `False` for full training
weights_flag = 'imagenet' # 'imagenet' or None
preprocess_flag = True # Should be true for ImageNet pre-trained typically


# Loads in InceptionV3
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model


input_size = 299

inception = InceptionV3(weights=weights_flag, include_top=False, input_shape=(input_size,input_size,3))

cifar_input = Input(shape=(160,320,3))

# Re-sizes the input with Kera's Lambda layer & attach to cifar_input
resized_input = Lambda(lambda image: __import__("tensorflow").image.resize_images(image, (299, 299)))(cifar_input)

inp = inception(resized_input)

#if freeze_flag == True:
#    ## TODO: Iterate through the layers of the Inception model
##    ##       loaded above and set all of them to have trainable = False
 #   for layer in inception.layers:
 #       layer.trainable = False
        

x = inception.get_output_at(-1)        
out = Flatten()(x)
out = Dense(1)(out)

model = Model(inputs=cifar_input, outputs=out)



#model.summary()
model.compile(loss="mse", optimizer="adam")
model.fit(X_train, y_train, validation_split=20, shuffle=True, epochs=2)

model.save("model2.h5")


#from keras.models import Sequential 
#from keras.layers import Flatten, Dense, Lambda

#model = Sequential()

#model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))

#model.add(Flatten())
#model.add(Dense(1))

#model.compile(loss="mse", optimizer="adam")
#model.fit(X_train, y_train, validation_split=20, shuffle=True, epochs=2)

#model.save("model.h5")
                       