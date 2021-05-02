import csv
import cv2
import numpy as np

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


from keras.models import Sequential 
from keras.layers import Flatten, Dense, Lambda

model = Sequential()

model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))

model.add(Flatten())
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam")
model.fit(X_train, y_train, validation_split=20, shuffle=True, epochs=2)

model.save("model.h5")
                       