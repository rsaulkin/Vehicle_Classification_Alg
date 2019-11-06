# Create the CNN and train it, using our input DS and image data generator, to increase the current data used fro the CNN

# Import the required packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from sklearn.metrics import classification_report

# Initialising the CNN
model = Sequential()

# Convolution
model.add(Conv2D(32, (3, 3), input_shape = (32, 32, 3), activation = 'relu'))

# SPooling
model.add(MaxPooling2D(pool_size = (2, 2)))

# Second convolution layer
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
model.add(Flatten())

# Full connection
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Perform Image Data Generator
from keras.preprocessing.image import ImageDataGenerator

trainDataGen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Load the training Set
trainSet = trainDataGen.flow_from_directory('./DS',
                                            target_size = (32, 32),
                                            batch_size = 32,
                                            class_mode = 'binary')
# Train the classifier model
model.fit_generator(trainSet, steps_per_epoch = 100, epochs = 25, validation_steps = 50)

# Convert the Model to json file and save it
modelJson = model.to_json()
with open("./model.json","w") as json_file:
    json_file.write(modelJson)

# Save the weights into a h5 file
model.save_weights("./model.h5")

print("Classifier trained Completed")