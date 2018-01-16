import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout
from keras.callbacks import TensorBoard
import tensorflow as tf
from sklearn.model_selection import train_test_split
import sklearn
import os

# log file directory name
DIR_NAME_TENSORFLOW_LOGS = "logs"

# simulator data file directory name
DIR_NAME_SIMULATOR_DATA = "simulator_data"

# the following PC_ values are used to 
# mark sample records for preprocessing
PC_CENTER_FLIPPED = 'CF'
PC_RIGHT = 'R'
PC_LEFT = 'L'
PC_CENTER = 'C'
PC_CORRECTION = .22


def load_samples(simulator_data_dir_name, add_flipped_images=True, add_left_right_camera_images=True, samples=None):
    if(samples is None):
        samples = []
    
    # loop through the sub directories under the top-level 
    # simulator directory and load log data
    for dir_name, subdir_list, file_list in os.walk(simulator_data_dir_name):
        for sub_dir_name in subdir_list:   
            log_dir_path = simulator_data_dir_name + "\\" + sub_dir_name
            log_file_path = log_dir_path + '\\driving_log.csv'                  
            with open(log_file_path) as csvfile:
                reader = csv.reader(csvfile)
                # loop through the log file adding a processing flag
                for line in reader:
                    # correct the image paths                    
                    for i in range(0,3):
                        path = line[i]                        
                        line[i] = log_dir_path + '\\IMG\\' + path.split('\\')[-1]                          
                        
                    if(add_flipped_images):                
                        samples.append(line + [PC_CENTER_FLIPPED])
                    if(add_left_right_camera_images):
                        samples.append(line + [PC_RIGHT])
                        samples.append(line + [PC_LEFT])            
                    samples.append(line + [PC_CENTER])
        break

    return samples

def get_line_image_and_angle(batch_sample):
    processing_code = batch_sample[-1]
                
    if(processing_code == PC_LEFT):                
        name = batch_sample[1]
        angle = float(batch_sample[3]) + PC_CORRECTION
    elif (processing_code == PC_RIGHT):                
        name = batch_sample[2]
        angle = float(batch_sample[3]) - PC_CORRECTION
    elif (processing_code == PC_CENTER):                
        name = batch_sample[0]
        angle = float(batch_sample[3])
    elif (processing_code == PC_CENTER_FLIPPED):                                
        name = batch_sample[0]
        angle = -float(batch_sample[3])

    image = cv2.imread(name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if(processing_code == PC_CENTER_FLIPPED):
        image = np.fliplr(image)   

    return image, angle


def generator(samples, batch_size=32):
    num_samples = len(samples)
    
    while 1: 
        sklearn.utils.shuffle(samples)        
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []            
            for batch_sample in batch_samples:
                image, angle = get_line_image_and_angle(batch_sample)                
                images.append(image)
                angles.append(angle)
            
            X_train = np.array(images)
            y_train = np.array(angles)    
            
            yield sklearn.utils.shuffle(X_train, y_train)


def split_train_validation_samples(samples, test_size):
    return train_test_split(samples, test_size=0.2)

def print_layer_info(model):
    for layer in model.layers:
        print(layer.name + " input shape: " + str(layer.input_shape))
        print(layer.name + " output shape: " + str(layer.output_shape))
        print()


# define the number of epochs and the batch size
EPOCHS = 16
BATCH_SIZE = 500

# load sample data collected from multiple simulation runs
samples = load_samples(DIR_NAME_SIMULATOR_DATA)

print("The number of samples in the sample set: " + str(len(samples)))

# split the sample data into training and validation sets
train_samples, validation_samples = split_train_validation_samples(samples, .2)

# setup the generator functions
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

# define the input shape
input_shape = (160, 320, 3)

# define the model
model = Sequential()
model.add(Cropping2D(cropping=((60,25), (0,0)), input_shape=input_shape, name='cropping_layer'))
model.add(Lambda(lambda x: x / 255.0 - 0.5, name='normalization_layer'))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu", name='conv_layer1'))
model.add(Dropout(0.2))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu", name='conv_layer2'))
model.add(Dropout(0.2))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu", name='conv_layer3'))
model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3, activation="relu", name='conv_layer4'))
model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3, activation="relu", name='conv_layer5'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100, name='fullyconnected_layer1'))
model.add(Dropout(0.2))
model.add(Dense(50, name='fullyconnected_layer2'))
model.add(Dropout(0.2))
model.add(Dense(10, name='fullyconnected_layer3'))
model.add(Dropout(0.2))
model.add(Dense(1, name='output_layer'))

print_layer_info(model)

# compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

tensorboard = TensorBoard(log_dir=DIR_NAME_TENSORFLOW_LOGS,
                 histogram_freq=1, 
                 write_graph=True, 
                 write_images=False)

# train the model
model.fit_generator(train_generator, 
    samples_per_epoch=len(train_samples), 
    validation_data=validation_generator, 
    nb_val_samples=len(validation_samples),
    nb_epoch=EPOCHS,
    callbacks=[tensorboard])

# save the model
model.save('model.h5')  


