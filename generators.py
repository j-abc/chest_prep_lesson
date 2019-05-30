from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2
import os
import keras
from keras.preprocessing import image
import tensorflow 

### SOME PARAMS ###
data_folder = '/home/jupyter/chest_xray_subset/'
set_labels   = ['test','val','train']
class_labels = ['PNEUMONIA','NORMAL']
class_mode   = 'binary'
image_size   = (224, 224)

def get_train_data_generator(augment = False, color = 'rgb'):
    '''
    Returns a Python generator (see intro Python lecture) that returns an image
    that has gone through the data augmentation process from the training set.
    '''

    # Data augmentation for training dataset.
    if augment:
        train_datagen = ImageDataGenerator(
              rescale=1./255,
              shear_range=0.2,
              zoom_range=0.2,
              horizontal_flip=True)        
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)

    # Create a Python 'generator' for reading pictures from 
    # the 'Datasets/chest_xray/train' folder, and indefinitely 
    # generate batches of augmented image data.
    image_directory = os.path.join(data_folder, 'train')  
    train_generator = train_datagen.flow_from_directory(
          image_directory, 
          target_size=image_size,
          batch_size=32,
          color_mode=color, # depends on the dataset
          class_mode=class_mode)    
    return train_generator

def get_val_data_generator(color = 'rgb'):
    '''
    Returns a Python generator (see intro Python lecture) that returns an image
    that has gone through the data augmentation process from the training set.
    '''

    val_datagen = ImageDataGenerator(rescale=1./255)

    # Create a Python 'generator' for reading pictures from 
    # the 'Datasets/chest_xray/train' folder, and indefinitely 
    # generate batches of augmented image data.
    image_directory = os.path.join(data_folder, 'test')  
    val_generator = val_datagen.flow_from_directory(
          image_directory, 
          target_size=image_size,
          batch_size=200,
          color_mode=color, # depends on the dataset
          class_mode=class_mode)    
    return val_generator