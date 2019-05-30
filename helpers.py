import keras
from keras import backend as K
import tensorflow 
import cv2
import os

def reset_keras():
    K.clear_session()
    
def download_data_from_master():
    return 1

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_callbacks(model_name, save_dir):
    from keras.callbacks import CSVLogger, ModelCheckpoint
    checkpoint = keras.callbacks.ModelCheckpoint(save_dir + model_name +'.h5', save_best_only = True) 
    csv_logger = CSVLogger(save_dir + model_name + "_history.csv", append=True)
    return [checkpoint, csv_logger]