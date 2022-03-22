import os
import keras


def load_model():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
    model =  keras.models.load_model(os.path.join("app/model","dogs-cats-cnn-ad.h5"))
    return model