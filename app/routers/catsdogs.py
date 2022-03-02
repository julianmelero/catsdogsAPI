#Python

import os
import json


# FastAPI
from io import BytesIO
from fastapi import APIRouter, File, UploadFile, HTTPException,status

# External
import numpy as np
from PIL import Image
import cv2
import keras
import tensorflow as tf

router = APIRouter(
    prefix="/catsdogs"
    )

IMAGES_ACCEPTED = ["image/png","image/jpeg"]

@router.post(path="/post-image", tags=["CatsDogs"])
def post_image(
    image: UploadFile 
):
    if image.content_type not in IMAGES_ACCEPTED:
        raise HTTPException(status_code=status.HTTP_406_NOT_ACCEPTABLE,detail="Image accepted type " + str(IMAGES_ACCEPTED))
        
    # I load the image with Pillow
    imagen = Image.open(BytesIO(image.file.read()))       
    # imagen = np.array(imagen).astype(float) 
    # image = cv2.resize(imagen,(100, 100)) # Thats becouse we need all images with the same size.
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Set images to white and black color.
    # image = image.reshape (100,100,1) # 1 becouse only has 1 channel of color. 
    # Image gray, because I need 1 channel
    # img_gray = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
    # Resize Image for requirements model


    # imagen_resize = imagen.resize((100,100))
    imagen = np.array(imagen).astype('uint8')
    imagen[0] = imagen[0] / 255
    imagen[1] = imagen[1] / 255  
    imagen = cv2.resize(imagen,(100, 100))
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen = imagen.reshape(100,100,1)




    # Load to numpy as array
    # image_np = np.array(imagen_resize).astype(float) 
    # image_np.reshape([0,255,255,3])
    # image_np[0] = image_np[0] / 255
    # image_np[1] = image_np[1] / 255   
    image_np = np.expand_dims(imagen,axis=0)
    print(image_np.shape)
    
    # image_np = tf.reshape(imagen, [150,150])
    
    # Image gray, because I need 1 channel
    # img_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)    
    # Load to numpy as array
    # image_np = np.array(img_gray).astype('uint8') 
    # Seek image for read again
    image.file.seek(0)
    model =  keras.models.load_model(os.path.join("model","dogs-cats-cnn-ad.h5"))
    prediction = model.predict(image_np)
    pred = prediction[0][0]
    return {
        "predict": int(pred)
    }