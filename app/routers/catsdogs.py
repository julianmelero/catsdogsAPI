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

IMAGES_ACCEPTED = ["image/png","image/jpeg","image/jpg"]

@router.post(path="/post-image", tags=["CatsDogs"])
def post_image(
    image: UploadFile = File(...)
):
    if image.content_type not in IMAGES_ACCEPTED:
        raise HTTPException(status_code=status.HTTP_406_NOT_ACCEPTABLE,detail="Image accepted type " + str(IMAGES_ACCEPTED))
          
    # I load the image with Pillow
    
    imagen = Image.open(BytesIO(image.file.read()))       
    
    imagen = np.array(imagen).astype('uint16')
    imagen = cv2.resize(imagen,(100, 100))
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen = imagen.reshape(100,100,1)
    imagen = np.array(imagen).astype(float)
    imagen = imagen / 255    
    image_np = np.expand_dims(imagen,axis=0)    
    
    
    model =  keras.models.load_model(os.path.join("model","dogs-cats-cnn-ad.h5"))
    prediction = model.predict(image_np)
    print(prediction[0][0])
    if prediction[0][0] <= 0.5:
        pred ="Cat"
    else:
        pred ="Dog"   
    
    return {
        "predict": pred,
        "prediction":  round(float(prediction[0][0]),2)
    }