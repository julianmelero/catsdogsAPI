#Python
import os



# FastAPI
from fastapi import APIRouter, File, UploadFile, HTTPException,status
# External
import numpy as np
from PIL import Image
import io
import cv2
import keras
import tensorflow as tf



router = APIRouter(
    prefix="/catsdogs"
    )

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
model =  keras.models.load_model(os.path.join("app/model","dogs-cats-cnn-ad.h5"))

IMAGES_ACCEPTED = ["image/png","image/jpeg","image/jpg"]

@router.post(path="/post-image", tags=["CatsDogs"])
def post_image(
    image: UploadFile = File(...)   
):
    print(image.content_type)
    if image.content_type not in IMAGES_ACCEPTED:
        raise HTTPException(status_code=502,
        detail="Esto no es una imagen con formato correcto")
   
    imagen = Image.open(io.BytesIO(image.file.read()))       
   
    
    imagen = np.array(imagen).astype('uint16')     
    imagen = cv2.resize(imagen,(100, 100))
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    # imagen = imagen.reshape(100,100,1)    
    imagen = np.array(imagen).astype(float)
    imagen = imagen / 255    
    image_np = np.expand_dims(imagen,axis=0)    


    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
    # model =  keras.models.load_model(os.path.join("app/model","dogs-cats-cnn-ad.h5"))
    
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
    