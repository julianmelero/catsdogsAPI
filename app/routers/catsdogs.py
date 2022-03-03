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



router = APIRouter(
    prefix="/catsdogs"
    )

IMAGES_ACCEPTED = ["image/png","image/jpeg","image/jpg"]

@router.post(path="/post-image", tags=["CatsDogs"])
def post_image(
    image: UploadFile = File(...)   
):
    
   
    imagen = Image.open(io.BytesIO(image.file.read()))       
   
    
    imagen = np.array(imagen).astype('uint16')
    imagen = cv2.resize(imagen,(100, 100))
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen = imagen.reshape(100,100,1)
    imagen = np.array(imagen).astype(float)
    imagen = imagen / 255    
    image_np = np.expand_dims(imagen,axis=0)    
        
    model =  keras.models.load_model(os.path.join("app/model","dogs-cats-cnn-ad.h5"))
    
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
    