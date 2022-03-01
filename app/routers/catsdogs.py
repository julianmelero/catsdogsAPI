#Python

import os


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
    # Resize Image for requirements model
    imagen_resize = imagen.resize((150,150))    
    # Load to numpy as array
    image_np = np.array(imagen_resize)#.astype('uint8') 
    # image_np.reshape([0,255,255,3])
    image_np[0] = image_np[0] / 255
    image_np[1] = image_np[1] / 255   
    image_np = np.expand_dims(image_np,0)

    
    # image_np = tf.reshape(imagen, [150,150])
    
    # Image gray, because I need 1 channel
    # img_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)    
    # Load to numpy as array
    # image_np = np.array(img_gray).astype('uint8') 
    # Seek image for read again
    image.file.seek(0)
    model =  keras.models.load_model(os.path.join("model","catsdogs.h5"))
    print(model.predict(image_np))
    return {
        "Filename": image.filename,
        "Format": image.content_type,
        "Size(kb)": round(len(image.file.read()) / 1024,2),
        "np": image_np.shape
    }