# Cargamos la imagen python (con ubuntu) con python 3.8
FROM python:3.8

# El directorio donde irá el código
WORKDIR /code

# Copia el fichero de las dependencias de instalación de python
COPY ./requirements.txt /code/requirements.txt

# Instalamos las dependencias de Python
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copia el código de la app a la carpeta de trabajo /code/app
COPY ./app /code/app

# Variables de entorno para Tensorlofw, para que no muestre Warnings y no busque GPU
ENV TF_CPP_MIN_LOG_LEVEL 2

ENV CUDA_VISIBLE_DEVICES -1

# Abrimos el puerto 8000
EXPOSE 8000

# Ejecutamos la instrucción para poner en marcha el servidor
CMD ["uvicorn","--reload", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
