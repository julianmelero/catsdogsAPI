from sys import prefix

# FastAPI
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Routes
from routers import catsdogs


app = FastAPI(prefix="/api/v1")

app.mount("/model", StaticFiles(directory="model"), name="model")
app.include_router(catsdogs.router, prefix="/api/v1")

