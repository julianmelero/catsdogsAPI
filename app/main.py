from sys import prefix

# FastAPI
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware


# Routes
from routers import catsdogs


app = FastAPI(prefix="/api/v1")


origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/model", StaticFiles(directory="model"), name="model")
app.include_router(catsdogs.router, prefix="/api/v1")

