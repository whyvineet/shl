import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

from app.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


app = FastAPI(
    title="SHL",
    version="1.0.0"
)

app.include_router(router)