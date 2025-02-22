from fastapi import FastAPI
import uvicorn
from app import app  # Import your FastAPI app from app.py

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
