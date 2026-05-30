from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"status": "running"}

@app.get("/health")
def health():
    return {"system": "healthy"}