from fastapi import FastAPI
from app.api.routes import router

# To initialize DB on app startup
from app.db import init_db


app = FastAPI(title="Text Insight Hub API", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(router, prefix="/api")

#app = FastAPI(title="Text Insight Hub API", version="0.1.0")

@app.on_event("startup")
def on_startup():
    init_db()
