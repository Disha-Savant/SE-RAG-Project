from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse  # Import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import os
import shutil

from . import database, rag, schemas

app = FastAPI()

# Initialize Jinja2 Templates
templates = Jinja2Templates(directory="app/templates")

# Serve the index.html page
@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"filename": file.filename}

@app.post("/populate/")
def populate_db(reset: bool = False):
    database.populate(reset=reset)
    return {"message": "Database populated."}

@app.post("/query/")
def ask_question(request: schemas.QueryRequest):
    result = rag.query_rag(request.query)
    return JSONResponse(content=result)
