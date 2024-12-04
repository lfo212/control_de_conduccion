from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Response, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from drivers import Driver, DriverModel, Base, LoginRequest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import shutil
import io

SECRET_TOKEN = "secret-token"
DATABASE_URL = "sqlite:///./drivers.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5000", "http://localhost:5000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def validate_token(authToken: Optional[str] = Cookie(None)):
    if authToken != SECRET_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    return True  # Return something useful if needed

@app.post("/register_driver/", response_model=Driver)
async def register_driver(
    driver_name: str = Form(...),
    driver_id: str = Form(...),
    photo: UploadFile = File(...),
    db: Session = Depends(get_db),
    auth: bool = Depends(validate_token)
):
    photo_path = f"photos/{driver_id}.jpg"
    with open(photo_path, "wb") as buffer:
        shutil.copyfileobj(photo.file, buffer)

    with open(photo_path, "rb") as buffer:
        photo_data = buffer.read()

    db_driver = DriverModel(
        driver_name=driver_name,
        driver_id=driver_id,
        photo=photo_data
    )
    db.add(db_driver)
    db.commit()
    db.refresh(db_driver)
    return db_driver

@app.get("/drivers_list/", response_model=List[Driver])
def drivers_list(
    skip: int = 0,
    limit: int = 10,
    db: Session = Depends(get_db),
    auth: bool = Depends(validate_token)
):
    drivers = db.query(DriverModel).offset(skip).limit(limit).all()
    return drivers

@app.get("/driver_photo/{driver_id}")
def get_driver_photo(
    driver_id: int, 
    db: Session = Depends(get_db), 
    auth: bool = Depends(validate_token)
):
    driver = db.query(DriverModel).filter(DriverModel.id == driver_id).first()
    if driver is None:
        raise HTTPException(status_code=404, detail="Driver not found")
    
    if driver.photo is None:
        raise HTTPException(status_code=404, detail="Photo not found")
    
    return StreamingResponse(io.BytesIO(driver.photo), media_type="image/jpeg")

@app.delete("/delete_driver/{driver_id}")
def delete_driver(
    driver_id: int, 
    db: Session = Depends(get_db),
    auth: bool = Depends(validate_token)
):
    driver = db.query(DriverModel).filter(DriverModel.id == driver_id).first()
    if not driver:
        raise HTTPException(status_code=404, detail="Conductor no encontrado")
    db.delete(driver)
    db.commit()
    return {"message": "Conductor eliminado con exito"}

@app.delete("/delete_all_drivers")
def delete_all_drivers(db: Session = Depends(get_db), auth: bool = Depends(validate_token)):
    db.query(DriverModel).delete()
    db.commit()
    return {"message": "Todos los conductores eliminados con exito"}

@app.post("/api/login")
def login(response: Response, request: LoginRequest):
    if request.username == "admin" and request.password == "admin":
        response.set_cookie(
            key="authToken",
            value=SECRET_TOKEN,
            httponly=True,
            secure=True,
            samesite="None",
            max_age=3600,
            path="/",
        )
        return {"message": "Login successful"}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/api/validate-token")
def validate_token(authToken: str = Cookie(None)):
    if authToken == SECRET_TOKEN:
        return {"valid": True}
    raise HTTPException(status_code=401, detail="Invalid token")
