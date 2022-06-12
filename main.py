from typing import Union

from fastapi import FastAPI
import cv2
import face_recognition
import urllib.request
import numpy as np
from skimage import io

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "Welcome to face verification API"}


@app.get("/verifyface/")
def verify_face(image_1: str, image_2: str):
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'
    req = urllib.request.urlopen(image_1)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)  # 'Load it as it is'
    im1 = io.imread(image_1)
    # img = cv2.imread("Messi1.webp")
    rgb_img = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    img_encoding = face_recognition.face_encodings(rgb_img)[0]

    req2 = urllib.request.urlopen(image_2)
    arr2 = np.asarray(bytearray(req2.read()), dtype=np.uint8)
    img2 = cv2.imdecode(arr2, -1)  # 'Load it as it is'
    
    im2 = io.imread(image_2)
    # img2 = cv2.imread("images/Jeff Bezoz.jpg")
    rgb_img2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

    result = face_recognition.compare_faces([img_encoding], img_encoding2)
    print("Result: ", result)
    return {"Info": "FaceVerification Data", 'verification': str(result)}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}



