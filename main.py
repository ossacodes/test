from typing import Union

from fastapi import FastAPI
import cv2
import face_recognition
import urllib.request
import numpy as np

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/verifyface")
def verify_face():
    req = urllib.request.urlopen(
        'https://imagesvc.meredithcorp.io/v3/mm/image?url=https%3A%2F%2Fstatic.onecms.io%2Fwp-content%2Fuploads%2Fsites%2F20%2F2022%2F04%2F19%2Fcristiano-ronaldo-1.jpg&q=60')
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)  # 'Load it as it is'

    # img = cv2.imread("Messi1.webp")
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_encoding = face_recognition.face_encodings(rgb_img)[0]

    req2 = urllib.request.urlopen(
        'https://i2-prod.mirror.co.uk/incoming/article26777809.ece/ALTERNATES/s1200c/0_GettyImages-1240041916-1.jpg')
    arr2 = np.asarray(bytearray(req2.read()), dtype=np.uint8)
    img2 = cv2.imdecode(arr2, -1)  # 'Load it as it is'

    # img2 = cv2.imread("images/Jeff Bezoz.jpg")
    rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

    result = face_recognition.compare_faces([img_encoding], img_encoding2)
    print("Result: ", result)
    return {"Hello": "World", 'verification': str(result)}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}



