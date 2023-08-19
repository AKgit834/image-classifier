from django.shortcuts import render
from all.models import images
import joblib
import json
import numpy as np
import base64
import cv2
import pywt

__model = None

def classify(b64_data=None,file_path=None):
    imgs=cropper(file_path,b64_data)

    for img in imgs:
        scalled_img = cv2.resize(img, (32, 32))
        img_h = w2d(img, 'db1', 5)
        scalled_img_h = cv2.resize(img_h, (32, 32))
        combined_img = np.vstack((scalled_img.reshape(32 * 32 * 3, 1), scalled_img_h.reshape(32 * 32, 1)))

        len_image = 32*32*3 + 32*32
        final = combined_img.reshape(1,len_image).astype(float)

        res=__model.predict(final)[0]
    return res


def w2d(img, mode='haar', level=1):
    arr = img

    arr = cv2.cvtColor( arr,cv2.COLOR_RGB2GRAY )
    arr =  np.float32(arr)
    arr /= 255
    coeffs=pywt.wavedec2(arr, mode, level=level)
    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0

    arr_h=pywt.waverec2(coeffs_H, mode)
    arr_h *= 255
    arr_h =  np.uint8(arr_h)

    return arr_h



def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def cropper(imgs_pth,b64_data):
    face = cv2.CascadeClassifier('D:/vs/jupyter/Source_files/Family_classifier/haarcascade_frontalface_default.xml')
    eye  = cv2.CascadeClassifier('D:/vs/jupyter/Source_files/Family_classifier/haarcascade_eye.xml')
    
    if imgs_pth:
        img = cv2.imread(imgs_pth)
    else:
        img=get_cv2_image_from_base64_string(b64_data)
    
    cropped=[]
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face_detected = face.detectMultiScale(gray_img,1.3,5)
    for (x,y,w,h) in face_detected:
        aoi_gray = gray_img[y:y+h,x:x+w]
        aoi_color = img[y:y+h,x:x+w]
        eyes = eye.detectMultiScale(aoi_gray)
        if len(eyes) >=2 :
            cropped.append(aoi_color)
    return cropped

def load_saved_artifacts():
    print("loading saved artifacts...start")

    global __model
    if __model is None:
        with open('D:/vs/jupyter/Source_files/Family_classifier/model.pkl', 'rb') as f:
            __model = joblib.load(f)
    print("loading saved artifacts...done")


def index(request):
    if request.method == "POST":
        img_pth = request.POST.get('path')
        print(type(img_pth))
        print(img_pth)
        # new_img_pth=img_pth.replace(,'/')
        # print(new_img_pth)
        load_saved_artifacts()
        ans=classify(file_path=img_pth)
        context={
            'ans':ans
        }


    return render(request,'index.html',context=context)
