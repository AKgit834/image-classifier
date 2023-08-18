import numpy as np
import pywt
import cv2

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