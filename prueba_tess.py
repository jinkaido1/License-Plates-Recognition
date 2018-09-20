from tesserocr import PyTessBaseAPI, PSM, OEM
import tesserocr
from PIL import Image
import cv2
import pytesseract


img_path = 'images/plate.jpg'
img_orig = Image.open(img_path)

print(pytesseract.image_to_string(img_path, config = "-l eng"))

print(tesserocr.image_to_text(img_orig))
# with PyTessBaseAPI(psm=PSM.OSD_ONLY, oem=OEM.LSTM_ONLY) as api:
#     api.SetImageFile(img_path)

#     print(api.GetUTF8Text())