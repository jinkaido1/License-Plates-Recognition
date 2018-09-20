# from tesserocr import PyTessBaseAPI
# import tesserocr
# import cv2

# images = ['images/ccc.jpg', 'sample2.jpg', 'sample3.jpg']
# img = 'images/ccc.jpg'

# image = cv2.imread(img)
# cv2.imshow('Image', image)
# cv2.waitKey(0)

# print(tesserocr.image_to_text(image))

# with PyTessBaseAPI() as api:
#     api.SetImageFile(img)
#     print(api.GetUTF8Text())
#     print(api.AllWordConfidences())

    
#     print('Done')
#     # for img in images:
#     #     api.SetImageFile(img)
#     #     print api.GetUTF8Text()
#     #     print api.AllWordConfidences()
# # api is automatically finalized when used in a with-statement (context manager).
# # otherwise api.End() should be explicitly called when it's no longer needed.

from tesserocr import PyTessBaseAPI, PSM, OEM
import tesserocr
from PIL import Image
import cv2
import pytesseract

image = cv2.imread('images/plate1.jpg')
img_o = image
cv2.imshow('Original', image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#image = cv2.resize(image, (640, -1), interpolation=cv2.INTER_CUBIC)
#image = cv2.resize(image, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC) # INTER_AREA to decrease
image = cv2.bilateralFilter(image, 11, 17, 17)
image = cv2.threshold(image, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#image = cv2.Canny(image, 170, 200)
cv2.imshow('Image', image)
#cv2.imwrite('images/ccc1.jpg', image)

print(tesserocr.tesseract_version())  # print tesseract-ocr version
print(tesserocr.get_languages())  # prints tessdata path and list of available languages

print("Without preprocessing: ")
img_orig = Image.open('images/plate2.jpg')
print("Pytesseract: {}".format(pytesseract.image_to_string(img_o)))
print("OCR: {}".format(tesserocr.image_to_text(img_orig)))
print("With preprocessing: ")
img = Image.fromarray(image)
print("Pytesseract: {}".format(pytesseract.image_to_string(img)))
print("OCR: {}".format(tesserocr.image_to_text(img)))  # print ocr text from image
# or
print('Hola')
#print(tesserocr.file_to_text('images/ccc1.jpg'))

# with PyTessBaseAPI(psm=PSM.OSD_ONLY, oem=OEM.LSTM_ONLY) as api:
#     api.SetImageFile("images/ccc.jpg")

#     os = api.DetectOrientationScript()
#     print ("Orientation: {orient_deg}\nOrientation confidence: {orient_conf}\n"
#             "Script: {script_name}\nScript confidence: {script_conf}").format(**os)
with PyTessBaseAPI(psm=PSM.OSD_ONLY, oem=OEM.LSTM_ONLY) as api:
    api.SetImageFile("images/plate.jpg")

    print(api.GetUTF8Text())
    # os = api.DetectOrientationScript()
    # print ("Orientation: {orient_deg}\nOrientation confidence: {orient_conf}\n"
    #        "Script: {script_name}\nScript confidence: {script_conf}").format(**os)

cv2.waitKey(0)