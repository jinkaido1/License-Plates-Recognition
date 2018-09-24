import numpy as np
import cv2
import  imutils
import tesserocr
from PIL import Image
import pytesseract

# Read the image file
original_img = cv2.imread('images/renault_back.jpg')
cv2.imshow('ooo', original_img)
original_img_height, original_img_width = original_img.shape[:2]
#original_img = image

# Resize the image - change width to 500
image = imutils.resize(original_img, width=500)

#original_img = imutils.resize(image, width=500)
# Store image height and width
image_height, image_width = image.shape[:2]
height_ratio = original_img_height / image_height
width_ratio = original_img_width / image_width
print("Ratios: Height -> {}, Width -> {}".format(height_ratio, width_ratio))

# Display the original image
cv2.imshow("Original Image", image)

# RGB to Gray scale conversion
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("1 - Grayscale Conversion", gray)

# Noise removal with iterative bilateral filter(removes noise while preserving edges)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("2 - Bilateral Filter", gray)

# Find Edges of the grayscale image
edged = cv2.Canny(gray, 170, 200)
cv2.imshow("4 - Canny Edges", edged)

# Find contours based on Edges
(new, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30] #sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
NumberPlateCnt = None #we currently have no Number plate contour

# loop over our contours to find the best possible approximate contour of number plate
count = 0
for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:  # Select the contour with 4 corners
            NumberPlateCnt = approx #This is our approx Number Plate Contour
            break

# Drawing the selected contour on the original image
cv2.drawContours(image, [NumberPlateCnt], -1, (0,255,0), 3)
cv2.imshow("Final Image With Number Plate Detected", image)

cv2.waitKey(0) #Wait for user input before closing the images displayed

# Recognition part
## Retrieve the bounding box where the plate's been detected
cv2.destroyAllWindows()
mask = np.zeros((original_img_height, original_img_width), dtype=np.uint8)
points = np.array(NumberPlateCnt)
cv2.fillPoly(mask, np.int32(points), (255))

res = cv2.bitwise_and(original_img, original_img, mask = mask)
#cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
cv2.polylines(image, np.int32(points), True, (122, 122, 122))

#print("Puntos: {}".format(points))
rect = cv2.boundingRect(np.int32(points)) # returns (x,y,w,h) of the rect
plate_x = int(rect[0]*width_ratio)
plate_w = int(rect[2]*width_ratio)
plate_y = int(rect[1]*height_ratio)
plate_h = int(rect[3]*height_ratio)
print("Caja. {}".format(rect))
# TODO: 
cropped = original_img[plate_y:plate_y+plate_h, plate_x:plate_x+plate_w]
# cropped = original_img[int(rect[1]*height_ratio): int((rect[1] + rect[3])*height_ratio), 
#                     int(rect[0]*width_ratio): int((rect[0] + rect[2])*width_ratio)]

cv2.imshow("orig" , original_img)
#cv2.imshow("same size" , res)

## Pass the plate through tesseract to perform recognition
#cropped = imutils.resize(cropped, width=320)
#cropped = cv2.resize(cropped, (320, 320))
cv2.imshow("cropped" , cropped)

print("Without preprocessing: ")
img = Image.fromarray(cropped)
print("TesserOCR: {}".format(tesserocr.image_to_text(img)))
print("Pytesseract: {}".format(pytesseract.image_to_string(cropped)))
print("With preprocessing: ")
cropped = original_img[int(plate_y-0.005*original_img_height):int(plate_y+plate_h+0.005*image_height), 
                    int(plate_x-0.005*original_img_width):int(plate_x+plate_w+0.005*original_img_width)]
crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
#image = cv2.resize(image, (640, -1), interpolation=cv2.INTER_CUBIC)
#image = cv2.resize(image, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC) # INTER_AREA to decrease
crop = cv2.bilateralFilter(crop, 11, 17, 17)
crop = cv2.threshold(crop, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

img = Image.fromarray(cropped)
print("TesserOCR: {}".format(tesserocr.image_to_text(img)))
print("Pytesseract: {}".format(pytesseract.image_to_string(crop)))

cv2.imshow("cropped_processed" , crop)
cv2.waitKey(0) #Wait for user input before closing the images displayed
