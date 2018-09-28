import numpy as np
import cv2
import  imutils
import tesserocr
from PIL import Image
import pytesseract

def scale_contour(contour, scale_factor):
    """Scales a given contour keeping its coordinates

    Args:
        contour (nparray): OpenCV contour to scale.
        scale_factor (float): Scaling factor to apply to the contour.

    Returns:
        contour (nparay): Original contour scaled.

    """
    # Extract the center of the contour
    x, y, w, h = cv2.boundingRect(contour)
    center = (int((x+w)/2), int((y+h)/2))
    # other way
    M = cv2.moments(contour)
    center = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
    # Substract the center from every point of the contour
    contour -= center
    # Scale the contour
    contour = contour*scale_factor
    # Add the center to the new contour
    contour += center

    return np.int32(contour)
    
def scale_contour2(contour, center, scale_factor):
    """

    """
    # Substract the center from every point of the contour
    contour -= center
    # Scale the contour
    contour = contour*scale_factor
    # Add the center to the new contour
    contour += center

    return np.int32(contour)

# Read the image file
original_img = cv2.imread('images/VW_front.jpg')
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
        print(approx)
        if len(approx) == 4:  # Select the contour with 4 corners
            NumberPlateCnt = approx #This is our approx Number Plate Contour
            # rect = cv2.minAreaRect(approx)
            # angle = rect[2]
            # box = cv2.boxPoints(rect)
            # box = np.int0(box)
            # cv2.drawContours(img,[box],0,(0,0,255),2)
            break

# Drawing the selected contour on the original image
#cv2.drawContours(image, [NumberPlateCnt], -1, (0,255,0), 3)
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
print("{}, {}, {}, {}".format(plate_x, plate_w, plate_y, plate_h))
print("Caja. {}".format(rect))
# TODO: 
cropped = original_img[plate_y:plate_y+plate_h, plate_x:plate_x+plate_w]
# cropped = original_img[int(rect[1]*height_ratio): int((rect[1] + rect[3])*height_ratio), 
#                     int(rect[0]*width_ratio): int((rect[0] + rect[2])*width_ratio)]

cv2.imshow("orig" , original_img)
rect = cv2.minAreaRect(approx)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(original_img,[box],0,(0,0,255),2)
cv2.imshow("orig" , image)
cv2.imshow("ccc" , cropped)
## Crop augmentation
# plate_x -= int(plate_x*(0.05*original_img_width))
# plate_w += int(plate_w*(0.05*original_img_width))
# plate_y -= int(plate_y*(0.05*original_img_height))
# plate_h += int(plate_h*(0.05*original_img_height))
# print("{}, {}, {}, {}".format(plate_x, plate_w, plate_y, plate_h))
new_contour = scale_contour(approx, 1.5)
#new_contour = scale_contour2(approx, (400, 284), 1.25)
rect = cv2.boundingRect(new_contour)
plate_x = int(rect[0]*width_ratio)
plate_w = int(rect[2]*width_ratio)
plate_y = int(rect[1]*height_ratio)
plate_h = int(rect[3]*height_ratio)
cropped = original_img[plate_y:plate_y+plate_h, plate_x:plate_x+plate_w]
rect = cv2.minAreaRect(new_contour)

box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(original_img,[box],0,(139,109,255),2)
cv2.imshow("orig" , original_img)
cv2.imshow("Scaling" , cropped)

plate_x -= int(plate_x*(0.05))
plate_w += int(plate_w*(0.25))
plate_y -= int(plate_y*(0.05))
plate_h += int(plate_h*(0.25))
print("{}, {}, {}, {}".format(plate_x, plate_w, plate_y, plate_h))
cropped = original_img[plate_y:plate_y+plate_h, plate_x:plate_x+plate_w]
cv2.imshow("NEW Crop" , cropped)
#cv2.imshow("same size" , res)

## Pass the plate through tesseract to perform recognition
#cropped = imutils.resize(cropped, width=320)
#cropped = cv2.resize(cropped, (320, 320))
rect = cv2.minAreaRect(approx)
angle = rect[2]
box = cv2.boxPoints(rect) 
box = np.int0(box)  
Xs = [i[0] for i in box]
Ys = [i[1] for i in box]
x1 = min(Xs)
x2 = max(Xs)
y1 = min(Ys)
y2 = max(Ys)

# center = ((x1+x2)/2,(y1+y2)/2)
# size = (x2-x1, y2-y1)
center = ((plate_x+plate_w)/2,(plate_y+plate_h)/2)
size = (plate_w, plate_h)

print(angle)
M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)
plate = cv2.getRectSubPix(cropped, size, center)
plate = cv2.warpAffine(plate, M, size)


cv2.imshow("cropped" , cropped)
cv2.imshow('Rotated', plate)
rotated = imutils.rotate_bound(cropped, -angle)
cv2.imshow('Rotated2', rotated)

print("Without preprocessing: ")
img = Image.fromarray(rotated)
print("TesserOCR: {}".format(tesserocr.image_to_text(img)))
print("Pytesseract: {}".format(pytesseract.image_to_string(rotated)))
print("With preprocessing: ")
cropped = original_img[int(plate_y-0.005*original_img_height):int(plate_y+plate_h+0.005*image_height), 
                    int(plate_x-0.005*original_img_width):int(plate_x+plate_w+0.005*original_img_width)]
crop = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
#image = cv2.resize(image, (640, -1), interpolation=cv2.INTER_CUBIC)
#image = cv2.resize(image, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC) # INTER_AREA to decrease
crop = cv2.bilateralFilter(crop, 11, 17, 17)
crop = cv2.threshold(crop, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

img = Image.fromarray(rotated)
print("TesserOCR: {}".format(tesserocr.image_to_text(img)))
print("Pytesseract: {}".format(pytesseract.image_to_string(crop)))

cv2.imshow("cropped_processed" , crop)
cv2.waitKey(0) #Wait for user input before closing the images displayed
