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
    # Extract the centroid of the contour
    M = cv2.moments(contour)
    center = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
    # Substract the center from every point of the contour
    contour -= center
    # Scale the contour
    contour = contour*scale_factor
    # Add the center to the new contour
    contour += center

    return np.int32(contour)

def plate_detection(image):
    """Detects the region where a license plate could be.

    Args:
        image (nparray): Image to procces to find the best contour candidate.

    Returns:
        plate_contour (nparray): Contour where a plate could be located.

    """
    # RGB to Gray scale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Noise removal with iterative bilateral filter(removes noise while preserving edges)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Find Edges of the grayscale image
    edged = cv2.Canny(gray, 170, 200)

    # Find contours based on Edges
    im, cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    ## Sort contours based on their area
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30] 
    plate_contour = None

    # Loop over our contours to find the best possible approximate contour of number plate
    for c in cnts:
        # Calculate the perimeter of a closed contour
        perimeter = cv2.arcLength(c, True)
        # Create new contour with a simplified shape with less vertices. Closed
        approx = cv2.approxPolyDP(c, 0.02*perimeter, True)
        # Check if the new approximate contour has 4 vertices
        if len(approx) == 4: ## TODO: and cv2.minArea > threshold
            plate_contour = approx
            rect = cv2.minAreaRect(approx)
            angle = rect[2]
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(image,[box],0,(0,0,255),2)
            #break

    return plate_contour
    #crop_contour(contour, image, width_ratio, height_ratio)
    """
    plate_contour = scale_contour(plate_contour, 1.7)
    bounding_rect = cv2.boundingRect(plate_contour) # Returns (top left (x,y), (width,height))
    # Conver the plate coordinates to the ones in the original image
    plate_x = int(bounding_rect[0]*width_ratio)
    plate_w = int(bounding_rect[2]*width_ratio)
    plate_y = int(bounding_rect[1]*height_ratio)
    plate_h = int(bounding_rect[3]*height_ratio)
    # Get the plate form the original image
    cropped = original_img[plate_y:plate_y+plate_h, plate_x:plate_x+plate_w]
    
    # Obtain a rotated bounding box for the plate
    rotated_rect = cv2.minAreaRect(plate_contour) # Returns a Box2D like (center (x,y), (w,h), angle)
    box = cv2.boxPoints(rotated_rect)
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, (139,109,255), 2)
    cv2.imshow("Plates detected" , image)
    #cv2.imshow("Plate crop" , cropped)

    # Get the plate cropped considering the angle 
    angle = rotated_rect[2]
    rotated = imutils.rotate_bound(cropped, -angle)
    cv2.imshow('Plate crop', rotated)
    cv2.waitKey(0)
    
    return rotated
    """

def crop_contour(contour, original_img, image, width_ratio=1, height_ratio=1):
    """Given a detected contour, crop the RoI

    Args:
        contour (nparray): Contour where a possible plate might've been detected.
        original_img (nparray): Image without any modifications.
        image (nparray): Image with several resizing modifications.
        width_ratio (float): Self-explanatory.
        height_ratio (float): Self-explanatory.
    
    Returns:
        crop (nparray): Crop from the image where the plate could be located.

    """
    try:
        plate_contour = scale_contour(contour, 1.6)
    except:
        print("Unable to perform scaling")
        plate_contour = contour
    bounding_rect = cv2.boundingRect(plate_contour) # Returns (top left (x,y), (width,height))
    # Convert the plate coordinates to the ones in the original image
    plate_x = int(bounding_rect[0]*width_ratio)
    plate_y = int(bounding_rect[1]*height_ratio)
    plate_w = int(bounding_rect[2]*width_ratio)
    plate_h = int(bounding_rect[3]*height_ratio)
    print("{}, {}, {}, {}".format(plate_x, plate_y, plate_w, plate_h))
    # Get the plate form the original image
    cropped = original_img[plate_y:plate_y+plate_h, plate_x:plate_x+plate_w]
    cv2.imshow("ccc" , cropped)
    cv2.waitKey(0)
    
    # Obtain a rotated bounding box for the plate
    rotated_rect = cv2.minAreaRect(plate_contour) # Returns a Box2D like (center (x,y), (w,h), angle)
    box = cv2.boxPoints(rotated_rect)
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, (139,109,255), 2)
    cv2.imshow("Plates detected" , image)
    cv2.waitKey(0)
    #cv2.imshow("Plate crop" , cropped)

    # Get the plate cropped considering the angle 
    angle = rotated_rect[2] if rotated_rect[2] < 45 and rotated_rect[2] > -45 else 0
    print("{}, {}".format(rotated_rect[2], angle))
    rotated = imutils.rotate_bound(cropped, -angle)
    cv2.imshow('Plate crop', rotated)
    cv2.waitKey(0)

    return rotated

def plate_recognition(plate):
    """Performs OCR with Tesseract to a plate image.

    Args:
        plate (nparray): License plate cropped from an image (OpenCV).
    
    Returns:
        plate_text (str): Text that Tesseract's been able to detect.

    """
    cv2.destroyAllWindows()
    print("Without preprocessing: ")
    cv2.imshow('Plate', plate)
    print("Pytesseract: {}".format(pytesseract.image_to_string(plate)))
    img = Image.fromarray(plate)
    print("OCR: {}".format(tesserocr.image_to_text(img)))

    print("With preprocessing: ")
    image = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    image = cv2.bilateralFilter(image, 11, 17, 17)
    image = cv2.threshold(image, 177, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow('Processed Plate', image)
    print("Pytesseract: {}".format(pytesseract.image_to_string(image)))
    img = Image.fromarray(image)
    print("OCR: {}".format(tesserocr.image_to_text(img)))
    cv2.waitKey(0)
    

def main():
    original_img = cv2.imread('images/89.jpg')
    original_img_height, original_img_width = original_img.shape[:2]

    # Resize the image - change width to 500
    image = imutils.resize(original_img, width=568)

    # Store the ratios after the resize
    original_img_height, original_img_width = original_img.shape[:2]
    image_height, image_width = image.shape[:2]
    height_ratio = original_img_height / image_height
    width_ratio = original_img_width / image_width

    plate_contour = plate_detection(image)
    print(plate_contour)
    if plate_contour is None:
        print("No license plates detected")
        exit()
    plate = crop_contour(plate_contour, original_img, image, width_ratio, height_ratio)
    plate_recognition(plate)


if __name__ == '__main__':
    main()





"""
# Read the image file
original_img = cv2.imread('images/renault_back.jpg')
cv2.imshow('ooo', original_img)
original_img_height, original_img_width = original_img.shape[:2]
#original_img = image

# Resize the image - change width to 500
#image = imutils.resize(original_img, width=568)
image = original_img

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

###### Recognition
plate_recognition(rotated)

# print("Without preprocessing: ")
# img = Image.fromarray(rotated)
# print("TesserOCR: {}".format(tesserocr.image_to_text(img)))
# print("Pytesseract: {}".format(pytesseract.image_to_string(rotated)))
# print("With preprocessing: ")
# cropped = original_img[int(plate_y-0.005*original_img_height):int(plate_y+plate_h+0.005*image_height), 
#                     int(plate_x-0.005*original_img_width):int(plate_x+plate_w+0.005*original_img_width)]
# crop = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
# #image = cv2.resize(image, (640, -1), interpolation=cv2.INTER_CUBIC)
# #image = cv2.resize(image, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC) # INTER_AREA to decrease
# crop = cv2.bilateralFilter(crop, 11, 17, 17)
# crop = cv2.threshold(crop, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# img = Image.fromarray(rotated)
# print("TesserOCR: {}".format(tesserocr.image_to_text(img)))
# print("Pytesseract: {}".format(pytesseract.image_to_string(crop)))

# cv2.imshow("cropped_processed" , crop)
# cv2.waitKey(0) #Wait for user input before closing the images displayed
"""