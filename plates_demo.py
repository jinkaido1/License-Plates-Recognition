import numpy as np
import cv2
import  imutils
import tesserocr
from PIL import Image
import pytesseract
from pathlib import Path

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
            # rect = cv2.minAreaRect(approx)
            # angle = rect[2]
            # box = cv2.boxPoints(rect)
            # box = np.int0(box)
            # cv2.drawContours(image,[box],0,(0,0,255),2)
            break

    return plate_contour

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
        plate_contour = scale_contour(contour, 1.0)
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
    
    # Obtain a rotated bounding box for the plate
    rotated_rect = cv2.minAreaRect(plate_contour) # Returns a Box2D like (center (x,y), (w,h), angle)
    box = cv2.boxPoints(rotated_rect)
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, (139,109,255), 2)
    cv2.imshow("Plates detected" , image)
    cv2.waitKey(0)

    # Get the plate cropped considering the angle 
    angle = rotated_rect[2] if rotated_rect[2] < 45 and rotated_rect[2] > -45 else 0
    print("{}, {}".format(rotated_rect[2], angle))
    rotated = imutils.rotate_bound(cropped, -angle)
    cv2.destroyAllWindows()
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
    dir = Path("images/").glob("**/*.jpg") # Modifiable. Takes all .jpg files in the given directory

    for image_path in dir:
        # Applies a Face - Detect for every image in the above directory
        image = cv2.imread(str(image_path))
        cv2.imshow('Original Image', image)
        cv2.waitKey(0)

        plate_contour = plate_detection(image)
        if plate_contour is None:
            print("No license plates detected")
            continue
        img = image.copy()
        try:
            plate = crop_contour(plate_contour, image, img)
        except:
            print("Plate could not be processed")
            continue
        cv2.destroyAllWindows()
        plate_recognition(plate)
    exit()


    # original_img = cv2.imread('images/renault_back.jpg')
    # original_img_height, original_img_width = original_img.shape[:2]

    # # Resize the image - change width to 500
    # image = imutils.resize(original_img, width=568)

    # # Store the ratios after the resize
    # original_img_height, original_img_width = original_img.shape[:2]
    # image_height, image_width = image.shape[:2]
    # height_ratio = original_img_height / image_height
    # width_ratio = original_img_width / image_width

    # plate_contour = plate_detection(image)
    # print(plate_contour)
    # if plate_contour is None:
    #     print("No license plates detected")
    #     exit()
    # image = original_img.copy()
    # plate = crop_contour(plate_contour, original_img, image, width_ratio, height_ratio)
    # plate_recognition(plate)


if __name__ == '__main__':
    main()
