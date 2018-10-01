import numpy as np
import argparse
import cv2
import  imutils
import tesserocr
from PIL import Image
import pytesseract
from pathlib import Path
import time

def argss():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--folder", type=str, default='images/', 
        help="Directory where the images are located")
    ap.add_argument("-sf", "--scaling_factor", type=float, default=0.98, 
        help="Scaling factor to apply to the detected plates in the image")
    args = vars(ap.parse_args())

    return args

def scale_contour(contour, scale_factor):
    """Scales a given contour keeping its coordinates

    Args:
        contour (nparray): OpenCV contour to scale.
        scale_factor (float): Scaling factor to apply to the contour.

    Returns:
        contour (nparray): Original contour scaled.

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
    ## Loop over our contours to find the best possible approximate contour of number plate
    for c in cnts:
        # Calculate the perimeter of a closed contour
        perimeter = cv2.arcLength(c, True)
        # Create new contour with a simplified shape with less vertices. Closed
        approx = cv2.approxPolyDP(c, 0.02*perimeter, True)
        # Check if the new approximate contour has 4 vertices
        if len(approx) == 4:
            plate_contour = approx
            break

    return plate_contour

def crop_contour(contour, original_img, image, scaling_factor = 9.8, 
                 width_ratio=1, height_ratio=1, visualize=True):
    """Given a detected contour, crop the RoI

    Args:
        contour (nparray): Contour where a possible plate might've been detected.
        original_img (nparray): Image without any modifications.
        image (nparray): Image with several resizing modifications.
        width_ratio (float): Self-explanatory.
        height_ratio (float): Self-explanatory.
        visualize (bool): Processing images displayed or not (default True).
    
    Returns:
        crop (nparray): Crop from the image where the plate could be located.

    """
    try:
        plate_contour = scale_contour(contour, scaling_factor)
    except:
        print("Unable to perform scaling")
        plate_contour = contour
    bounding_rect = cv2.boundingRect(plate_contour) # Returns (top left (x,y), (width,height))
    # Convert the plate coordinates to the ones in the original image
    plate_x = int(bounding_rect[0]*width_ratio)
    plate_y = int(bounding_rect[1]*height_ratio)
    plate_w = int(bounding_rect[2]*width_ratio)
    plate_h = int(bounding_rect[3]*height_ratio)
    # Get the plate form the original image
    cropped = original_img[plate_y:plate_y+plate_h, plate_x:plate_x+plate_w]
    
    # Obtain a rotated bounding box for the plate
    rotated_rect = cv2.minAreaRect(plate_contour) # Returns a Box2D like (center (x,y), (w,h), angle)
    box = cv2.boxPoints(rotated_rect)
    box = np.int0(box)
    if visualize:
        cv2.drawContours(image, [box], 0, (139,109,255), 2)
        cv2.imshow("Plates detected" , image)
        cv2.waitKey(3000)

    # Get the plate cropped considering the angle 
    angle = rotated_rect[2] if rotated_rect[2] < 45 and rotated_rect[2] > -45 else 0
    rotated = imutils.rotate_bound(cropped, -angle)

    return rotated

def plate_recognition(plate, visualize=True):
    """Performs OCR with Tesseract to a plate image.

    Args:
        plate (nparray): License plate cropped from an image (OpenCV).
        visualize (bool): Processing images displayed or not (default True).
    
    Returns:
        plate_text (str): Text that Tesseract's been able to detect.

    """
    # Process the current plate crop to make it easier to recognize
    plate_processed = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    plate_processed = cv2.bilateralFilter(plate_processed, 11, 17, 17)
    plate_processed = cv2.threshold(plate_processed, 177, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    if visualize:
        cv2.imshow('Plate crop', plate)
        cv2.moveWindow('Plate crop', 0, 0)
        cv2.imshow('Processed plate', plate_processed)
        cv2.moveWindow('Processed plate', 0, plate.shape[1])
        cv2.waitKey(1000)

    # Pass the plate crop through an OCR system
    print("Without preprocessing: ")
    print("Pytesseract: {}".format(pytesseract.image_to_string(plate)))
    img = Image.fromarray(plate)
    print("OCR: {}".format(tesserocr.image_to_text(img)))

    print("With preprocessing: ")
    plate_text = pytesseract.image_to_string(plate_processed)
    print("Pytesseract: {}".format(plate_text))
    img = Image.fromarray(plate)
    print("OCR: {}".format(tesserocr.image_to_text(img)))
    #cv2.waitKey(0)
    plate_text = ''.join(ch for ch in plate_text if ch.isalnum())
    print('Recognized text into the license plate: {}'.format(plate_text))
    cv2.destroyAllWindows()

    return plate_text


def main():
    args = argss()
    dir = Path("images/").glob("**/*.jpg") # Takes all .jpg files in the given directory

    for image_path in dir:
        # Applies the recognition pipeline to every image in the folder
        ## Open an image
        try:
            image = cv2.imread(str(image_path))
        except:
            print("Unable to find {}".format(str(image_path)))
            continue

        start_time = time.time()

        ## Find the contour of the plate
        plate_contour = plate_detection(image)
        if plate_contour is None:
            print("No license plates detected")
            continue
        img = image.copy()
        ## Crop only the plate portion of the original image
        try:
            plate = crop_contour(plate_contour, image, img, 
                scaling_factor=args['scaling_factor'], visualize=False)
        except:
            print("Plate could not be processed")
            continue
        cv2.destroyAllWindows()
        ## Recognize the text in the license plate
        plate_text = plate_recognition(plate, visualize=False)
        elapsed_time = time.time() - start_time
        print(elapsed_time)

        exit()

    for image_path in dir:
        # Applies the recognition pipeline to every image in the folder
        ## Open an image
        try:
            image = cv2.imread(str(image_path))
        except:
            print("Unable to find {}".format(str(image_path)))
            continue

        ## Find the contour of the plate
        plate_contour = plate_detection(image)
        if plate_contour is None:
            print("No license plates detected")
            continue
        img = image.copy()
        ## Crop only the plate portion of the original image
        try:
            plate = crop_contour(plate_contour, image, img, scaling_factor=args['scaling_factor'])
        except:
            print("Plate could not be processed")
            continue
        cv2.destroyAllWindows()
        ## Recognize the text in the license plate
        plate_text = plate_recognition(plate)
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
