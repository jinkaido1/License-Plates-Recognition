# USAGE
# python text_detection.py --image images/lebron_james.jpg --east frozen_east_text_detection.pb

# import the necessary packages
import numpy as np
import argparse
import time
import cv2

from decode import decode
from draw import drawPolygons, drawBoxes
import utils

from nms import nms


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
    help="path to input image")
ap.add_argument("-east", "--east", type=str,
    help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
    help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
    help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320,
    help="resized image height (should be multiple of 32)")
args = vars(ap.parse_args())

# load the input image and grab the image dimensions
image = cv2.imread(args["image"])
orig = image.copy()
(origHeight, origWidth) = image.shape[:2]

# set the new width and height and then determine the ratio in change
# for both the width and height
(newW, newH) = (args["width"], args["height"])
ratioWidth = origWidth / float(newW)
ratioHeight = origHeight / float(newH)

# resize the image and grab the new image dimensions
image = cv2.resize(image, (newW, newH))
(imageHeight, imageWidth) = image.shape[:2]
cv2.imshow('hhh', image)

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

# construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (imageWidth, imageHeight), (123.68, 116.78, 103.94), swapRB=True, crop=False)

start = time.time()
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
end = time.time()

# show timing information on text prediction
print("[INFO] text detection took {:.6f} seconds".format(end - start))


# NMS on the the unrotated rects
confidenceThreshold = args['min_confidence']
nmsThreshold = 0.4

# decode the blob info
(rects, confidences, baggage) = decode(scores, geometry, confidenceThreshold)

offsets = []
thetas = []
for b in baggage:
    offsets.append(b['offset'])
    thetas.append(b['angle'])

##########################################################

functions = [nms.felzenszwalb.nms, nms.fast.nms, nms.malisiewicz.nms]

print("[INFO] Running nms.boxes . . .")

for i, function in enumerate(functions):

    start = time.time()
    indicies = nms.boxes(rects, confidences, nms_function=function, confidence_threshold=confidenceThreshold,
                             nsm_threshold=nmsThreshold)
    end = time.time()

    indicies = np.array(indicies).reshape(-1)

    drawrects = np.array(rects)[indicies]

    name = function.__module__.split('.')[-1].title()
    print("[INFO] {} NMS took {:.6f} seconds and found {} boxes".format(name, end - start, len(drawrects)))

    drawOn = orig.copy()
    drawBoxes(drawOn, drawrects, ratioWidth, ratioHeight, (0, 255, 0), 2)

    ############## Lo mío ################
    print(drawrects, indicies, ratioHeight, ratioWidth)
    # drawrects contiene las coordenadas de las cajas de texto detectadas en formato (x, y, w, h)
    # Usarlas para recortar la imagen y pasárselo a tesseract
    y = [int(drawrects[0][1]*ratioHeight), int((drawrects[0][1]+drawrects[0][3])*ratioHeight)]
    x = [int(drawrects[0][0]*ratioWidth), int((drawrects[0][0]+drawrects[0][2])*ratioWidth)]
    print(x, y)
    crop = orig[y[0]:y[1], x[0]:x[1]]
    cv2.imshow('Prueba', crop)
    cv2.waitKey(0)
    #######################################

    title = "nms.boxes {}".format(name)
    cv2.imshow(title, drawOn)
    cv2.moveWindow(title, 150+i*300, 150)

cv2.waitKey(0)


# convert rects to polys
polygons = utils.rects2polys(rects, thetas, offsets, ratioWidth, ratioHeight)

print("[INFO] Running nms.polygons . . .")

for i, function in enumerate(functions):

    start = time.time()
    indicies = nms.polygons(polygons, confidences, nms_function=function, confidence_threshold=confidenceThreshold,
                             nsm_threshold=nmsThreshold)
    end = time.time()

    indicies = np.array(indicies).reshape(-1)

    drawpolys = np.array(polygons)[indicies]

    name = function.__module__.split('.')[-1].title()

    print("[INFO] {} NMS took {:.6f} seconds and found {} boxes".format(name, end - start, len(drawpolys)))

    drawOn = orig.copy()
    drawPolygons(drawOn, drawpolys, ratioWidth, ratioHeight, (0, 255, 0), 2)

    ############## Lo mío ################
    print(polygons[32], indicies, ratioHeight, ratioWidth, offsets[32], thetas[32])


    mask = np.zeros((origHeight, origWidth), dtype=np.uint8)
    points = np.array([polygons[32]])
    print(points, type(points))
    cv2.fillPoly(mask, np.int32(points), (255))

    res = cv2.bitwise_and(orig,orig,mask = mask)
    #cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
    cv2.polylines(res, np.int32(points), True, (122, 122, 122))

    rect = cv2.boundingRect(np.int32(points)) # returns (x,y,w,h) of the rect
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    print(cropped)

    #cv2.imshow("cropped" , cropped )
    cv2.imshow("same size" , res)

    # print(points.max)
    # M = cv2.getRotationMatrix2D((int(offsets[32][0]),int(offsets[32][1])),10,1)
    # dst = cv2.warpAffine(res,M,(int(offsets[32][0]),int(offsets[32][1])))
    M = cv2.getRotationMatrix2D((int(100),int(100)),-np.rad2deg(thetas[32]),1)
    dst = cv2.warpAffine(res,M,(int(100),int(100)))
    cv2.imshow("Test" , dst)











    # # drawrects contiene las coordenadas de las cajas de texto detectadas en formato (x, y, w, h)
    # # Usarlas para recortar la imagen y pasárselo a tesseract
    # y = [int(drawpolys[0][1]*ratioHeight), int((drawpolys[0][1]+drawpolys[0][3])*ratioHeight)]
    # x = [int(drawpolys[0][0]*ratioWidth), int((drawpolys[0][0]+drawpolys[0][2])*ratioWidth)]
    # print(x, y)
    # crop = orig[y[0]:y[1], x[0]:x[1]]
    # cv2.imshow('Prueba2', crop)
    # cv2.waitKey(0)
    # #######################################

    title = "nms.polygons {}".format(name)
    cv2.imshow(title,drawOn)
    cv2.moveWindow(title, 150+i*300, 150)

cv2.waitKey(0)