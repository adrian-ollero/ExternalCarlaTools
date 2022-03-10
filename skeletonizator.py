from concurrent.futures import process
from skimage.morphology import skeletonize
from skimage.transform import rotate
from skimage import util, data, color
import sknw

import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread, imshow, imsave
from skimage.filters import threshold_mean
import cv2
from sknw.sknw import neighbors

import timeit

#            Configuration              #
# name = "forestMap/river/ForestMap_Tile" # Relative to input/
name = "forestMap/lake/ForestMap_Tile" # Relative to input/
# name = "4_1LakeLayer"
# processingType = "river" # 'river' or 'lake'
processingType = "lake" # 'river' or 'lake'
showPlots = False
savePlotToFile = True
rotation = -90

multiTile = True	# Whether the input is a single file or multiple files
rowsNum = 20         # Number of rows to be processed
colsNum = 20         # Number of culumns to be processed
# ##################################### #

inputName = "input/" + name + '.png'
outputName = "output/" + name + ".out"

def processGraphToOutputText(graph, outputFileName):
    output_file = open(outputFileName, "w")

    for (s, e) in graph.edges():
        edgePoints = graph[s][e]['pts']
        np.savetxt(output_file, edgePoints, fmt='%d', header="_")
    
    output_file.write("# _")
    output_file.close()
    return True

def getPlotName(inputFileName):
    inputArray = inputFileName.split(".")
    name = inputArray[0] + "_plot" + inputArray[1]
    return name


def river(inputFileName,outputFileName):
    print("Processing river...🏞️   --> (" + inputFileName + ")" )
    img = imread(inputFileName,0)
    img = rotate(img, rotation)

    threshold = threshold_mean(img)
    img_thresholded = img > threshold

    # img_in = util.invert(img_thresholded)

    ske = skeletonize(~img_thresholded).astype(np.uint8)

    # build graph from skeleton
    graph = sknw.build_sknw(ske)

    # draw image
    plt.imshow(img, cmap='gray')

    # draw edges by pts
    for (s,e) in graph.edges():
        ps = graph[s][e]['pts']
        plt.plot(ps[:,1], ps[:,0], 'green')
        
    # draw node by o
    nodes = graph.nodes()
    ps = np.array([nodes[i]['o'] for i in nodes])
    plt.plot(ps[:,1], ps[:,0], 'r.')
    

    processGraphToOutputText(graph, outputFileName)

    if savePlotToFile:
        plt.savefig(getPlotName(outputFileName))

    # title and show
    if showPlots:
        plt.title('Build Graph')
        plt.show()

    plt.clf()


def lake(inputFileName, outputFileName):
    print("Processing lake... 🌅   --> (" + inputFileName + ")")

    # Rotate Image with scikit image
    prev_img = imread(inputFileName,0)
    prev_img = rotate(prev_img, rotation)
    imsave('temp/auxrot.png', prev_img)
    # imsave('temp/auxrot.png', (color.convert_colorspace(prev_img, 'HSV', 'RGB')*255).astype(np.uint8))

    img = cv2.imread('temp/auxrot.png',0)
    # img_rot = cv2.rotate(img, 90)
    ret,img_thresholded = cv2.threshold(img,127,255,0)
    contours,hierarchy = cv2.findContours(img_thresholded, 1, 2)

    if len(contours) == 0:
        print("No lakes in tile " + inputFileName)
        return True

    if showPlots:
        cv2.imshow('Image', img_thresholded)
        cv2.waitKey(0)

    drawing = np.zeros((img_thresholded.shape[0], img_thresholded.shape[1], 3), np.uint8)

    output_file = open(outputFileName, "w")

    for cnt in contours: 
        # hull = cv2.convexHull(cnt)
        rect = cv2.minAreaRect(cnt)

        if showPlots or savePlotToFile:
            # print(rect)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            color_contours = (0, 255, 0) # green - color for contours
            color_box = (255, 0, 0) # blue - color for convex hull
            # draw ith contour
            cv2.drawContours(drawing, contours, -1, color_contours, 1, 8, hierarchy)
            cv2.drawContours(drawing, [box], -1, color_box, 1, 8)

            

        center, size, angle = rect
        centerx, centery = center
        sizex, sizey = size

        output_file.write("# _L\n")
        lakedata = "{} {} {} {} {}\n".format(int(centerx), int(centery), int(sizex), int(sizey), int(angle))
        output_file.write(lakedata)

    if showPlots:
        cv2.imshow('Contours', drawing)
        cv2.waitKey(0)
    if savePlotToFile:
        oname = outputFileName + ".png"
        # oname = getPlotName(outputFileName)
        print(oname)
        cv2.imwrite(oname, drawing)


    output_file.write("# _L\n")
    output_file.close()     

    


def mainSingleTile(inputFileName, outputFileName):
    if(processingType == "river"):
        river(inputFileName, outputFileName)
        return True
    elif(processingType == "lake"):
        lake(inputFileName, outputFileName)
        return True
    else:
        print("Invalid processing type")
        return False

def mainMultiTiles(inputFileName, outputFileName):
    for i in range(0, rowsNum):
        for j in range(0, colsNum):
            inputArray = inputFileName.split(".")
            outputArray = outputFileName.split(".")
            newInputName = inputArray[0] + "_" + str(i) + "_" + str(j) + "." + inputArray[1]
            newOutputName = outputArray[0] + "_" + str(i) + "_" + str(j) + "." + outputArray[1]
            result = mainSingleTile(newInputName, newOutputName)
            if result == False:
                return False

    return True

def main():
    if multiTile:
        return mainMultiTiles(inputName, outputName)
    else:
        return mainSingleTile(inputName, outputName)

    


if __name__ == "__main__":
    startTime = timeit.default_timer()
    result = main()
    stopTime = timeit.default_timer()
    duration = stopTime - startTime
    if result:
        print("Success! 🤩  ")
    else:
        print("Ups... Failed... 😵  ")
    
    print("⌛ Time: ", duration, "s")