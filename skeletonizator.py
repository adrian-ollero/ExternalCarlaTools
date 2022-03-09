from concurrent.futures import process
from skimage.morphology import skeletonize
from skimage.transform import rotate
from skimage import util, data
import sknw

import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread, imshow, imsave
from skimage.filters import threshold_mean
import cv2
from sknw.sknw import neighbors

#            Configuration              #
name = "testi"
# name = "4_1LakeLayer"
processingType = "river" # 'river' or 'lake'
# processingType = "lake" # 'river' or 'lake'
showPlots = True
rotation = -90
# ##################################### #

inputName = name + '.png'
outputName = name + ".out"

def processGraphToOutputText(graph):
    output_file = open(outputName, "w")

    for (s, e) in graph.edges():
        edgePoints = graph[s][e]['pts']
        np.savetxt(output_file, edgePoints, fmt='%d', header="_")
    
    output_file.write("# _")
    output_file.close()
    return True


def river():
    print("Processing river...")
    img = imread(inputName,0)
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

    processGraphToOutputText(graph)

    # title and show
    if showPlots:
        plt.title('Build Graph')
        plt.show()


def lake():
    print("Processing lake...")

    # Rotate Image with scikit image
    prev_img = imread(inputName,0)
    prev_img = rotate(prev_img, rotation)
    imsave('auxrot.png', prev_img)

    img = cv2.imread('auxrot.png',0)
    # img_rot = cv2.rotate(img, 90)
    ret,img_thresholded = cv2.threshold(img,127,255,0)
    contours,hierarchy = cv2.findContours(img_thresholded, 1, 2)

    if showPlots:
        cv2.imshow('Image', img_thresholded)
        cv2.waitKey(0)

    drawing = np.zeros((img_thresholded.shape[0], img_thresholded.shape[1], 3), np.uint8)

    output_file = open(outputName, "w")

    for cnt in contours: 
        # hull = cv2.convexHull(cnt)
        rect = cv2.minAreaRect(cnt)

        if showPlots:
            print(rect)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            color_contours = (0, 255, 0) # green - color for contours
            color = (255, 0, 0) # blue - color for convex hull
            # draw ith contour
            cv2.drawContours(drawing, contours, 0, color_contours, 1, 8, hierarchy)
            cv2.drawContours(drawing, [box], 0, color, 1, 8)

        center, size, angle = rect
        centerx, centery = center
        sizex, sizey = size

        output_file.write("# _L\n")
        lakedata = "{} {} {} {} {}\n".format(int(centerx), int(centery), int(sizex), int(sizey), int(angle))
        output_file.write(lakedata)


    output_file.write("# _L\n")
    output_file.close()     

    if showPlots:
        cv2.imshow('Contours', drawing)
        cv2.waitKey(0)




def main():
    if(processingType == "river"):
        river()
        return True
    elif(processingType == "lake"):
        lake()
        return True
    else:
        print("Invalid processing type")
        return False


if __name__ == "__main__":
    result = main()
    if result:
        print("Success! 🤩")
    else:
        print("Ups... Failed... 😵")