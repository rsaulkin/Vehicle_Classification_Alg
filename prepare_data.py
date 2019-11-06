# import all pkgs
from PIL import Image
import pandas as pd
import os
import math


# Class Point
class Point:
    def __init__(self):
        self.x = 0
        self.y = 0

    def __int__(self, other):
        self.x = other.x
        self.y = other.y

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def prt(self):
        print("X=" + str(self.x) + ", Y=" + str(self.y))
        return

    def __str__(self):
        theStr = "X=" + str(self.x) + ", Y=" + str(self.y)
        return theStr

# Class Bounding Box
class BoundingBox:
    upperLeft = Point(0, 0)
    bottomRight = Point(0, 0)

    def __init__(self, ul, br):
        self.upperLeft.x = ul.x
        self.upperLeft.y = ul.y
        self.bottomRight.x = br.x
        self.bottomRight.y = br.y


def prepareData(inPath, outPath, csvFilePath):

    # read CSV file
    df = pd.read_csv(csvFilePath)

    numRows = df.shape[0]
    print('Num of rows in csv file = ' + str(numRows))

    # define variables
    prevImageId = 0

    fullOutPath = ""
    p1 = Point(0, 0)
    p2 = Point(0, 0)
    p3 = Point(0, 0)
    p4 = Point(0, 0)
    upperLeft = Point(0, 0)
    bottomRight = Point(0, 0)
    image = None
    allTagsCoordsList = dict()

    # each row = 1 vehicle in the image - read the coordinates of it bounding box and crop the image accordingly
    for lineNum in range(0, numRows):
        curImageId = df.loc[lineNum, 'image_id']

        # read the coordinates of the bounding box of the vehicle
        p1.x = df.iloc[lineNum, 2]
        p1.y = df.iloc[lineNum, 3]
        p2.x = df.iloc[lineNum, 4]
        p2.y = df.iloc[lineNum, 5]
        p3.x = df.iloc[lineNum, 6]
        p3.y = df.iloc[lineNum, 7]
        p4.x = df.iloc[lineNum, 8]
        p4.y = df.iloc[lineNum, 9]
        print("Line " + str(lineNum) + ": P1= " + str(p1) + " P2= " + str(p2) + " P3= " + str(p3) + " P4= " + str(p4))

        # set upper left point
        upperLeft.x = p1.x
        if p2.x < upperLeft.x:
            upperLeft.x = p2.x
        if p3.x < upperLeft.x:
            upperLeft.x = p3.x
        if p4.x < upperLeft.x:
            upperLeft.x = p4.x
        if upperLeft.x < 0:
            upperLeft.x = 0

        upperLeft.y = p1.y
        if p2.y > upperLeft.y:
            upperLeft.y = p2.y
        if p3.y > upperLeft.y:
            upperLeft.y = p3.y
        if p4.y > upperLeft.y:
            upperLeft.y = p4.y
        if upperLeft.y < 0:
            upperLeft.y = 0

        # set bottom right point
        bottomRight.x = p1.x
        if p2.x > bottomRight.x:
            bottomRight.x = p2.x
        if p3.x > bottomRight.x:
            bottomRight.x = p3.x
        if p4.x > bottomRight.x:
            bottomRight.x = p4.x
        if bottomRight.x < 0:
            bottomRight.x = 0

        bottomRight.y = p1.y
        if p2.y < bottomRight.y:
            bottomRight.y = p2.y
        if p3.y < bottomRight.y:
            bottomRight.y = p3.y
        if p4.y < bottomRight.y:
            bottomRight.y = p4.y
        if bottomRight.y < 0:
            bottomRight.y = 0

        print("Line " + str(lineNum) + ": UpperLeft= " + str(upperLeft) + " BottomRight= " + str(bottomRight))

        # set tag id of current vehicle image
        tagId = df.loc[lineNum, 'tag_id']
        curFileName = str(curImageId) + "_" + str(tagId)
        print("Line " + str(lineNum) + ": " + curFileName)

        # add the coords to the tags list
        allTagsCoordsList[str(tagId)] = (Point(upperLeft.x, upperLeft.y), Point(bottomRight.x, bottomRight.y))

        # open current image
        if curImageId != prevImageId:

            # close prev image
            if prevImageId != 0:
                image.close()

            # load the image

            # check if the file is jpg
            fullImgPath = inPath + str(curImageId) + ".jpg"
            if os.path.isfile(fullImgPath):
                print("Line " + str(lineNum) + ": Full image path = " + fullImgPath)
                image = Image.open(fullImgPath)
            else:
                # check if the file is tiff
                fullImgPath = inPath + str(curImageId) + ".tiff"
                if os.path.isfile(fullImgPath):
                    print("Line " + str(lineNum) + ": Full image path = " + fullImgPath)
                    image = Image.open(fullImgPath)
                else:
                    # chck if the file is tif
                    fullImgPath = inPath + str(curImageId) + ".tif"
                    if os.path.isfile(fullImgPath):
                        print("Line " + str(lineNum) + ": Full image path = " + fullImgPath)
                        image = Image.open(fullImgPath)
                    else:
                        print("File not found: " + fullImgPath)
                        continue

            # change the prev img name
            prevImageId = curImageId

        # create the cropped image of the current vehicle in the image
        croppedImg = image.crop((int(upperLeft.x), int(bottomRight.y), int(bottomRight.x), int(upperLeft.y)))

        # get the classification and tag id
        try:
            typeClass = df.loc[lineNum, 'general_class']
            print("Line " + str(lineNum) + ": " + "Type: " + typeClass)
        except:
            typeClass = "no type"
            print("Line " + str(lineNum) + ": " + "No Type")

        # save the cropped img
        if typeClass == "small vehicle":
            fullOutPath = outPath + "small/" + curFileName + ".jpg"
        elif typeClass == "large vehicle":
            fullOutPath = outPath + "large/" + curFileName + ".jpg"
        else:
            fullOutPath = outPath + "out/" + curFileName + ".jpg"
        print("Line " + str(lineNum) + ": Full image path = " + fullOutPath)


        croppedImg.save(fullOutPath)
        croppedImg.close()

    return allTagsCoordsList

# Prepare the data
#csvTestFilePath = '/home/revital/Documents/Project/DS/test/test1.csv'
#inTestPath = '/home/revital/Documents/Project/DS/test/'
#outTestPath = '/home/revital/Documents/Project/DS/test/'

#prepareData(inTestPath, outTestPath, csvTestFilePath)