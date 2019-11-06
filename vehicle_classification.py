# Classify vehicles in optical images, to small and large, using simple cnn
# We trained the cnn before
# The steps are:
# (1) Load the model that was saved in the training stage
# (2) Compile the model
# (3) Get a new image
# (4) Display it
# (5) Find all vegicle in the image (detection phase was before - we get collection of coordinate of bounding boxes of the vehicle)
#     crop all vehicle into small images and saves them in a separate folder
# (6) Run the alg on each vehicle image and find its classification
# (7) According to the result of the alg, draw rectangle around the vehicle in the original image
#     Different color for small - yellow, and large - light blue
# (8) Saves the new image with the rectangles and display it

# Import all pkgs
import prepare_data
from prepare_data import BoundingBox
from prepare_data import  Point
import cv2
import os
import numpy as np
from keras.models import model_from_json
import csv

# Define variables
csvFileName = '/home/revital/Documents/Project/DS/test/train1.csv'
inputPath = '/home/revital/Documents/Project/DS/test/inputImages/'
outputPath = '/home/revital/Documents/Project/DS/test/outImages/'
csvResFile = '/home/revital/Documents/Project/DS/test/results.csv'
classifiedVehiclesInImg = dict()
resDict = dict()

# Load the model from Json file that was saved in the training stage
jsonFile = open('./model.json', 'r')
loadedModelJson = jsonFile.read()
jsonFile.close()
loadedModel = model_from_json(loadedModelJson)

# Load the weights from the file that was save in the training stage
loadedModel.load_weights("./model.h5")
print("Model is loaded")

# Compile the model
loadedModel.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
print("Model is compiled")

# Prepare the data of the input image
tagsCoordsList = prepare_data.prepareData(inputPath, outputPath, csvFileName)

# Get the image, display it and classify all vehicles in the image
listOfFile = os.listdir(inputPath)
for imageFileName in listOfFile:

    # Clear the previous list of classified tags
    classifiedVehiclesInImg.clear()

    # Read the image and display it
    fullFileName = inputPath + imageFileName
    print("Input image: " + fullFileName)

    # get the image id
    imageFileNameNoPrefix = imageFileName.split(".")
    imageId = imageFileNameNoPrefix[0]

    try:
        image = cv2.imread(fullFileName)
        cv2.imshow("Input Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("bad image")
        continue

    # For each image that was cropped, check its classification using the CNN
    croppedPath = outputPath + "out/"
    listOfFile = os.listdir(croppedPath)

    for croppedFileName in listOfFile:

        # Read the cropped image
        fullCroppedFileName = croppedPath + croppedFileName
        print("Cropped image: " + fullCroppedFileName)

        # Get the tag id and image id from the cropped file name
        croppedFileName = croppedFileName.split(".")
        croppedIdsList = croppedFileName[0].split("_")
        imageIdOfCropped = croppedIdsList[0]
        tagId = croppedIdsList[1]

        # files of different image - not to process currently
        if imageIdOfCropped != imageId:
            continue

        try:
            croppedImage = cv2.imread(fullCroppedFileName)
            croppedImage = cv2.resize(croppedImage, (32, 32))
            croppedImage = croppedImage.reshape(1, 32, 32, 3)
        except:
            print("Bad Image")
            continue

        # Predict to which class the cropped image was classified
        result = loadedModel.predict_classes(croppedImage)

        if result[0][0] == 1:
            print("Small vehicle")
            # add the classified cur vehicle as small to the list of all vehicles in the image
            classifiedVehiclesInImg[tagId] = "small"
        else:
            # add the classified cur vehicle as large to the list of all vehicles in the image
            classifiedVehiclesInImg[tagId] = "large"
            print("Large vehicle")

    # Go over the classified list and print rectangles with corresponding colors to the classification around each vehicle in the image

    # Go over the vehicles in the list
    for curVehicleTag in classifiedVehiclesInImg:

        # get the coords
        curVehicleUpperLeft = tagsCoordsList[curVehicleTag][0]
        curVehicleBottomRight = tagsCoordsList[curVehicleTag][1]

        # draw the rectangle

        # check the classification and decide the color accordingly

        # small = yellow
        if classifiedVehiclesInImg[curVehicleTag] == "small":
            cv2.rectangle(image, (int(curVehicleUpperLeft.x), int(curVehicleBottomRight.y)),
                                  (int(curVehicleBottomRight.x), int(curVehicleUpperLeft.y)),
                                  (0,255,255), 3)

            #cv2.putText(image, curVehicleTag, (int(curVehicleUpperLeft.x), int(curVehicleBottomRight.y)),
            #                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), lineType=cv2.LINE_AA)

            # Insert result to dictionary results list for saving to file
            resDict[curVehicleTag] = ("small", imageId)

        # large = light blue
        else:
            cv2.rectangle(image, (int(curVehicleUpperLeft.x), int(curVehicleBottomRight.y)),
                                  (int(curVehicleBottomRight.x), int(curVehicleUpperLeft.y)),
                                  (255,255,0), 3)

            #cv2.putText(image, curVehicleTag, (int(curVehicleUpperLeft.x), int(curVehicleBottomRight.y)),
            #           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), lineType=cv2.LINE_AA)

            # Insert result to dictionary results list for saving to file
            resDict[curVehicleTag] = ("large", imageId)

    # save the new image and display it
    newImageFullPath = outputPath + "results/res_" + imageFileName
    cv2.imwrite(newImageFullPath, image)

    cv2.imshow("Output Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Save results to csv file
fieldsNames = ['image_id', 'tag_id', 'classification']
with open(csvResFile, 'w') as csvFile:
    dict_writer = csv.writer(csvFile)
    csvFile.write("%s,%s,%s\n" % ('tag_id', 'classification', 'image_id'))
    for tagIdKey in resDict.keys():
        csvFile.write("%s,%s,%s\n" % (tagIdKey, resDict[tagIdKey][0], resDict[tagIdKey][1]))
print("Results CSV file completed")
csvFile.close()