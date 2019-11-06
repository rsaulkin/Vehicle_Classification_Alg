# Vehicle_Classification_Alg
Simple Vehicle clasiffication algorithm (for classifying large and small vehicles).

This is an implementaion of vehicles classification algorithm on aerial optical images.
The alg classify the vehicles to large and small vehicles, using simple CNN.
It is part of my final paper of my degree in Software Engineering.

The DS is a collection of aerial images (tiff and jpg) with corresponding csv file, indicating the location of the vehicles in the images.
The DS is part of the COFGA DS that was introduces in the MAFAT challange.
The DS can be found here: https://github.com/bok11/02456-MAFAT/tree/master/dataset_v2

For the training part, the CSV also contained a classification type for each vehicle, indicating if the car is large or small.

It contains 4 python source code files:
1) check_classify_alg.py - Perform predictions on several algorithmic models (knn, svm, decision tree, random forst and MLP) using sklearn library, and imutils library (of pyimagesearch).
2) prepare_data.py - Prepare the DS. It recieves input images (tiff and jpg) and the csv file, and for each vehicle in the image, it crops it according to the coordinates in the csv file, and saves it. It prepares the data for the training stage, and then, saves the small cropped vehicles in a specific folder and the large ones in another. It also prepares the data for the test stage, there it saves all cropped vehicles in the same folder.
3) create_and_train_cnn.py - The initial stage of creating and training the network. The output is save in json file and H5 file for later.
4) vehicle_classification.py - The main source code. It runs the classification itself. Loads the previous saved files, and train the network accordingly. It gets an input image(s) and csv file with no known classification, and for each image, creates and save a new classified image, which display around each vehicle a bounding box with different colors: yellow, for small vehicles and blue for large ones.

I also attached a video file: vehivle_class_alg_run.mp4 - showing the running of the classifier alg.
