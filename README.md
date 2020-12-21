The project involves processing of the extracted area of interest. After identifying the area of interest, which is primarily the face of an individual, a Haar classifier is used to classify the facial region. 

Multiple factors like lighting, contrast, pose, types of facial hair, external accessories etc. interfere with the accuracy of the face detection. This interference demands determining the face accurately which is achieved by adding data at preprocessing stages.

Libraries Used
•	openCV python-4.4.0.46 which is an Image processing library
•	os for reading the training directories and filenames 
•	PIL for adding image processing abilities to python interpreter
•	numpy to import image data into numpy array
•	mysql.connector for accessing the database
•	tkinter as the GUI package 

Face Recognition has following stages:
1)	Generating the dataset
2)	Training the classifier
3)	Face detection and identification
4)	GUI Face Recognition application
5)	Database Connection

