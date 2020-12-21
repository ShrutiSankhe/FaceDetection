import cv2
import os
from PIL import Image
import numpy as np
import mysql.connector
import tkinter as tk
from tkinter import messagebox


window = tk.Tk()
window.title("Face Recognition System")

label_name = tk.Label(window, text="Name", font=("Helvetica",20))
label_name.grid(column=0, row=0)
input_name = tk.Entry(window, width=50, bd=5)
input_name.grid(column=1, row=0)

label_age = tk.Label(window, text="Age", font=("Helvetica",20))
label_age.grid(column=0, row=1)
input_age = tk.Entry(window, width=50, bd=5)
input_age.grid(column=1, row=1)

label_city = tk.Label(window, text="City", font=("Helvetica",20))
label_city.grid(column=0, row=2)
input_city = tk.Entry(window, width=50, bd=5)
input_city.grid(column=1, row=2)

face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def train_classifier():
    # Path to directory and each image in the folder
    data_dir = "/data"
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f !='.DS_Store']
    
    face = []
    ids = []
    
    for images in path:
        # Method to convert the images to grayscale
        image = Image.open(images).convert('L');
        # Get image in array format
        imageNp = np.array(image, 'uint8')
        # Get the user id from the image path name
        id = int(os.path.split(images)[1].split("_")[1])
    
        # Append the image and id in face and ids list
        face.append(imageNp)
        ids.append(id)

        
    # Convert ids in array format
    ids = np.array(ids)
    
    # Train the classifier using LBPHFaceRecognizer
    clf = cv2.face.LBPHFaceRecognizer_create()
    # Pass the face and ids value to train the classsifier
    clf.train(face,np.array(ids))
    # Saving the classfier
    clf.write("classifier.xml")
    messagebox.showinfo("Result","Dataset Training Completed.")
    
b1 = tk.Button(window, text="Train Dataset", font=("Helvetica",20), command=train_classifier)
b1.grid(column=0, row=4)

def detect_face():
    
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        # Convert to grayscale image
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("Convert to grayscale image")
        # Classifier to detect features
        features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
        print("Classifier to detect feature")

        for (x,y,w,h) in features:
            # Draw rectangle on the face
            cv2.rectangle(img, (x,y), (x+w,y+h), color, 2 )
            print("Draw rectangle on the face")

            # Classifier predicts the id
            id, pred = clf.predict(gray_img[y:y+h,x:x+w])
            print("Classifier predicts the id")

            # Get the confidence value from pred value
            confidence = int(100*(1-pred/300))
            print("Get the confidence value from pred value")
            
            # Database Connection
            mydb = mysql.connector.connect(
            host = "localhost",
            user = "root",
            passwd = "",
            database = "face_recognition_user"
            )
            print("Database Connection")
            mycursor = mydb.cursor()
            mycursor.execute("Select display_name from app_user where id=" + str(id))
            # Fetch the name of the user based on id, s is a tuple
            username = mycursor.fetchone() 
            # Convert the tuple into a string
            username = ''+''.join(username) 

            if confidence>75:
                cv2.putText(img, username, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            else:
                cv2.putText(img, "UNKNOWN", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)

        return img

    # Loading classifier
    #face_classifier = cv2.CascadeClassifier("/Users/shruti/anaconda3/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_default.xml")
    print("Loading classifier")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")
    print("clf.read classifier.xml")

    video_capture = cv2.VideoCapture(0)
    print("capturing video")

    # Identify the person from the captured video
    while True:
        ret, img = video_capture.read()
        img = draw_boundary(img, face_classifier, 1.3, 6, (0,0,255), "Face", clf)
        cv2.imshow("Face Detection", img)
        print("Inn the Face Detection window")

        if cv2.waitKey(1)==13:
            break
    video_capture.release()
    cv2.destroyAllWindows()
    
b2 = tk.Button(window, text="Detect Faces", font=("Helvetica",20), command=detect_face)
b2.grid(column=1, row=4)

def generate_dataset():
    print("inside generate_dataset")
    if (input_name.get()=="" or input_age.get()=="" or input_city.get()==""):
        messagebox.showinfo("Result","Please provide complete details of the user.")
    else:
        # Database connection 
        mydb = mysql.connector.connect(
            host = "localhost",
            user = "root",
            passwd = "",
            database = "face_recognition_user"
        )
        print("established db connection")
        mycursor = mydb.cursor()
        print("mycursor = mydb.cursor()")
        mycursor.execute("SELECT * from app_user")
        print("mycursor.execute SELECT * from app_user")
        myresult = mycursor.fetchall()
        print("myresult = mycursor.fetchall")
        id = 1
        for x in myresult:
            id +=1
        sql = "insert into app_user(id,display_name,age,city) values(%s, %s, %s, %s)"
        val = (id, input_name.get(), input_age.get(), input_city.get())
        mycursor.execute(sql,val)
        mydb.commit()
         
        #face_classifier = cv2.CascadeClassifier("/Users/shruti/anaconda3/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_default.xml")
    
        # Convert image to grayscale and crop the image
        def face_crop(image):
            print(image.shape)
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Scaling factor = 1.3 and Minimum neighbor = 5
            faces_detected = face_classifier.detectMultiScale(grayscale_image, 1.3, 5)

            # If there are no faces detected return nothing
            if faces_detected is ():
                return None
            # x is width of the face detected and y is the height of face detected, w and h are width and height of the image
            for(x,y,w,h) in faces_detected:
                cropped_faces = image[y:y+h, x:x+w]
            return cropped_faces

        # Captures the image of person with inbuilt camera, for external camera sources change the value to 1
        print("Captures the image of person with inbuilt camera")
        capture_cam_image = cv2.VideoCapture(0)

        # Number of images of the authorized person, initially there are no images so the value is 0
        img_id = 0

        while True:
            # Read the captured image and return values
            print("Read the captured image and return values")
            ret,frame = capture_cam_image.read()

            # The captured image frame is converted to grayscale and cropped
            if face_crop(frame) is not None:
                # Increase the image count 
                img_id += 1
                print(str(img_id))

                # Resize the captured image to 200 by 200
                cap_image_resize = cv2.resize(face_crop(frame), (200, 200))
                print("Resize the captured image to 200 by 200")

                # Convert to grayscale
                cap_image_grayscale = cv2.cvtColor(cap_image_resize, cv2.COLOR_BGR2GRAY)
                print("Convert to grayscale")

                # Specify the path of captured image for the respective user and image id
                file_path = "data/user_"+str(id)+"_"+str(img_id)+".jpg"
                print("Specify the diectory to store the captured image")

                # Store the captured image
                cv2.imwrite(file_path,cap_image_grayscale)
                print("Store the captured image")

                # Tagging the captured image 
                # 50,50 is the origin point from where the text starts, Hershey is font style
                # 1 is the font scale, R = 255 so fontcolor is Red and 2 is the font thickness
                cv2.putText(cap_image_grayscale,str(img_id),(150,150),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                print("Tagging the captured image")

                # Show the captured image
                cv2.imshow("Cropped image with face", cap_image_grayscale)
                print("Show the captured image")


                # If we press enter key(Ascii value = 13) or the captured images exceed 50 the loop will break
                if cv2.waitKey(1) == 13 or int(img_id) == 50:
                    break 


        # Release the camera and stop capturing images
        capture_cam_image.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Result","Dataset Generation Completed.")

b3 = tk.Button(window, text="Generate Dataset", font=("Helvetica",20),command=generate_dataset)
b3.grid(column=2, row=4)

window.geometry("813x200")
window.mainloop()
