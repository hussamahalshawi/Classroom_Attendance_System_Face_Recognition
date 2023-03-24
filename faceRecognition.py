

# Importing the required libraries
import cv2 as cv
import numpy as np
import json
from helper.preprocess import preprocess_image, face_detector
import datetime
import sqlite3


font = cv.FONT_HERSHEY_COMPLEX
con = sqlite3.connect('example.db')


cur = con.cursor()

cur.execute("CREATE TABLE IF NOT EXISTS stocks ( ID,name, time)")





with open('labels_to_name.json') as json_file:
    labels_to_name = json.load(json_file)


# Load the model from our assets
model = cv.face.LBPHFaceRecognizer_create()
model.read("trainer/model.xml")




# Initializing the webcam to capture live video
video_capture = cv.VideoCapture(0)
i = 0
a = cur.execute("select ID from stocks")
while True:
        _, img = video_capture.read()
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        face, coord = face_detector(gray_img)
        x,y,w,h = coord
        if face is not None:
            face = preprocess_image(face)
            # Pass the face to model
            # "results" comprises of a tuple
            # containing label and confidence value

            results = model.predict(face)



            if results[1] < 500:
                confidence = int(100 * (1-(results[1]/300)))
                matched_id = str(results[0])
                student_name = labels_to_name[matched_id]["full_name"]
                student_id = labels_to_name[matched_id]["student_id"]
                display_string  = "Name: {} ID: {}".format(student_name, student_id)

            

            if confidence > 70:

                cv.putText(img, display_string, (x+10, y-30), font, 1, (255, 120, 150), 2)

                dt = datetime.datetime.now()
                #print(dt)
                #cur.execute("select * from stocks where ID = 14141")

                #q=cur.execute("select * from stocks where ID ='%s'"%student_id)
                #q = cur.execute("select * from stocks where ID ")
                #print(q)
                #result = cur.fetchone()
                #a = cur.execute("select ID from stocks")
                for idrow in a:
                    print(idrow[0])
                    if idrow[0] == student_id:
                        print("t")
                    else:
                        cur.execute("INSERT INTO stocks VALUES (?,?,?)", (student_id,student_name,dt))





                #print(result[0])
                '''for idrow in a:
                    #print(idrow[i])
                    if result is not None :
                        print(result)
                        #cur.execute("INSERT INTO stocks VALUES (?,?,?)", (student_id,student_name,dt))
                    else:
                        print(result)
                        cur.execute("INSERT INTO stocks VALUES (?,?,?)", (student_name, student_id, dt))


                    i += 1'''
                print(i)
                con.commit()

                cv.imshow("Face Cropper", img)
            else:
                cv.putText(img, "Unknown", (x+10, y-30), font, 1, (255, 120, 150), 2)
                cv.imshow("Face Cropper", img)
    
        else:
            cv.imshow("Face Cropper", img)
        i += 1
        if (cv.waitKey(1) & 0xFF == ord("q")):
            break


con.close()
video_capture.release()
cv.destroyAllWindows()
# Our face recognition is finally working