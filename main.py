## Import necessary packages
import os
from kivymd.app import MDApp
from kivymd.theming import ThemeManager
from kivy.uix.boxlayout import BoxLayout
from kivymd.toast.kivytoast.kivytoast import toast
from kivy.utils import get_hex_from_color

from kivymd.uix.dialog import  MDDialog
# NavigationDrawer
from kivy.properties import StringProperty

from kivy.properties import ObjectProperty
from kivymd.uix.list import OneLineAvatarListItem
from kivymd.uix.list import OneLineListItem, MDList, TwoLineListItem, ThreeLineListItem
from kivymd.uix.list import OneLineIconListItem, IconLeftWidget
from kivy.uix.scrollview import ScrollView
import sqlite3
import cv2 as cv
import numpy as np
import json
from helper.preprocess import preprocess_image, face_detector
import datetime

class ContentNavigationDrawer(BoxLayout):
    pass


class NavigationItem(OneLineAvatarListItem):
    icon = StringProperty()




class RootWidget(BoxLayout):
    box = ObjectProperty(None)
    def ViewAttendanceScreen(self):
        con = sqlite3.connect('example.db')
        cur = con.cursor()

        a = cur.execute("select * from stocks")
        i = 0

        for idrowID in a :
            items = OneLineIconListItem(text=str(i) + ':    '+'id= '+idrowID[0]+
                                             '  name= '+idrowID[1]+'    time= '+idrowID[2])

            self.ids.container.add_widget(items)

            print(" id= ",idrowID[0],"name= ",idrowID[1],"time= ",idrowID[2])

            i +=1
            print(i)


    def ViewDatabase(self):
        #os.system("python serach_for_student.py")
        font = cv.FONT_HERSHEY_COMPLEX
        with open('labels_to_name.json') as json_file:
            labels_to_name = json.load(json_file)

        # Load the model from our assets
        model = cv.face.LBPHFaceRecognizer_create()
        model.read("trainer/model.xml")

        # Initializing the webcam to capture live video
        video_capture = cv.VideoCapture(0)
        count = 0
        while True:
            _, img = video_capture.read()
            gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            face, coord = face_detector(gray_img)
            x, y, w, h = coord
            if face is not None:
                face = preprocess_image(face)
                count = count + 1
                # Pass the face to model
                # "results" comprises of a tuple
                # containing label and confidence value

                results = model.predict(face)

                if results[1] < 500:
                    confidence = int(100 * (1 - (results[1] / 300)))
                    matched_id = str(results[0])
                    student_name = labels_to_name[matched_id]["full_name"]
                    student_id = labels_to_name[matched_id]["student_id"]
                    display_string = "Name: {} ID: {}".format(student_name, student_id)
                    if count == 10:
                        print(display_string)
                    if count==1:
                        itemss = OneLineIconListItem(text=str(count) + ':    ' + display_string)
                        self.ids.containerr.add_widget(itemss)
                if confidence > 70:

                    cv.putText(img, display_string, (x + 10, y - 30), font, 1, (255, 120, 150), 2)


                    cv.imshow("Face Cropper", img)
                else:
                    cv.putText(img, "Unknown", (x + 10, y - 30), font, 1, (255, 120, 150), 2)
                    cv.imshow("Face Cropper", img)

            else:
                cv.imshow("Face Cropper", img)

            if (cv.waitKey(1) & 0xFF == ord("q")) or count == 10:
                break

        video_capture.release()
        cv.destroyAllWindows()
## This is the main app.

class MainApp(MDApp):

    def __init__(self, **kwargs):
        self.title = "Face Recognition"
        self.theme_cls = ThemeManager()
        self.theme_cls.primary_palette = "Teal"
        self.theme_cls.accent_palette = "Blue"
        self.theme_cls.theme_style="Light"

        super().__init__(**kwargs)
    
    
    def build(self):
        self.icon="logo1.png"
        return RootWidget()
    
    def back_to_home_screen(self):
        self.root.ids.student_name.text = ""
        self.root.ids.student_id.text = ""
        self.root.ids.screen_manager.current = "HomeScreen"








    def show_ExitDialog(self):
        dialog = MDDialog(
            title="Attendance Management System", 
            text = "Are you [color=%s][b]sure[/b][/color] ?"
            % get_hex_from_color(self.theme_cls.primary_color), 
            size_hint=[.5, .3],
        events_callback=self.stopApp,
        text_button_ok="Exit",
        text_button_cancel="Cancel"
        )
        dialog.open()
    
    def stopApp(self, text_of_selection, popup_widget):
        
        if text_of_selection == "Exit":
            self.stop()
        else:
            pass
    
    def performAttendance(self):
        os.system("python faceRecognition.py")







    
    def captureTrainingImages(self, student_name, student_id, screen_manager):
        
        print(len(student_name), len(student_id))
        if len(student_name) > 0 and len(student_name) <= 23 and len(student_id) > 0 and len(student_id) <= 13:
            os.system("python captureTrainingImages.py {} {}".format(student_name, student_id))
            toast("Training Images Collected")
            screen_manager.current = "HomeScreen"
        else:
            toast("Please Enter Correct Details")
    



MainApp().run()
