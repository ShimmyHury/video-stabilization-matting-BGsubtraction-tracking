from tkinter import Tk, Label, Button
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

"""
Example for a Graphical User Interface.
Using tkinter library.
"""

global CroppedWin
global img

def is_image(file_name):
    ret_val = False
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        ret_val = True
    else:
        print("The file {} is not an image file! Please select a valid file".format(file_name))
    return ret_val


class TrackingGUI:
    def __init__(self, master_window):
        self.master_window = master_window
        master_window.title("Video Processing Lab GUI Example")
        self.filename = None
        self.crop_image_roi = None

        self.show_image_label = Label(master_window, text="View Selected Image:")
        self.show_image_label.pack()
        self.show_image_button = Button(master_window, text="Show Image", command=self.show_image)
        self.show_image_button.pack()

        self.get_interest_region_label = Label(master_window, text="Select Interest Region:")
        self.get_interest_region_label.pack()
        self.get_interest_region_button = Button(master_window, text="Select",
                                                 command=self.select_interest_region)
        self.get_interest_region_button.pack()
        
        self.view_selected_region_label = Label(master_window, text="View Selected Region: ")
        self.view_selected_region_label.pack()
        self.view_selected_region_button = Button(master_window, text="View Selection",
                                                  command=self.view_selected_region)
        self.view_selected_region_button.pack()

        self.view_selected_region_label = Label(master_window, text="Return cropped region and start tracking: ")
        self.view_selected_region_label.pack()
        self.view_selected_region_button = Button(master_window, text="Start tracking",
                                                  command=self.Return_cropped_region_and_exit)
        self.view_selected_region_button.pack()

    """
    Definition of functions that would be called in a case of a button click
    """

    def show_image(self):
        global img
        try:
            cv2.imshow('First Frame',img)
            cv2.waitKey(1)
        except:
            print("Couldn't show image. Verify that first frame is an image")


    def select_interest_region(self):
        global img
        try:
            image = img
            # Select ROI
            from_center = False
            roi = cv2.selectROI("Drag the rect from the top left to the bottom right corner of the forground object,"
                                " then press ENTER.",
                                image, from_center)
            # Crop image
            self.crop_image_roi = roi
            cv2.destroyAllWindows()
            cv2.waitKey(1000)
            print("Region to track chosen.")

        except:
            print("Couldn't load image. Verify that first frame is an image")

    def view_selected_region(self):
        # Display cropped image
        global img
        if self.crop_image_roi:
            image = img
            crop_image = image[int(self.crop_image_roi[1]):int(self.crop_image_roi[1]+self.crop_image_roi[3]),
                               int(self.crop_image_roi[0]):int(self.crop_image_roi[0]+self.crop_image_roi[2])]
            cv2.imshow("Selected Part", crop_image)
            cv2.waitKey(1)
    def Return_cropped_region_and_exit(self):
        # Display cropped image
        self.master_window.destroy()
        global CroppedWin
        CroppedWin = self.crop_image_roi
        print("Region to track validated. Starting tracking.")
        return


def GUI_selection_of_tracked_person(First_frame):
    global img
    img = First_frame
    root = Tk()
    root.geometry("300x300")  # You want the size of the app to be 200x200
    example_gui = TrackingGUI(root)
    root.mainloop()
    return CroppedWin
