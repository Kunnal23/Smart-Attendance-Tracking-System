import os  
import check_camera
import capture_image
import train_image
import recognize


def title_bar():
    os.system('cls')
    print("\t***** Smart Attendance Tracking System *****")


def mainMenu():
    title_bar()
    print()
    print(10 * "*", "DASHBOARD", 10 * "*")
    print("[1] Check Camera")
    print("[2] Capture Faces")
    print("[3] Train Images")
    print("[4] Recognize & Attendance")
    print("[5] Quit")
    while True:
        try:
            choice = int(input("Enter Choice: "))
            if choice == 1:
                checkCamera()
                break
            elif choice == 2:
                CaptureFaces()  
                break
            elif choice == 3:
                Trainimages()
                break
            elif choice == 4:
                recognizeFaces()
                break
            elif choice == 5:
                print("Thank You")
                break
            else:
                print("Invalid Choice. Enter 1-4")
                mainMenu()
        except ValueError:
            print("Invalid Choice. Enter 1-4\n Try Again")

def checkCamera():
    check_camera.camer()
    input("Enter any key to return main menu")
    mainMenu()


def CaptureFaces():
    capture_image.takeImages()
    input("Enter any key to return main menu")
    mainMenu()


def Trainimages():
    train_image.TrainImages()
    input("Enter any key to return main menu")
    mainMenu()


def recognizeFaces():
    recognize.recognize_attendance()
    input("Enter any key to return main menu")
    mainMenu()


mainMenu()

