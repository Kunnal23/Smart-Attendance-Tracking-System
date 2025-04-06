import csv
import cv2
import os
import numpy as np


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def enhance_image(image):
    return cv2.equalizeHist(image)


def takeImages():
    Id = input("Enter Your ID (numeric): ")
    name = input("Enter Your Name (alphabetic): ")

    
    if not is_number(Id):
        print("❌ Error: ID must be numeric!")
        return
    if not name.isalpha():
        print("❌ Error: Name must contain only alphabets!")
        return

    os.makedirs("TrainingImage", exist_ok=True)
    os.makedirs("StudentDetails", exist_ok=True)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("❌ Error: Camera not accessible!")
        return

    harcascadePath = "haarcascade_default.xml"
    if not os.path.exists(harcascadePath):
        print("❌ Error: Haarcascade file missing!")
        return
    detector = cv2.CascadeClassifier(harcascadePath)

    sampleNum = 0
    print("\n📸 Capturing images... Look straight, left, right, up, and down!")

    while True:
        ret, img = cam.read()
        if not ret:
            print("❌ Error: Unable to access the camera.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = enhance_image(gray)  
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(40, 40))

        for (x, y, w, h) in faces:
            sampleNum += 1
            face_image = gray[y:y+h, x:x+w]
            
            filename = f"TrainingImage/{name}.{Id}.{sampleNum}.jpg"
            cv2.imwrite(filename, face_image)

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if sampleNum % 10 == 0:
                print(f"📷 {sampleNum} images captured... Move your face slightly!")

        cv2.imshow("Capturing Faces - Press 'Q' to Quit", img)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n🛑 Capture stopped by user!")
            break
        elif sampleNum >= 150:  
            print("\n✅ 150 images captured successfully!")
            break

    cam.release()
    cv2.destroyAllWindows()
    
    student_file = "StudentDetails/StudentDetails.csv"
    file_exists = os.path.exists(student_file)

    with open(student_file, 'a+', newline='') as csvFile:
        writer = csv.writer(csvFile)
        if not file_exists:
            writer.writerow(["Id", "Name"])  
        writer.writerow([Id, name])

    print(f"✅ Images saved for ID: {Id}, Name: {name}")

if __name__ == "__main__":
    takeImages()
