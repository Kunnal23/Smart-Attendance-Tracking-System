import cv2
import os
import pandas as pd
import time
import datetime

def recognize_attendance():
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel/Trainer.yml")

    harcascadePath = "haarcascade_default.xml"
    if not os.path.exists(harcascadePath):
        print("❌ Error: Haarcascade file is missing!")
        return
    faceCascade = cv2.CascadeClassifier(harcascadePath)

    student_file = "StudentDetails/StudentDetails.csv"
    if not os.path.exists(student_file):
        print("❌ Error: Student details file is missing!")
        return
    df = pd.read_csv(student_file)

    attendance_file = "Attendance/Attendance.csv"
    os.makedirs("Attendance", exist_ok=True)
    
    if os.path.exists(attendance_file):
        attendance = pd.read_csv(attendance_file)
        serial_no = len(attendance) + 1  
    else:
        attendance = pd.DataFrame(columns=["Serial No", "Id", "Name", "Date", "Time"])
        serial_no = 1  

    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    font = cv2.FONT_HERSHEY_SIMPLEX
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    print("📸 Recognizing Faces... Press 'Q' to quit.")

    recognized_faces = set()  

    while True:
        ret, im = cam.read()
        if not ret:
            print("❌ Error: Unable to access camera!")
            break

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(int(minW), int(minH)), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])

            if conf < 100:
                
                name_data = df.loc[df['Id'] == Id, 'Name']
                if not name_data.empty:
                    name = name_data.values[0]
                else:
                    name = "Unknown"
            else:
                Id = "Unknown"
                name = "Unknown"


            if (100 - conf) > 60 and Id not in recognized_faces:
                
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

                new_entry = pd.DataFrame([[serial_no, Id, name, date, timeStamp]], columns=["Serial No", "Id", "Name", "Date", "Time"])
                attendance = pd.concat([attendance, new_entry], ignore_index=True)
                recognized_faces.add(Id) 
                serial_no += 1  

            display_text = f"{Id} - {name}"
            conf_str = f"{round(100 - conf)}%"
            cv2.putText(im, display_text, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(im, conf_str, (x + 5, y + h - 5), font, 1, (0, 255, 0), 1)

        cv2.imshow("Face Recognition", im)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n🛑 Stopped by user.")
            break


    attendance.to_csv(attendance_file, index=False)
    print("\n✅ Attendance successfully recorded in 'Attendance.csv'.")


    cam.release()
    cv2.destroyAllWindows()

