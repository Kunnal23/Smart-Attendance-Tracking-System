import os
import cv2
import numpy as np
from PIL import Image


def getImagesAndLabels(path):
    os.makedirs(path, exist_ok=True)  
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")]
    faces = []
    Ids = []

    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')  
        imageNp = np.array(pilImage, 'uint8')

        try:
            Id = int(os.path.split(imagePath)[-1].split(".")[1])  
        except ValueError:
            print(f"⚠️ Skipping invalid file: {imagePath}")
            continue

        faces.append(imageNp)
        Ids.append(Id)

    return faces, Ids


def TrainImages():
    os.makedirs("TrainingImageLabel", exist_ok=True)  

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create() 
    except AttributeError:
        print("❌ OpenCV not installed correctly! Run:")
        print("   pip install opencv-contrib-python")
        return

    faces, Ids = getImagesAndLabels("TrainingImage")

    if not faces:
        print("❌ No training images found!")
        return

    recognizer.train(faces, np.array(Ids))  
    recognizer.save("TrainingImageLabel/Trainer.yml")  
    print("\n✅ Training Completed Successfully!")

if __name__ == "__main__":
    TrainImages()
