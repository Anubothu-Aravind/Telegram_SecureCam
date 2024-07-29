
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime
import telebot
import os
from dotenv import load_dotenv
load_dotenv()

token=os.getenv("telebottoken")
os.environ['telebottoken'] = telegram_bot_token
chat_id = os.getenv("chat_id")
os.environ['chat_id']=telegram_chat_id
bot = telebot.TeleBot(telegram_bot_token)

model = load_model("C:\\users\\Downloads\\model_latest.h5")

classes = ['small gun', 'large gun', 'knife']
def preprocess_image(img, target_size=(224, 224)):
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    return img

def predict_object(img):
    img = preprocess_image(img)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    return classes[class_index], prediction[0][class_index]


camera = cv2.VideoCapture(0)
ret, frame = camera.read()
if ret:
    cv2.imwrite("initial_snapshot.jpg", frame)
    photo = open("initial_snapshot.jpg", "rb")
    bot.send_photo(telegram_chat_id, photo)
else:
    print("Error capturing the initial photo.")
camera.release()

fullbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
camera = cv2.VideoCapture(0)
start_time = datetime.now()

while True:
    ret, frame = camera.read()
    if not ret:
        print("Error reading the camera feed.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    bodies = fullbody_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(bodies) > 0:
        for (x, y, w, h) in bodies:
            current_time = datetime.now()
            waiting_time = (current_time - start_time).total_seconds() / 60
            body_image = frame[y:y + h, x:x + w]

            # Predict objects in the detected body
            object_class, confidence = predict_object(body_image)

            if object_class in classes:
                alert_message = f"Visitor has a {object_class} with confidence {confidence:.2f}."
                bot.send_message(telegram_chat_id, alert_message)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Change color to red
                cv2.imwrite("visitor_snapshot.jpg", frame)
                photo = open("visitor_snapshot.jpg", "rb")
                bot.send_photo(telegram_chat_id, photo)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Full Body Snapshot", frame)
    cv2.waitKey(1000)

    if 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
