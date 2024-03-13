import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np

def process_image(img, hands, model, classes, data):
    frameRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    hands_points = results.multi_hand_landmarks
    h, w, _ = img.shape

    if hands_points is not None:
        for hand in hands_points:
            x_min, x_max, y_min, y_max = find_hand_boundaries(hand, h, w)
            cv2.rectangle(img, (x_min - 50, y_min - 50), (x_max + 50, y_max + 50), (255, 255, 255), 2)

            try:
                process_hand_crop(img, x_min, x_max, y_min, y_max, model, classes, data)

            except Exception as e:
                print(f"Error processing hand crop: {e}")

def find_hand_boundaries(hand, h, w):
    x_max, y_max, x_min, y_min = 0, 0, w, h

    for lm in hand.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        x_max = max(x_max, x)
        x_min = min(x_min, x)
        y_max = max(y_max, y)
        y_min = min(y_min, y)

    return x_min, x_max, y_min, y_max

def process_hand_crop(img, x_min, x_max, y_min, y_max, model, classes, data):
    img_crop = img[y_min - 50:y_max + 50, x_min - 50:x_max + 50]
    img_crop = cv2.resize(img_crop, (224, 224))
    img_array = np.asarray(img_crop)
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index_val = np.argmax(prediction)
    cv2.putText(img, classes[index_val], (x_min - 50, y_min - 65), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

def main():
    cap = cv2.VideoCapture(0)
    hands = mp.solutions.hands.Hands(max_num_hands=1)
    classes = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
    'V', 'W', 'X', 'Y', 'Z', 'Eu te amo'
    ]
    model = load_model('keras_model.h5')
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    while True:
        success, img = cap.read()

        if not success:
            break

        process_image(img, hands, model, classes, data)

        cv2.imshow('Libras Now', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
