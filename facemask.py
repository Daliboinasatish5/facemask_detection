import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import datetime

# BUILDING MODEL TO CLASSIFY BETWEEN MASK AND NO MASK - Updated architecture
def create_model():
    model = Sequential([
        Input(shape=(150, 150, 3)),  # Proper way to define input shape
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create model
model = create_model()

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'train',
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary')

# Check which class corresponds to mask vs no mask
print(f"Class indices: {training_set.class_indices}")

test_set = test_datagen.flow_from_directory(
    'test',
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary')

# Train model
history = model.fit(
    training_set,
    epochs=5,
    validation_data=test_set
)

# Save model
model.save('mymodel.h5')

# To test for individual images
mymodel = load_model('mymodel.h5')
test_image = cv2.imread(r'test\without_mask\31.jpg')  # Updated path to be relative
test_image = cv2.resize(test_image, (150, 150))
test_image = np.expand_dims(test_image, axis=0)
prediction = mymodel.predict(test_image)[0][0]
print(f"Prediction for test image: {prediction}")

# IMPLEMENTING LIVE DETECTION OF FACE MASK
def start_webcam_detection():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Important: Check the class mapping from training
    # This is critical - we need to reverse the prediction logic
    # Based on your image showing incorrect detection
    print("Starting webcam detection with REVERSED prediction logic")
    
    while cap.isOpened():
        _, img = cap.read()
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, qminNeighbors=4)
        
        for (x, y, w, h) in faces:
            face_img = img[y:y+h, x:x+w]
            cv2.imwrite('temp.jpg', face_img)
            test_image = cv2.imread('temp.jpg')
            test_image = cv2.resize(test_image, (150, 150))
            test_image = np.expand_dims(test_image, axis=0)
            pred = mymodel.predict(test_image)[0][0]
            
            # Debug: Print the prediction value to console
            print(f"Face prediction value: {pred:.4f}")
            
            # *** REVERSED LOGIC - This is the key fix ***
            # If your model is incorrectly classifying, we need to reverse the comparison
            if pred < 0.5:  # Now this means NO_MASK based on our reversed mapping
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)  # Red box for no mask
                cv2.putText(img, 'NO MASK', ((x+w)//2, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:  # This means HAS_MASK based on our reversed mapping
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)  # Green box for mask
                cv2.putText(img, 'MASK', ((x+w)//2, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                
            datet = str(datetime.datetime.now())
            cv2.putText(img, datet, (400, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display the prediction value on screen for debugging
        cv2.putText(img, f"Pred: {pred:.4f}" if 'pred' in locals() else "No face", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.imshow('Face Mask Detection', img)
        
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Model training complete. Run webcam detection? (y/n)")
    choice = input().lower()
    if choice == 'y':
        start_webcam_detection()