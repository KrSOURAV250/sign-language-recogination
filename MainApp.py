# Import libraries
import cv2
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False)
model.trainable = False

# Define the classes
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Define the region of interest (ROI)
top, right, bottom, left = 50, 350, 250, 590

# Define the font
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0, 255, 0)
thickness = 2

# Start the webcam
cap = cv2.VideoCapture(0)

# Loop over the frames
while(True):
    # Read the frame
    ret, frame = cap.read()
    
    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Extract the ROI
    roi = frame[top:bottom, right:left]

    # Resize the ROI
    roi = cv2.resize(roi, (224, 224))

    # Preprocess the image
    x = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(roi, axis=0))

    # Use the model to make a prediction
    features = model.predict(x)



    # Get the predicted class
    pred = tf.argmax(features[0])
    # pred = np.argmax(features[0])



    # Get the predicted class name

    # Get the predicted class
    pred = np.argmax(features[0])

    # Check if pred is within the range of the classes list
    if pred < len(classes):
        # Get the predicted class name
        pred_class_name = classes[pred]
    else:
        # If pred is out of range, set the predicted class name to None
        pred_class_name = None



    # Draw a rectangle around the ROI
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # Draw the predicted class name on the frame
    cv2.putText(frame, pred_class_name, (left, top-10), font, fontScale, fontColor, thickness, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('frame',frame)
    
    # Stop the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
