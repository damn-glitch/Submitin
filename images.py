import pytesseract
from PIL import Image
import os
import cv2
import numpy as np

import cv2

def main():
    # Read the text from the image
    # Provide the path to the image file
    image_path = 'C:\\Users\\alish\\Downloads\\e91ca08c8097e23d190c9ea788bfbf72.jpg'

    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply image preprocessing techniques (e.g., noise removal, thresholding)
    gray = cv2.medianBlur(gray, 3)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Create a temporary image file to feed into Tesseract
    temp_image_path = "temp_image.png"
    cv2.imwrite(temp_image_path, gray)

    # Perform OCR using Tesseract
    text = pytesseract.image_to_string(Image.open(temp_image_path))

    # Print the extracted text
    print(text)

    # Clean up temporary image file
    os.remove(temp_image_path)
    
    
    #See the objects on the image
    # Load the pre-trained model and configuration files
    model_path = 'path_to_model_file.pb'
    config_path = 'path_to_config_file.pbtxt'
    model = cv2.dnn_DetectionModel(model_path, config_path)

    # Set the input size and scale factor for the model
    input_size = (300, 300)
    scale = 1.0

    # Load the image
    image_path = 'path_to_image.jpg'
    image = cv2.imread(image_path)

    # Preprocess the image
    blob = cv2.dnn.blobFromImage(image, scale, input_size, (104, 177, 123))

    # Set the input to the model
    model.setInput(blob)

    # Perform object detection
    detections = model.forward()

    # Iterate over the detected objects
    for detection in detections[0, 0]:
        confidence = detection[2]
        class_id = int(detection[1])

        # Print the class label and confidence score
        print(f"Class ID: {class_id}, Confidence: {confidence}")

        # Draw a bounding box around the object
        if confidence > 0.5:
            box = detection[3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            (start_x, start_y, end_x, end_y) = box.astype("int")
            cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    # Display the image with bounding boxes
    cv2.imshow("Object Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()