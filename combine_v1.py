import pytesseract
from PIL import Image
import os
import cv2
import numpy as np
import spacy
import PyPDF2
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# Function to read PDF
def read_pdf(file_path):
    pdf_file = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)
    text = " ".join([pdf_reader.getPage(i).extractText() for i in range(pdf_reader.numPages)])
    pdf_file.close()
    return text

# Function to read DOCX
def read_docx(file_path):
    doc = docx.Document(file_path)
    text = " ".join([para.text for para in doc.paragraphs])
    return text

# Read the text from the image
def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    temp_image_path = "temp_image.png"
    cv2.imwrite(temp_image_path, gray)
    text = pytesseract.image_to_string(Image.open(temp_image_path))
    os.remove(temp_image_path)
    return text

# See the objects on the image
def detect_objects(image_path, model_path, config_path):
    model = cv2.dnn_DetectionModel(model_path, config_path)
    input_size = (300, 300)
    scale = 1.0
    image = cv2.imread(image_path)
    blob = cv2.dnn.blobFromImage(image, scale, input_size, (104, 177, 123))
    model.setInput(blob)
    detections = model.forward()
    for detection in detections[0, 0]:
        confidence = detection[2]
        class_id = int(detection[1])
        print(f"Class ID: {class_id}, Confidence: {confidence}")
        if confidence > 0.5:
            box = detection[3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            (start_x, start_y, end_x, end_y) = box.astype("int")
            cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    cv2.imshow("Object Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Preprocess the texts
def preprocess_text(text):
    doc = nlp(text)
    return " ".join(token.lemma_ for token in doc if not token.is_stop and token.is_alpha)

# Load data
author_files = [...]  # List of file paths by the author (PDF or DOCX)
other_files = [...]  # List of file paths by others (PDF or DOCX)

author_texts = []
other_texts = []

# Read the files
for file in author_files:
    if file.endswith('.pdf'):
        author_texts.append(read_pdf(file))
    elif file.endswith('.docx'):
        author_texts.append(read_docx(file))

for file in other_files:
    if file.endswith('.pdf'):
        other_texts.append(read_pdf(file))
    elif file.endswith('.docx'):
        other_texts.append(read_docx(file))

# Preprocess the texts
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
author_texts_preprocessed = [preprocess_text(text) for text in author_texts]
other_texts_preprocessed = [preprocess_text(text) for text in other_texts]

# Create feature representation
vectorizer = TfidfVectorizer()
X_author = vectorizer.fit_transform(author_texts_preprocessed)
X_other = vectorizer.transform(other_texts_preprocessed)

# Create labels
y_author = np.ones(len(author_texts))
y_other = np.zeros(len(other_texts))

# Combine data and split into train and test sets
X = np.vstack((X_author.toarray(), X_other.toarray()))
y = np.hstack((y_author, y_other))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf = SVC(kernel="linear", probability=True)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
with open("model.pickle", "wb") as model_file:
    pickle.dump(clf, model_file)

with open("vectorizer.pickle", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

# Load model and vectorizer
with open("model.pickle", "rb") as model_file:
    clf_loaded = pickle.load(model_file)

with open("vectorizer.pickle", "rb") as vec_file:
    vectorizer_loaded = pickle.load(vec_file)

# Check plagiarism
def check_plagiarism(text):
    preprocessed = preprocess_text(text)
    features = vectorizer_loaded.transform([preprocessed]).toarray()
    prob = clf_loaded.predict_proba(features)
    return prob[0][1]  # Probability of being written by the author

# Main function
def main():
    image_path = 'C:\\Users\\alish\\Downloads\\e91ca08c8097e23d190c9ea788bfbf72.jpg'
    model_path = 'path_to_model_file.pb'
    config_path = 'path_to_config_file.pbtxt'

    # Extract text from the image
    image_text = extract_text_from_image(image_path)
    print("Extracted text from image:")
    print(image_text)

    # Detect objects in the image
    detect_objects(image_path, model_path, config_path)

    # Check plagiarism
    text_to_check = "Some text to check for plagiarism"
    plagiarism_prob = check_plagiarism(text_to_check)
    print(f"Plagiarism probability: {plagiarism_prob}")

# Run the main function
if __name__ == "__main__":
    main()
