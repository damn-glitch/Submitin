import spacy
import numpy as np
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


def preprocess(text):
    doc = nlp(text)
    return " ".join(token.lemma_ for token in doc if not token.is_stop and token.is_alpha)


author_texts_preprocessed = [preprocess(text) for text in author_texts]
other_texts_preprocessed = [preprocess(text) for text in other_texts]

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
    preprocessed = preprocess(text)
    features = vectorizer_loaded.transform([preprocessed]).toarray()
    prob = clf_loaded.predict_proba(features)
    return prob[0][1]  # Probability of being written by the author
