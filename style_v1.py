import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

# 1. Load and preprocess the dataset
data = pd.read_csv('author_text_samples.csv')  # Replace with your dataset file
data['text'] = data['text'].apply(lambda x: x.lower())  # Convert to lowercase

# 2. Tokenize text and create sequences
max_features = 10000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['text'].values)
sequences = tokenizer.texts_to_sequences(data['text'].values)
padded_sequences = pad_sequences(sequences)

# 3. Encode author labels
label_encoder = LabelEncoder()
integer_labels = label_encoder.fit_transform(data['author'])

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, integer_labels, test_size=0.2)

# 5. Define the model
embedding_dim = 128
lstm_units = 64

model = Sequential()
model.add(Embedding(max_features, embedding_dim, input_length=padded_sequences.shape[1]))
model.add(LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(len(np.unique(integer_labels)), activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 6. Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# 7. Test the model
score, accuracy = model.evaluate(X_test, y_test, batch_size=32)

# 8. Save the model for future use
model.save('author_style_model.h5')