import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np


with open('dataset.json', 'r') as file:
    data = json.load(file)

df = pd.DataFrame(data)

df = df[(df['externalStatus'].str.strip() != '') & (df['internalStatus'].str.strip() != '')]

label_encoder = LabelEncoder()
df['internalStatusEncoded'] = label_encoder.fit_transform(df['internalStatus'])

X_train, X_test, y_train, y_test = train_test_split(df['externalStatus'], df['internalStatusEncoded'], test_size=0.2, random_state=42)


max_words = 1000  
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

max_sequence_length = 100 
pad_sequence = tf.keras.preprocessing.sequence.pad_sequences
X_train_padded = pad_sequence(X_train_sequences, maxlen=max_sequence_length, padding='post', truncating='post')
X_test_padded = pad_sequence(X_test_sequences, maxlen=max_sequence_length, padding='post', truncating='post')


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_words, output_dim=16, input_length=max_sequence_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_padded, y_train, epochs=10, validation_data=(X_test_padded, y_test))

loss, accuracy = model.evaluate(X_test_padded, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


class ExternalStatus(BaseModel):
    external_status: str

class PredictionResult(BaseModel):
    internal_status: str
    
app = FastAPI()

@app.post("/predict/", response_model=PredictionResult)
async def predict_internal_status(external_status: ExternalStatus):
    input_text = [external_status.external_status]
    input_sequences = tokenizer.texts_to_sequences(input_text)
    padded_sequences = pad_sequence(input_sequences, maxlen=max_sequence_length, padding='post', truncating='post')
    
    predictions = model.predict(padded_sequences)
    
    predicted_label_index = np.argmax(predictions, axis=1)[0]
    predicted_internal_status = label_encoder.classes_[predicted_label_index]
    
    return {"internal_status": predicted_internal_status}
