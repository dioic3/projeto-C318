import os
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

# Criar diretórios necessários
os.makedirs('lstm/confusion_matriz', exist_ok=True)

# Carregar o dataset
df = pd.read_csv('./archive/dataset.csv')
texts = df['preprocessed_news'].astype(str).values
labels = df['label'].values

# Tokenização e padding
max_words = 5000
max_sequence_len = 200
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=max_sequence_len)

# Dividir os dados
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Parâmetros
epochs_list = list(range(5, 20, 5))
testes = 15

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Loop sobre as épocas
for epocas in epochs_list:
    cm_acumulada = np.zeros((2, 2), dtype=int)

    # Loop para múltiplos testes
    for i in range(testes):
        # Modelo LSTM
        model = Sequential([
            Embedding(max_words, 128, input_length=max_sequence_len),
            SpatialDropout1D(0.2),
            LSTM(128, return_sequences=False),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

        # Treinamento do modelo
        model.fit(X_train, y_train, epochs=epocas, batch_size=128, validation_data=(X_test, y_test), verbose=0, callbacks=[early_stopping])

        # Previsões
        y_pred = (model.predict(X_test) > 0.5).astype("int32")

        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        cm_acumulada += cm

    # Plotar matriz de confusão média
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_acumulada, annot=True, cmap='Blues', fmt=".0f", cbar=False)
    plt.title(f'Média da Matriz de Confusão - Épocas: {epocas}')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.savefig(f'lstm/confusion_matriz/confusion_matrix_{epocas}.png')
    plt.close()
