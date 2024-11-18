import os
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Embedding, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

def create_directory(directory):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)

def load_dataset(filepath):
    """Load dataset from a CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No such file or directory: '{filepath}'")
    return pd.read_csv(filepath)

def preprocess_data(df, max_words, max_sequence_len):
    """Preprocess text data and tokenize."""
    texts = df['preprocessed_news'].astype(str).values
    labels = df['label'].values
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=max_sequence_len)
    return X, labels, tokenizer

def build_model(max_words, max_sequence_len):
    """Build and compile the GRU model."""
    model = Sequential([
        Embedding(max_words, 128, input_length=max_sequence_len),
        SpatialDropout1D(0.2),
        GRU(128, return_sequences=False),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

def plot_confusion_matrix(cm, epochs, directory):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Matriz confusão - Épocas: {epochs}')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.savefig(f'{directory}/confusion_matrix_{epochs}.png')
    plt.close()

def plot_final_confusion_matrix(y_true, y_pred, directory):
    """Plot and save the final confusion matrix for all epochs."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Matriz de Confusão Final')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.savefig(f'{directory}/final_confusion_matrix.png')
    plt.close()

def plot_metrics(epochs_list, accuracy_epochs, precision_epochs, recall_epochs, f1_epochs, output_dir):
    """Plot and save all metrics in a single plot."""
    plt.figure(figsize=(12, 8))
    plt.plot(epochs_list, accuracy_epochs, 'b', label='Acurácia')
    plt.plot(epochs_list, precision_epochs, 'g', label='Precisão')
    plt.plot(epochs_list, recall_epochs, 'r', label='Recall')
    plt.plot(epochs_list, f1_epochs, 'm', label='F1-Score')
    
    plt.xlabel('Épocas')
    plt.ylabel('Métricas')
    plt.title('Métricas vs Épocas')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'epochs_metrics_plot.png'))
    plt.show()

def calculate_execution_time(start_time, end_time):
    """Calculate and print the execution time of the code."""
    execution_time = end_time - start_time
    hours, rem = divmod(execution_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Tempo de execução: {int(hours):02}:{int(minutes):02}:{seconds:05.2f}")

def save_results_to_csv(results, filepath):
    """Save results to a CSV file."""
    df = pd.DataFrame(results)
    df.to_csv(filepath, index=False)

def calculate_batch_size(training_size, desired_batches):
    """Calculate the batch size to achieve the desired number of batches per epoch."""
    return int(np.ceil(training_size / desired_batches))

# Exemplo de uso das funções
if __name__ == "__main__":
    start_time = time.time()
    
    # Definir parâmetros
    max_words = 5000
    max_sequence_len = 200
    epochs_list = list(range(5, 60, 5))
    results_dir = 'gru_results'
    testes = 20  # Alterado para 20
    saves = 8
    desired_batches = 128  # Número desejado de batches por época
    
    # Criar diretório para resultados
    create_directory(results_dir)
    
    # Carregar e preprocessar dados
    dataset_path = '../../archive/dataset.csv'
    try:
        df = load_dataset(dataset_path)
    except FileNotFoundError as e:
        print(e)
        exit(1)
    
    X, labels, tokenizer = preprocess_data(df, max_words, max_sequence_len)
    
    # Dividir dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    
    # Calcular o tamanho do batch para obter o número desejado de batches por época
    batch_size = calculate_batch_size(len(X_train), desired_batches)
    
    # Inicializar listas para armazenar métricas por épocas
    accuracy_epochs = []
    precision_epochs = []
    recall_epochs = []
    f1_epochs = []

    # Lista para armazenar resultados
    all_results = []
    final_y_true = []
    final_y_pred = []

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Arquivo CSV para as médias
    csv_file_path = f'algoritmo/gru/gru_csv/gru_metrics_{saves}.csv'
    create_directory(os.path.dirname(csv_file_path))
    with open(csv_file_path, "w") as csv_file:
        csv_file.write('Épocas,Média Acurácia,Média Precisão,Média Recall,Média F1-Score\n')

    # Arquivo TXT para salvar resultados detalhados
    txt_file_path = f"algoritmo/gru/gru_txt/gru_results_{saves}.txt"
    create_directory(os.path.dirname(txt_file_path))
    with open(txt_file_path, "a") as file:

        # Loop sobre as épocas
        for epocas in epochs_list:
            accuracy_list, precision_list, recall_list, f1_list = [], [], [], []
            cm_acumulada = np.zeros((2, 2), dtype=int)

            avg_accuracy = avg_precision = avg_recall = avg_f1 = total_time = 0

            # Loop para múltiplos testes
            for i in range(testes):
                print(f"Executando teste {i + 1}/{testes} para {epocas} épocas")
                model = build_model(max_words, max_sequence_len)
                history = model.fit(X_train, y_train, epochs=epocas, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])
                
                y_pred = (model.predict(X_test) > 0.5).astype("int32")
                accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)

                # Salvar resultados
                accuracy_list.append(accuracy)
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)
                cm_acumulada += cm

                # Salvar no arquivo texto
                file.write(f"Épocas: {epocas}, Repetição: {i+1}\nAcurácia: {accuracy:.3f}\nPrecisão: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}\n\n")

            # Cálculo das médias
            avg_accuracy = np.mean(accuracy_list)
            avg_precision = np.mean(precision_list)
            avg_recall = np.mean(recall_list)
            avg_f1 = np.mean(f1_list)

            # Adicionar às listas de épocas
            accuracy_epochs.append(avg_accuracy)
            precision_epochs.append(avg_precision)
            recall_epochs.append(avg_recall)
            f1_epochs.append(avg_f1)

            # Salvar as médias no CSV
            with open(csv_file_path, "a") as csv_file:
                csv_file.write(f"{epocas},{avg_accuracy:.3f},{avg_precision:.3f},{avg_recall:.3f},{avg_f1:.3f}\n")

            # Salvar as médias no arquivo texto
            file.write(f"Épocas: {epocas}\nMédia Acurácia: {avg_accuracy:.3f}\nMédia Precisão: {avg_precision:.3f}\nMédia Recall: {avg_recall:.3f}\nMédia F1-Score: {avg_f1:.3f}\nTempo total: {total_time:.2f} min\n---------------------------------\n")

            # Plotar matriz de confusão média
            cm_media = cm_acumulada / testes
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_media, annot=True, cmap='Blues', fmt=".0f", cbar=False)
            plt.title(f'Média da Matriz de Confusão - Épocas: {epocas}')
            plt.xlabel('Predito')
            plt.ylabel('Real')
            confusion_matrix_dir = 'algoritmo/gru/gru_matriz_confusao'
            create_directory(confusion_matrix_dir)
            plt.savefig(f'{confusion_matrix_dir}/confusion_matrix_{epocas}.png')

    # Garantir que o diretório exista
    output_dir = 'algoritmo/gru/gru_plots'
    os.makedirs(output_dir, exist_ok=True)

    # Gráfico das métricas por épocas
    plot_metrics(epochs_list, accuracy_epochs, precision_epochs, recall_epochs, f1_epochs, output_dir)

    end_time = time.time()
    calculate_execution_time(start_time, end_time)