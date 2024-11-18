from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt

# Carregar os datasets de fake e true news
df = pd.read_csv('./archive/dataset.csv')

# Pre-processamento
texts = df['preprocessed_news'].astype(str).values  # Converter o conteúdo de texto em strings
labels = df['label'].values  # Usar a coluna de rótulo

# Vetorizar os dados com TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(texts)

# Parâmetros para otimização do SVM
param_dist = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
}

# DataFrame para armazenar os resultados
results = pd.DataFrame(columns=['Algoritmo', 'Accuracy', 'F1-Score', 'Precision', 'Recall'])

# Loop para rodar o modelo 10 vezes
for i in range(10):
    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=i)

    # Configurar o RandomizedSearchCV para o SVM
    svm_model = SVC()
    random_search = RandomizedSearchCV(estimator=svm_model, param_distributions=param_dist, scoring='accuracy', cv=3, verbose=0, n_jobs=-1, n_iter=10)
    random_search.fit(X_train, y_train)

    # Melhor modelo encontrado
    best_svm = random_search.best_estimator_

    # Fazer predições e avaliar o SVM otimizado
    y_pred_svm = best_svm.predict(X_test)
    
    # Calcular métricas de avaliação
    accuracy = accuracy_score(y_test, y_pred_svm)
    f1 = f1_score(y_test, y_pred_svm)
    precision = precision_score(y_test, y_pred_svm)
    recall = recall_score(y_test, y_pred_svm)
    
    # Adicionar os resultados ao DataFrame
    results = results.append({
        'Algoritmo': 'SVM',
        'Accuracy': accuracy,
        'F1-Score': f1,
        'Precision': precision,
        'Recall': recall
    }, ignore_index=True)

# Salvar o DataFrame em um arquivo CSV
results.to_csv('svm.csv', index=False)
print("Resultados salvos em 'svm.csv'")
