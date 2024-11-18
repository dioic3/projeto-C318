import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score

# Carregar os datasets de fake e true news
df = pd.read_csv('./archive/dataset.csv')

# Pré-processamento (garante que o modelo receba dados limpos)
texts = df['preprocessed_news'].astype(str).values  # Converter o conteúdo de texto em strings
labels = df['label'].values  # Usar a coluna de rótulo

# Vetorizar/Tokenizar os dados com TF-IDF (Essa técnica transforma o conjunto de textos em representações numéricas que o modelo pode entender, facilitando a análise e a classificação)
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(texts)

# Divisão dos dados em treino e teste (garante que os modelos sejam avaliados de maneira eficaz)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Definir o modelo XGBoost
xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=62, tree_method='hist')  

# Definir a grid de parâmetros para o RandomizedSearchCV (facilita a busca por configurações ótimas, aumentando a eficácia e robustez do modelo.)
param_dist = {
    'n_estimators': [50, 100, 150, 200, 250, 300], 
    'learning_rate': [0.01, 0.1], 
    'max_depth': [3, 4],  
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9] 
}

# Configurar o RandomizedSearchCV (para otimizar o desempenho do XGboost)
random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, scoring='accuracy', cv=3, verbose=2, n_jobs=-1, n_iter=10)

# Treinar o modelo com RandomizedSearchCV
random_search.fit(X_train, y_train)

# Melhor combinação de hiperparâmetros
print("Melhores parâmetros encontrados:", random_search.best_params_)

# Prever com o melhor modelo
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)

# Extraindo os valores da matriz de confusão
VN, FP, FN, VP = conf_matrix.ravel()

# Avaliar o modelo
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Precisão:", precision_score(y_test, y_pred))
#print("Relatório de Classificação do XGBoost:")
#print(classification_report(y_test, y_pred))
print("Matriz de Confusão:")
print(conf_matrix)
print("Verdadeiros Positivos (VP):", VP)
print("Falsos Negativos (FN):", FN)
print("Falsos Positivos (FP):", FP)
print("Verdadeiros Negativos (VN):", VN)




