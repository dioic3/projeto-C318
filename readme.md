# C318 Project

Este projeto contém implementações de vários algoritmos de aprendizado de máquina para classificar notícias falsas e verdadeiras. Os algoritmos utilizados incluem XGBoost, SVM, LSTM e GRU.

## Pré-requisitos

Certifique-se de ter o seguinte instalado:
- Python 3.6 ou superior
- pip (instalador de pacotes Python)

## Instalação

1. Clone o repositório:
    ```bash
    git clone https://github.com/yourusername/projeto-C318.git
    cd projeto-C318
    ```

2. Crie um ambiente virtual e ative-o:
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows use `venv\Scripts\activate`
    ```

3. Instale os pacotes necessários:
    ```bash
    pip install -r requirements.txt
    ```

## Conjunto de Dados

Coloque seu conjunto de dados no diretório `./archive/`. Certifique-se de que o arquivo do conjunto de dados seja nomeado `dataset.csv` e contenha as colunas `preprocessed_news` e `label`.

## Executando os Algoritmos

### XGBoost

Para executar o algoritmo XGBoost:
```bash
python xg_boost/xg_boost.py
```

### SVM

Para executar o algoritmo SVM:
```bash
python algoritmo/svm/svm.py
```

### LSTM

Para executar o algoritmo LSTM:
```bash
python algoritmo/lstm/lstm.py
```

### GRU

Para executar o algoritmo GRU:
```bash
python algoritmo/gru/gru.py
```

## Resultados

Os resultados de cada algoritmo serão salvos em seus respectivos diretórios:
- XGBoost: Saída do console
- SVM: `svm.csv`
- LSTM: `algoritmo/lstm/lstm_csv/`, `algoritmo/lstm/lstm_txt/`, `algoritmo/lstm/lstm_plots/`, `algoritmo/lstm/lstm_matriz_confusao/`
- GRU: `algoritmo/gru/gru_csv/`, `algoritmo/gru/gru_txt/`, `algoritmo/gru/gru_plots/`, `algoritmo/gru/gru_matriz_confusao/`

## Licença

Este projeto está licenciado sob a Licença MIT.