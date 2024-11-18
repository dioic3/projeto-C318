# TCC Project

This project contains implementations of various machine learning algorithms for classifying fake and true news. The algorithms used include XGBoost, SVM, LSTM, and GRU.

## Prerequisites

Ensure you have the following installed:
- Python 3.6 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/TCC.git
    cd TCC
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

Place your dataset in the `./archive/` directory. Ensure the dataset file is named `dataset.csv` and contains the columns `preprocessed_news` and `label`.

## Running the Algorithms

### XGBoost

To run the XGBoost algorithm:
```bash
python xg_boost/xg_boost.py
```

### SVM

To run the SVM algorithm:
```bash
python algoritmo/svm/svm.py
```

### LSTM

To run the LSTM algorithm:
```bash
python algoritmo/lstm/lstm.py
```

### GRU

To run the GRU algorithm:
```bash
python algoritmo/gru/gru.py
```

## Results

The results for each algorithm will be saved in their respective directories:
- XGBoost: Console output
- SVM: `svm.csv`
- LSTM: `algoritmo/lstm/lstm_csv/`, `algoritmo/lstm/lstm_txt/`, `algoritmo/lstm/lstm_plots/`, `algoritmo/lstm/lstm_matriz_confusao/`
- GRU: `algoritmo/gru/gru_csv/`, `algoritmo/gru/gru_txt/`, `algoritmo/gru/gru_plots/`, `algoritmo/gru/gru_matriz_confusao/`

## License

This project is licensed under the MIT License.
```

This README file provides instructions on how to set up the environment, run the different algorithms, and where to find the results. Adjust the repository URL and any other specific details as necessary.