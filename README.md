# Classificação Automática de Documentos Jurídicos

Este projeto tem como objetivo construir, treinar e avaliar modelos de Machine Learning para classificar documentos jurídicos. O pipeline foi desenvolvido para prever duas dimensões processuais distintas: a **Classe** e o **Assunto**.

Após otimizações recentes, o projeto passou a contar com o versionamento dos dados em duas abordagens para redução de ruído: o uso do **Texto Integral (v1_original)** e a **Segmentação Fato/Tese/Pedido (v2_ftp)**, onde as classes são extraídas logicamente pelo cabeçalho da petição e os assuntos filtrados pela seção de fatos.

## Estrutura do Projeto

A organização dos diretórios e arquivos segue uma arquitetura modular para facilitar o versionamento e a reprodutibilidade dos experimentos.

* `data/`:
    * `v1_original/`: Bases originais utilizando o inteiro teor das petições (`raw/` e `processed/`).
    * `v2_ftp/`: Bases estruturadas com segmentação explícita de fato, tese e pedido (`raw/` e `processed/`).
* `models/`: Artefatos `.joblib` (modelos, vetorizadores e label encoders) separados em subpastas correspondentes ao versionamento (`v1_original/` e `v2_ftp/`).
* `imgs/`: Exportação das matrizes de confusão geradas nas avaliações, também versionadas.
* `src/`:
    * `preprocessing.py`: Funções de limpeza de texto, unificação de classes via regex e isolamento de target (cabeçalho vs. fato).
    * `make_dataset.py`: Script de automação que itera sobre as versões e gera os datasets higienizados.
    * `train_models.py`: Funções auxiliares de vetorização (TF-IDF) e salvamento de artefatos.
    * `train.py`: Pipeline principal de treinamento que itera sobre as bases (v1 e v2) registrando logs de execução.
    * `evaluate.py`: Geração e registro de métricas de avaliação.
    * `predict.py`: Pipeline de inferência para novos textos.
* `Notebooks Jupyter`:
    * `01_eda_classes.ipynb`: Análise exploratória, distribuição, volumetria e unificação de Classes.
    * `analise_exploratoria_assuntos.ipynb`: Análise exploratória, distribuição de categorias e volumetria de Assuntos.
    * `experimentacao_classes.ipynb` / `experimentacao_assuntos.ipynb`: Ambientes de experimentação e comparação de algoritmos.

## Stack Tecnológico

* Python
* Polars e Pandas
* Scikit-learn
* XGBoost e LightGBM
* Matplotlib e Seaborn

## Abordagem de Modelagem

O processamento de linguagem natural (NLP) neste projeto utiliza vetorização baseada em frequência com TF-IDF (Term Frequency-Inverse Document Frequency) para transformar o texto limpo das petições em matrizes numéricas esparsas (limitadas a 15.000 features de n-gramas). 

Foram conduzidos experimentos com os seguintes algoritmos para identificar o melhor desempenho focado na métrica F1-Score macro:

* LinearSVC
* Logistic Regression
* Multinomial Naive Bayes
* Random Forest Classifier
* LightGBM
* XGBoost

## Como Executar

Clone o repositório, configure seu ambiente virtual isolado e execute os pipelines de preparação e treinamento.

```bash
# Configuração do ambiente
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install xgboost lightgbm polars

# Geração das bases limpas (V1 e V2)
python src/make_dataset.py

# Treinamento massivo dos modelos e vetorizadores
python src/train.py
```