# Classificação Automática de Documentos Jurídicos

Este projeto tem como objetivo construir, treinar e avaliar modelos de Machine Learning clássico para classificar documentos jurídicos com base no seu texto (inteiro teor). O pipeline foi desenvolvido para prever duas dimensões processuais distintas: a **Classe** e o **Assunto**.

## Estrutura do Projeto

A organização dos diretórios e arquivos segue uma arquitetura modular para facilitar a manutenção e a reprodutibilidade dos experimentos.

* `data/raw/`: Bases de dados originais.
* `data/processed/`: Datasets higienizados e padronizados, consumidos pelos modelos.
* `imgs/`: Exportação das matrizes de confusão geradas nas avaliações.
* `src/`:
    * `preprocessing.py`: Funções de limpeza de texto.
    * `make_dataset.py`: Script de automação que processa os dados brutos e gera a base final.
    * `train_models.py`: Funções de extração de features textuais e treinamento de modelos.
    * `evaluate.py`: Geração e registro de métricas de avaliação.
    * `predict.py`: Pipeline de inferência para novos textos.
* `Notebooks Jupyter`:
    * `analise_exploratoria_assuntos.ipynb`: Análise exploratória, distribuição de categorias e volumetria. Obs.: a EDA de classes foi perdida
    * `experimentacao_classes.ipynb` / `experimentacao_assuntos.ipynb`: Ambientes de experimentação, vetorização e comparação de algoritmos.

## Stack Tecnológico

* Python
* Polars e Pandas
* Scikit-learn
* Matplotlib e Seaborn

## Abordagem de Modelagem

O processamento de linguagem natural (NLP) neste projeto utiliza vetorização baseada em frequência com TF-IDF (Term Frequency-Inverse Document Frequency) para transformar o texto das petições em matrizes numéricas esparsas. 

Foram conduzidos experimentos com os seguintes algoritmos para identificar o melhor desempenho focado na métrica F1-Score macro:

* Random Forest Classifier
* Logistic Regression
* Multinomial Naive Bayes
* LinearSVC

## Como Executar

Clone o repositório e configure seu ambiente virtual isolado.

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt