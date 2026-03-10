import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def extrair_features_e_dividir(df: pl.DataFrame, coluna_texto: str = "texto_limpo", coluna_alvo: str = "classe", max_features: int = 10000):
    """
    Vetoriza o texto usando TF-IDF e divide os dados em treino e teste.
    Retorna: X_treino, X_teste, y_treino, y_teste, vetorizador
    """
    X = df[coluna_texto].to_list()
    y = df[coluna_alvo].to_list()
    
    vetorizador = TfidfVectorizer(max_features=max_features)
    X_vetorizado = vetorizador.fit_transform(X)
    
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X_vetorizado, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_treino, X_teste, y_treino, y_teste, vetorizador