import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import os

def extrair_features_e_dividir(df: pl.DataFrame, coluna_texto: str = "texto_limpo", coluna_alvo: str = "classe", max_features: int = 15000):
    X = df[coluna_texto].to_list()
    y = df[coluna_alvo].to_list()
    
    vetorizador = TfidfVectorizer(
        max_features=max_features, 
        ngram_range=(1, 2), 
        min_df=5, 
        max_df=0.8
    )
    X_vetorizado = vetorizador.fit_transform(X)
    
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X_vetorizado, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_treino, X_teste, y_treino, y_teste, vetorizador

def salvar_modelo_e_vetorizador(modelo, vetorizador, nome_modelo: str, nome_vetorizador: str = "vetorizador_tfidf"):
    caminho_pasta = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    os.makedirs(caminho_pasta, exist_ok=True)
    
    joblib.dump(modelo, os.path.join(caminho_pasta, f'{nome_modelo}.joblib'))
    joblib.dump(vetorizador, os.path.join(caminho_pasta, f'{nome_vetorizador}.joblib'))