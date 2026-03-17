import joblib
import os
import polars as pl
from src.preprocessing import limpar_texto_peticao

def classificar_nova_peticao(texto_peticao: str):
    caminho_pasta = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    
    caminho_modelo_classe = os.path.join(caminho_pasta, 'modelo_regressao_logistica_classe.joblib')
    caminho_vetor_classe = os.path.join(caminho_pasta, 'vetorizador_tfidf_classe.joblib')
    
    caminho_modelo_assunto = os.path.join(caminho_pasta, 'modelo_regressao_logistica_assunto.joblib')
    caminho_vetor_assunto = os.path.join(caminho_pasta, 'vetorizador_tfidf_assunto.joblib')
    
    modelo_classe = joblib.load(caminho_modelo_classe)
    vetorizador_classe = joblib.load(caminho_vetor_classe)
    
    modelo_assunto = joblib.load(caminho_modelo_assunto)
    vetorizador_assunto = joblib.load(caminho_vetor_assunto)
    
    df_novo = pl.DataFrame({"inteiro_teor": [texto_peticao]})
    df_limpo = limpar_texto_peticao(df_novo)
    
    texto_processado = df_limpo["texto_limpo"].to_list()
    
    if not texto_processado[0].strip():
        raise ValueError("O texto da peticao ficou vazio apos a limpeza.")
        
    texto_vetorizado_classe = vetorizador_classe.transform(texto_processado)
    predicao_classe = modelo_classe.predict(texto_vetorizado_classe)[0]
    
    texto_vetorizado_assunto = vetorizador_assunto.transform(texto_processado)
    predicao_assunto = modelo_assunto.predict(texto_vetorizado_assunto)[0]
    
    return {
        "classe": predicao_classe,
        "assunto": predicao_assunto
    }