import joblib
import os
import polars as pl
from src.preprocessing import limpar_texto_peticao

def classificar_nova_peticao(texto_peticao: str, nome_modelo: str = 'modelo_regressao_logistica'):
    caminho_pasta = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    caminho_modelo = os.path.join(caminho_pasta, f'{nome_modelo}.joblib')
    caminho_vetorizador = os.path.join(caminho_pasta, 'vetorizador_tfidf.joblib')
    
    modelo = joblib.load(caminho_modelo)
    vetorizador = joblib.load(caminho_vetorizador)
    
    df_novo = pl.DataFrame({"inteiro_teor": [texto_peticao]})
    df_limpo = limpar_texto_peticao(df_novo)
    
    texto_processado = df_limpo["texto_limpo"].to_list()
    
    if not texto_processado[0].strip():
        raise ValueError("O texto da petição ficou vazio após a limpeza.")
        
    texto_vetorizado = vetorizador.transform(texto_processado)
    predicao = modelo.predict(texto_vetorizado)
    
    return predicao[0]