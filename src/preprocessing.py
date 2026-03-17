import polars as pl
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

def limpar_texto_peticao(df: pl.DataFrame, coluna_origem: str = "inteiro_teor", coluna_destino: str = "texto_limpo") -> pl.DataFrame:
    """Limpa ruídos, formatações, caracteres especiais e STOPWORDS do texto jurídico."""
    
    stop_words_pt = stopwords.words('portuguese')
    padrao_stopwords = r'\b(' + '|'.join(stop_words_pt) + r')\b'
    
    return df.with_columns(
        pl.col(coluna_origem)
        .str.to_lowercase()
        .str.replace_all(r">>>>>inicio<<<<<", " ")
        .str.replace_all(r"[\n\r\t]", " ")
        .str.replace_all(r"[^a-záéíóúâêôãõç\s]", " ")
        .str.replace_all(padrao_stopwords, " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias(coluna_destino)
    )

def unificar_classes_frequentes(df: pl.DataFrame, coluna_alvo: str = "classe") -> pl.DataFrame:
    """Unifica classes com nomenclaturas longas ou redundantes."""
    classes_alvo = [
        "Procedimento do Juizado Especial Cível",
        "Procedimento Comum",
        "Execução Fiscal",
        "Execução de Título Extrajudicial ( L.E. )",
        "Agravo de Instrumento ( CPC )",
        "Carta Precatória Cível"
    ]
    padrao_regex = f"({'|'.join(re.escape(c) for c in classes_alvo)})"
    
    return df.with_columns(
        pl.coalesce(
            pl.col(coluna_alvo).str.extract(padrao_regex, 1),
            pl.col(coluna_alvo)
        ).alias(coluna_alvo)
    )

def preparar_dados_limpos(caminho_csv: str, tipo_alvo: str = "classe") -> pl.DataFrame:
    df = pl.read_csv(caminho_csv, separator="#")
    
    if tipo_alvo == "classe":
        if "classe" in df.columns:
            df = unificar_classes_frequentes(df)
    elif tipo_alvo == "assunto":
        pass
        
    df = limpar_texto_peticao(df)
    
    coluna_alvo = "classe" if tipo_alvo == "classe" else "assunto"
    
    df = df.filter(
        pl.col(coluna_alvo).is_not_null() & 
        (pl.col("texto_limpo").str.len_chars() > 0)
    )
    
    return df