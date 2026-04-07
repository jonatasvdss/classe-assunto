import polars as pl
import re
import unicodedata

def limpar_texto_peticao(texto):
    if not texto:
        return ""
    texto = str(texto).lower()
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
    texto = re.sub(r'[^a-z0-9\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()

def unificar_classes_frequentes(df, coluna_alvo="classe"):
    mapa_unificacao = {
        "Procedimento do Juizado Especial Cível": "Procedimento Comum",
        "Procedimento do Juizado Especial da Fazenda Pública": "Procedimento Comum"
    }
    
    if coluna_alvo in df.columns:
        df = df.with_columns(
            pl.col(coluna_alvo).replace(mapa_unificacao, default=pl.col(coluna_alvo))
        )
    return df

def preparar_dados_limpos(caminho_csv, tipo_alvo="classe", versao="v1_original"):
    df = pl.read_csv(caminho_csv, separator="#")

    if tipo_alvo == "classe":
        df = unificar_classes_frequentes(df, "classe")

    colunas_esperadas = ["inteiro_teor", "fato", "tese", "pedido"]
    for col in colunas_esperadas:
        if col not in df.columns:
            df = df.with_columns(pl.lit("").alias(col))
    
    df = df.with_columns([pl.col(c).fill_null("") for c in colunas_esperadas])

    if versao == "v2_ftp":
        if tipo_alvo == "classe":
            df = df.with_columns(
                pl.struct(colunas_esperadas).map_elements(
                    lambda row: row["inteiro_teor"].replace(row["fato"], "").replace(row["tese"], "").replace(row["pedido"], ""),
                    return_dtype=pl.String
                ).alias("texto_bruto")
            )
        elif tipo_alvo == "assunto":
            df = df.with_columns(pl.col("fato").alias("texto_bruto"))
    else:
        df = df.with_columns(pl.col("inteiro_teor").alias("texto_bruto"))

    df = df.with_columns(
        pl.col("texto_bruto").map_elements(limpar_texto_peticao, return_dtype=pl.String).alias("texto_limpo")
    )

    df = df.filter(pl.col("texto_limpo").str.len_chars() > 0)
    df = df.select(["texto_limpo", tipo_alvo])
    
    return df