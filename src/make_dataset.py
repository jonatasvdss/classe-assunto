import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import preparar_dados_limpos

def processar_datasets(caminho_base, versao):
    pasta_raw = os.path.join(caminho_base, 'data', versao, 'raw')
    pasta_processed = os.path.join(caminho_base, 'data', versao, 'processed')

    os.makedirs(pasta_processed, exist_ok=True)

    nome_arquivo_classes = 'classes_recorrentes.csv'
    nome_arquivo_assuntos = 'assuntos_recorrentes.csv'

    raw_classes = os.path.join(pasta_raw, nome_arquivo_classes)
    raw_assuntos = os.path.join(pasta_raw, nome_arquivo_assuntos)
    
    processed_classes = os.path.join(pasta_processed, 'classes_recorrentes_processado.csv')
    processed_assuntos = os.path.join(pasta_processed, 'assuntos_recorrentes_processado.csv')

    if os.path.exists(raw_classes):
        print(f"[{versao}] Processando dataset de classes...")
        df_classes = preparar_dados_limpos(raw_classes, tipo_alvo="classe", versao=versao)
        df_classes.write_csv(processed_classes)
        print(f"[{versao}] Salvo em: {processed_classes}")
    else:
        print(f"[{versao}] Aviso: Arquivo raw não encontrado em {raw_classes}")

    if os.path.exists(raw_assuntos):
        print(f"[{versao}] Processando dataset de assuntos...")
        df_assuntos = preparar_dados_limpos(raw_assuntos, tipo_alvo="assunto", versao=versao)
        df_assuntos.write_csv(processed_assuntos)
        print(f"[{versao}] Salvo em: {processed_assuntos}")
    else:
        print(f"[{versao}] Aviso: Arquivo raw não encontrado em {raw_assuntos}")

def main():
    caminho_base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    print("=== Pipeline de Processamento: V1 Original (Texto Integral) ===")
    processar_datasets(caminho_base, 'v1_original')
    
    print("\n=== Pipeline de Processamento: V2 FTP (Segmentado) ===")
    processar_datasets(caminho_base, 'v2_ftp')

if __name__ == "__main__":
    main()