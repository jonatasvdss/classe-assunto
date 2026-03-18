import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import preparar_dados_limpos

def main():
    caminho_base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    pasta_raw = os.path.join(caminho_base, 'data', 'raw')
    pasta_processed = os.path.join(caminho_base, 'data', 'processed')

    os.makedirs(pasta_processed, exist_ok=True)

    arquivo_raw_classes = os.path.join(pasta_raw, 'amostra_processos_classes_recorrentes_27022025.csv')
    df_classes_processado = preparar_dados_limpos(arquivo_raw_classes, tipo_alvo="classe")
    
    arquivo_processed_classes = os.path.join(pasta_processed, 'dataset_classes_processado.csv')
    df_classes_processado.write_csv(arquivo_processed_classes)
    print(f"Dataset de classes salvo em: {arquivo_processed_classes}")

    arquivo_raw_assuntos = os.path.join(pasta_raw, 'amostra_processos_assuntos_recorrentes_27022025.csv')
    df_assuntos_processado = preparar_dados_limpos(arquivo_raw_assuntos, tipo_alvo="assunto")
    
    arquivo_processed_assuntos = os.path.join(pasta_processed, 'dataset_assuntos_processado.csv')
    df_assuntos_processado.write_csv(arquivo_processed_assuntos)
    print(f"Dataset de assuntos salvo em: {arquivo_processed_assuntos}")

if __name__ == "__main__":
    main()