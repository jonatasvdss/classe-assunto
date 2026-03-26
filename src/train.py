import sys
import os
import polars as pl
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train_models import extrair_features_e_dividir, salvar_modelo_e_vetorizador

def treinar_pipeline(caminho_dados, alvo):
    df = pl.read_csv(caminho_dados)
    
    X_treino, _, y_treino, _, vetorizador = extrair_features_e_dividir(
        df, 
        coluna_texto="texto_limpo", 
        coluna_alvo=alvo
    )
    
    print(f"Treinando Naive Bayes ({alvo})...")
    modelo_nb = MultinomialNB()
    modelo_nb.fit(X_treino, y_treino)
    salvar_modelo_e_vetorizador(modelo_nb, vetorizador, f"modelo_naive_bayes_{alvo}", f"vetorizador_tfidf_{alvo}")
    
    print(f"Treinando Random Forest ({alvo})...")
    modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    modelo_rf.fit(X_treino, y_treino)
    salvar_modelo_e_vetorizador(modelo_rf, vetorizador, f"modelo_random_forest_{alvo}", f"vetorizador_tfidf_{alvo}")
    
    print(f"Treinando Regressão Logística ({alvo})...")
    modelo_lr = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42, class_weight='balanced', C=2.0)
    modelo_lr.fit(X_treino, y_treino)
    salvar_modelo_e_vetorizador(modelo_lr, vetorizador, f"modelo_regressao_logistica_{alvo}", f"vetorizador_tfidf_{alvo}")
    
    if alvo == "assunto":
        print(f"Treinando LinearSVC ({alvo})...")
        modelo_svc = LinearSVC(random_state=42, class_weight='balanced', dual=False, max_iter=1000)
        modelo_svc.fit(X_treino, y_treino)
        salvar_modelo_e_vetorizador(modelo_svc, vetorizador, f"modelo_linearsvc_{alvo}", f"vetorizador_tfidf_{alvo}")

def main():
    caminho_base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    caminho_classes = os.path.join(caminho_base, 'data', 'processed', 'dataset_classes_processado.csv')
    print("Iniciando pipeline para CLASSES...")
    treinar_pipeline(caminho_classes, "classe")
    
    caminho_assuntos = os.path.join(caminho_base, 'data', 'processed', 'dataset_assuntos_processado.csv')
    print("\nIniciando pipeline para ASSUNTOS...")
    treinar_pipeline(caminho_assuntos, "assunto")
    
    print("\nTreinamento de todos os modelos concluído com sucesso.")

if __name__ == "__main__":
    main()