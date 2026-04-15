import sys
import os
import joblib
import polars as pl
import logging
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train_models import extrair_features_e_dividir, salvar_modelo_e_vetorizador

def treinar_pipeline(caminho_dados, alvo, subpasta_modelo):
    logging.info(f"Iniciando pipeline para {alvo.upper()} na pasta {subpasta_modelo}...")
    
    logging.info("Carregando dados...")
    df = pl.read_csv(caminho_dados)
    
    logging.info("Extraindo features e dividindo base...")
    X_treino, X_teste, y_treino, y_teste, vetorizador = extrair_features_e_dividir(
        df, 
        coluna_texto="texto_limpo", 
        coluna_alvo=alvo
    )
    
    logging.info("Treinando Naive Bayes...")
    modelo_nb = MultinomialNB()
    modelo_nb.fit(X_treino, y_treino)
    salvar_modelo_e_vetorizador(modelo_nb, vetorizador, f"modelo_naive_bayes_{alvo}", f"vetorizador_tfidf_{alvo}", subpasta_modelo)
    
    logging.info("Treinando Random Forest...")
    modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    modelo_rf.fit(X_treino, y_treino)
    salvar_modelo_e_vetorizador(modelo_rf, vetorizador, f"modelo_random_forest_{alvo}", f"vetorizador_tfidf_{alvo}", subpasta_modelo)
    
    logging.info("Treinando Regressão Logística...")
    modelo_lr = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42, class_weight='balanced', C=2.0)
    modelo_lr.fit(X_treino, y_treino)
    salvar_modelo_e_vetorizador(modelo_lr, vetorizador, f"modelo_regressao_logistica_{alvo}", f"vetorizador_tfidf_{alvo}", subpasta_modelo)
    
    if alvo == "assunto":
        logging.info("Treinando LinearSVC...")
        modelo_svc = LinearSVC(random_state=42, class_weight='balanced', dual=False, max_iter=1000)
        modelo_svc.fit(X_treino, y_treino)
        salvar_modelo_e_vetorizador(modelo_svc, vetorizador, f"modelo_linearsvc_{alvo}", f"vetorizador_tfidf_{alvo}", subpasta_modelo)

    logging.info("Aplicando LabelEncoder...")
    le = LabelEncoder()
    y_treino_enc = le.fit_transform(y_treino)
    
    caminho_pasta = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', subpasta_modelo))
    os.makedirs(caminho_pasta, exist_ok=True)
    joblib.dump(le, os.path.join(caminho_pasta, f'label_encoder_{alvo}.joblib'))
    
    logging.info("Treinando LightGBM...")
    modelo_lgbm = LGBMClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
        verbosity=-1,
        max_depth=15
    )
    modelo_lgbm.fit(X_treino, y_treino_enc)
    salvar_modelo_e_vetorizador(modelo_lgbm, vetorizador, f"modelo_lightgbm_{alvo}", f"vetorizador_tfidf_{alvo}", subpasta_modelo)
    
    logging.info("Treinando XGBoost...")
    modelo_xgb = XGBClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
        verbosity=0,
        max_depth=15,
        tree_method='hist'
    )
    modelo_xgb.fit(X_treino, y_treino_enc)
    salvar_modelo_e_vetorizador(modelo_xgb, vetorizador, f"modelo_xgboost_{alvo}", f"vetorizador_tfidf_{alvo}", subpasta_modelo)

    logging.info(f"Pipeline para {alvo.upper()} concluído.\n")

def main():
    caminho_base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    versoes = ["v1_original", "v2_ftp"]
    
    for versao in versoes:
        caminho_classes = os.path.join(caminho_base, 'data', versao, 'processed', 'classes_recorrentes_processado.csv')
        if os.path.exists(caminho_classes):
            treinar_pipeline(caminho_classes, "classe", versao)
            
        caminho_assuntos = os.path.join(caminho_base, 'data', versao, 'processed', 'assuntos_recorrentes_processado.csv')
        if os.path.exists(caminho_assuntos):
            treinar_pipeline(caminho_assuntos, "assunto", versao)

if __name__ == "__main__":
    main()