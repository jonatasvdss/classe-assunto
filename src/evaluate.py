import os
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def avaliar_modelo(modelo, X_teste, y_teste, nome_modelo="Modelo"):
    y_pred = modelo.predict(X_teste)
    
    print("\n" + "="*60)
    print(f"RELATÓRIO DE CLASSIFICAÇÃO - {nome_modelo.upper()}")
    print("="*60)
    print(classification_report(y_teste, y_pred))
    
    fig, ax = plt.subplots(figsize=(16, 14))
    
    ConfusionMatrixDisplay.from_predictions(
        y_teste, y_pred, cmap=plt.cm.Blues, xticks_rotation='vertical', ax=ax
    )
    
    plt.title(f"Matriz de Confusão - {nome_modelo}", fontsize=14, pad=20)
    
    caminho_pasta = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'imgs'))
    os.makedirs(caminho_pasta, exist_ok=True)
    
    nome_arquivo = f"matriz_confusao_{nome_modelo.lower().replace(' ', '_')}.png"
    caminho_completo = os.path.join(caminho_pasta, nome_arquivo)
    
    plt.savefig(caminho_completo, dpi=300, bbox_inches='tight')
    print(f"Matriz de confusão salva em: {caminho_completo}")
    
    plt.close(fig)